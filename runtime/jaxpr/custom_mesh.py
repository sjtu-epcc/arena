#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
A script related to the customized virtual and physical device mesh.
This script is modified from alpa/device_mesh. 
"""

from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict, namedtuple
from collections.abc import Iterable
import logging
from operator import attrgetter
import os
import pickle
import shutil
import threading
import time
from typing import Any, List, Union, Sequence, Tuple, Optional

from jax import core, xla, device_put
from jax._src.api import ShapeDtypeStruct
from jax._src.lib import xla_bridge as xb, xla_extension as xe
from jax._src.tree_util import tree_leaves
from jax.abstract_arrays import array_types
from jax.core import ShapedArray
from jax.interpreters import pxla
from jax.interpreters.pxla import (ShardingSpec, _hashable_index,
                                   ShardedDeviceArray, Index)
from jax.lib import xla_client
import jax.numpy as jnp
import numpy as np
import ray
from ray.util.placement_group import remove_placement_group

from alpa import mesh_profiling
import alpa.collective as col
from alpa.global_env import global_config
from alpa.monkey_patch import set_override_backend
from alpa.device_mesh import (
    DaemonMoveWorker, PhysicalDeviceMesh, DistributedArray, ReplicatedDistributedArray,
    device_id_to_str, _shard_array, shard_arg_handlers)
from alpa.shard_parallel.auto_sharding import (LogicalDeviceMesh)
from alpa.parallel_plan import PlacementSpec
from alpa.timer import timers, tracer
from alpa.util import (benchmark_func, list_gpu_info, OrderedSet,
                       update_jax_platform, is_ray_node_resource,
                       try_import_ray_worker, create_placement_group,
                       get_bundle_idx, retrieve_placement_group, get_bundle2ip,
                       check_server_port)

ray_worker = try_import_ray_worker()

if global_config.backend == "gpu" and global_config.has_cuda:
    from alpa.collective import worker_nccl_util

# Used ports for XLA distributed runtime servers.
used_port_set = set((None,))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ReshardingTileSpec = namedtuple("ReshardingTileSpec",
                                ["offset", "rank", "gpu_idx"])
ReshardingSendSpec = namedtuple("ReshardingSendSpec",
                                ["device_id", "tile_spec"])
ReshardingSendTask = namedtuple("ReshardingSendTask",
                                ["tile_specs", "group_name"])
ReshardingRecvSpec = namedtuple("ReshardingRecvSpec",
                                ["device_id", "shape", "dtype", "tile_specs"])
ReshardingRecvTask = namedtuple("ReshardingRecvTask",
                                ["recv_specs", "group_name"])
ReshardingBroadcastSpec = namedtuple("ReshardingBroadcastSpec", [
    "comm_key", "world_size", "devices_ids", "devices_global_rank",
    "tensor_slices", "recv_tile_shape", "dtype"
])
ReshardingBroadcastTask = namedtuple("ReshardingBroadcastTask",
                                     ["broadcast_specs", "group_name"])

SINGLE_HOST_NUM = 1


class MeshHostWorker:
    """
    A ray actor that manages the xla computation and buffers on a single host.
    """

    def __init__(self, server_address: str, num_hosts: int, host_id: int,
                 mesh_id: int, move_worker: DaemonMoveWorker,
                 runtime_random_seed: int):
        self.num_hosts = num_hosts
        self.host_id = host_id
        self.mesh_id = mesh_id
        self.move_worker = move_worker

        print("Getting distributed runtime client...")

        self.distributed_client = (
            xla_client._xla.get_distributed_runtime_client(
                server_address, host_id, use_coordination_service=False))
        
        print("Distributed runtime client is constructed, waiting to be connected...")

        
        print(f"{host_id}: Trying to connect to xla runtime at {server_address}")
        self.distributed_client.connect()
        
        print("Connected, waiting to make GPU client...")
        
        logger.debug(
            f"{host_id}: Success to connect to xla runtime at {server_address}")
        if global_config.backend == "gpu":
            self.backend = xla_client.make_gpu_client(self.distributed_client,
                                                      node_id=host_id)
        else:
            raise NotImplementedError(
                f"backend {global_config.backend} is not supported")

        self.backend = xb.get_backend("gpu")
        
        print("GPU client is completed, waiting to locate local devices...")

        # Monkey patch the backend
        set_override_backend(self.backend)
        self.local_devices = self.backend.local_devices()
        self.num_devices = len(self.local_devices)
        if global_config.enable_overlapping:
            xe.set_num_device_on_host(self.num_devices)

        self.local_devices = [self.backend.local_devices()[0]]
        self.num_devices = len(self.local_devices)

        print("Local devices are located.")

        self.buffers = {}  # Dict[uuid -> Sequence[DeviceArray]]
        self.executables = {}  # Dict[uud -> MeshWorkerExecutable]

        self.send_tasks = {}  # Dict[uuid -> ReshardingSendTask]
        self.recv_tasks = {}  # Dict[uuid -> ReshardingRecvTask]
        self.broadcast_tasks = {}  # Dict[uuid -> BroadcastTask]
        self.broadcast_communicators = {}

        self.data_loaders = {}  # Dict[uuid -> MeshWorkerDataLoader]
        self.data_loader_iters = {}  # Dict[uuid -> iterator]

        self.set_runtime_random_seed(runtime_random_seed)

        if global_config.pipeline_use_signal_send_recv:
            print("Use signal send recv for debugging.")
            self.signal_buffers = []
            for d in self.local_devices:
                jax_tensor = device_put(jnp.ones((1,), dtype=jnp.int8), d)
                self.signal_buffers.append(
                    worker_nccl_util.to_signal_buffer(jax_tensor))

    ##### Buffer Related Functions #####
    def put_buffers(self,
                    uuids: Union[int, Sequence[int]],
                    datas: Sequence[np.ndarray],
                    num_batch=1,
                    batch_dim=0):
        assert len(datas) == self.num_devices
        if not isinstance(uuids, Iterable):
            uuids = [uuids]
        assert len(uuids) == num_batch
        if num_batch > 1:
            split_datas = []
            for data in datas:
                split_buffers = np.split(data, num_batch, batch_dim)
                split_datas.extend(split_buffers)
            datas = split_datas
        arys = [([None] * self.num_devices) for _ in range(num_batch)]
        for i, data in enumerate(datas):
            if data.dtype == np.int64:
                data = data.astype(np.int32)
            device_id, batch_id = divmod(i, num_batch)
            arys[batch_id][device_id] = (self.backend.buffer_from_pyval(
                data, self.local_devices[device_id]))

        for uuid, ary in zip(uuids, arys):
            self.buffers[uuid] = ary

    def shard_and_put_non_zero_buffer(self, uuids: Union[Sequence[int], int],
                                      shape: Sequence[int], dtype: np.dtype,
                                      indices: Sequence, num_batch: int):
        if isinstance(uuids, int):
            uuids = [uuids]
        assert len(uuids) == num_batch
        assert len(indices) == self.num_devices * num_batch
        arys = [([None] * self.num_devices) for _ in range(num_batch)]
        for device_id in range(self.num_devices):
            for b in range(num_batch):
                shard_shape = []
                idx = device_id * num_batch + b
                for j, s in enumerate(indices[idx]):
                    filled_slice = s.indices(shape[j])
                    dim_size = len(range(*filled_slice))
                    shard_shape.append(dim_size)
                arys[b][device_id] = (self.backend.buffer_from_pyval(
                    np.full(shard_shape, 1e-8, dtype),
                    self.local_devices[device_id]))
        for uuid, ary in zip(uuids, arys):
            self.buffers[uuid] = ary

    def _get_buffers_with_local_ids(self, uuid: int, device_ids: Sequence[int]):
        bufs = self.buffers[uuid]
        # TODO(yonghao): sync communication events. Currently it's safe because
        # we never get values immediately after a cross-mesh communication.
        if device_ids is None:
            return map(np.asarray, bufs)
        elif not isinstance(device_ids, Iterable):
            return np.asarray(bufs[device_ids])
        return [np.asarray(bufs[device_id]) for device_id in device_ids]

    def get_buffers(self,
                    uuids: Union[Sequence[int], int],
                    device_indices: Sequence[int] = None):
        if not isinstance(uuids, Iterable):
            return self._get_buffers_with_local_ids(uuids, device_indices)
        if device_indices is not None:
            assert len(uuids) == len(device_indices)
        else:
            device_indices = [None] * len(uuids)
        return [
            self._get_buffers_with_local_ids(uuid, local_ids)
            for uuid, local_ids in zip(uuids, device_indices)
        ]

    def delete_buffers(self, uuids: Union[Sequence[int], int]):
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                del self.buffers[uuid]
        else:
            del self.buffers[uuids]

    def block_until_ready_buffers(self, uuids: Union[Sequence[int], int]):
        # We have to block all buffers to avoid the last operation is
        # cross-mesh resharding(not SPMD)
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                for buf in self.buffers[uuid]:
                    buf.block_until_ready()
        else:
            for buf in self.buffers[uuids]:
                buf.block_until_ready()

    def get_memory_allocated(self):
        self.sync()
        return max(d.memory_allocated() for d in self.local_devices)

    def get_max_memory_allocated(self):
        self.sync()
        return max(d.max_memory_allocated() for d in self.local_devices)

    def get_available_memory(self):
        self.sync()
        return min(d.available_memory() for d in self.local_devices)

    def reset_memory_stats(self):
        self.sync()
        for device in self.local_devices:
            device.clear_memory_stats()

    ##### Executable Related Functions #####
    def put_executable(self, uuid: int,
                       executable_class: "MeshWorkerExecutable", *args):
        self.executables[uuid] = executable_class(self, uuid, *args)

    def delete_executable(self, uuid: int):
        if uuid in self.executables:
            del self.executables[uuid]

    def run_executable(self, uuid: int, *args, **kwargs):
        self.executables[uuid].execute_on_worker(*args, **kwargs)

    def get_exec_hlo_text(self, uuid: int):
        return self.executables[uuid].get_hlo_text()

    def get_exec_total_allocation_size(self, uuid: int):
        return self.executables[uuid].get_total_allocation_size()

    def get_exec_grad_sync_channel_ids(self, uuid: int):
        return self.executables[uuid].grad_sync_channel_ids

    def set_runtime_random_seed(self, seed: int):
        seed = seed + (self.mesh_id << 20 if self.mesh_id else 0)
        for d in self.local_devices:
            d.set_seed(seed)

    ##### Serialization Related Functions #####
    def sync_move_worker(self):
        ray.get(self.move_worker.sync.remote())

    def save_array(self, ckpt_dir: str, local_cache_dir: Union[str, None],
                   uuid: int, device_ids: Sequence[int],
                   shard_indices: Sequence[Index], global_shape: Sequence[int]):
        assert uuid in self.buffers
        array_buffers = self.buffers[uuid]

        shard_names = [
            f"shard_{self.host_id}.{i}" for i in range(len(device_ids))
        ]

        metadata = {
            "global_shape": global_shape,
            "dtype": self.buffers[uuid][0].dtype,
            "shard_names": shard_names,
            "shard_indices": shard_indices,
        }

        # create directories if not exist
        os.makedirs(ckpt_dir, exist_ok=True)
        if local_cache_dir is not None:
            os.makedirs(local_cache_dir, exist_ok=True)
            save_dir = local_cache_dir
        else:
            save_dir = ckpt_dir

        for shard_name, device_id in zip(shard_names, device_ids):
            with open(os.path.join(save_dir, shard_name), "wb") as datafile:
                np.save(datafile, array_buffers[device_id])

        with open(os.path.join(save_dir, f"metadata_{self.host_id}"),
                  "wb") as metafile:
            pickle.dump(metadata, metafile)

        # move data
        if local_cache_dir is not None:
            self.move_worker.move.remote(local_cache_dir, ckpt_dir)

    def load_array(self, ckpt_dir: str, uuid: Sequence[int],
                   device_ids: Sequence[int], shard_indices: Sequence[Index]):
        metadatas = list(
            filter(lambda fname: fname.startswith("metadata"),
                   os.listdir(ckpt_dir)))
        # pylint: disable=import-outside-toplevel
        from alpa.serialization import load_sharded_array
        entire_arr = load_sharded_array(ckpt_dir, metadatas)
        array_buffers = [None] * self.num_devices
        for index, device_id in zip(shard_indices, device_ids):
            data = entire_arr[index]
            if data.dtype == np.int64:
                data = data.astype(np.int32)
            array_buffers[device_id] = (self.backend.buffer_from_pyval(
                data, self.local_devices[device_id]))
        self.buffers[uuid] = array_buffers

    ##### Data loader Related Functions #####
    def put_data_loader(self, uuid: int, *args):
        # pylint: disable=import-outside-toplevel
        from alpa.data_loader import MeshWorkerDataLoader
        self.data_loaders[uuid] = MeshWorkerDataLoader(self, *args)

    def data_loader_iter(self, uuid: int):
        self.data_loader_iters[uuid] = iter(self.data_loaders[uuid])

    def data_loader_next(self, uuid: int):
        next(self.data_loader_iters[uuid])

    def delete_data_loader(self, uuid: int):
        del self.data_loaders[uuid]

    ##### Cross Mesh Resharding Related Functions #####
    @staticmethod
    def init_collective_group(world_size, rank, backend, group_name):
        """Initialize the collective group eagerly."""
        col.init_collective_group(world_size,
                                  rank,
                                  backend=backend,
                                  group_name=group_name)

    @staticmethod
    def generate_nccl_uid(group_name):
        """Generate the NCCL unique ID in advance."""
        g = col.check_and_get_group(group_name)
        uid = g.generate_nccl_uid()
        return uid

    @staticmethod
    def init_p2p_communicator(group_name, my_rank, my_gpu_idx, peer_rank,
                              peer_gpu_idx, nccl_uid):
        """Initialize the P2P communicator from within the mesh workers."""
        assert col.is_group_initialized(group_name)
        assert col.get_rank(group_name) == my_rank
        g = col.check_and_get_group(group_name)
        g.create_p2p_communicator(my_gpu_idx, peer_rank, peer_gpu_idx, nccl_uid)

    @staticmethod
    def init_broadcast_communicator(group_name, comm_key, world_size,
                                    device_ids, devices_global_rank, nccl_uid):
        """Initialize the P2P communicator from within the mesh workers."""
        assert col.is_group_initialized(group_name)
        g = col.check_and_get_group(group_name)
        g.create_nccl_broadcast_communicator(comm_key, world_size, device_ids,
                                             devices_global_rank, nccl_uid)

    @staticmethod
    def destroy_collective_group(group_name: str = "default"):
        col.destroy_collective_group(group_name)

    def create_and_set_cross_mesh_communicators(self, world_size, rank, backend,
                                                group_name, key):
        """Create collective communicators for the cross mesh group."""
        if not col.is_group_initialized(group_name):
            self.init_collective_group(world_size, rank, backend, group_name)
        g = col.check_and_get_group(group_name)
        devices = list(range(self.num_devices))
        g.create_and_set_xla_communicators(devices, key)

    def put_resharding_send_task(self, uuid, tasks, group_name):
        self.send_tasks[uuid] = ReshardingSendTask(tile_specs=tasks,
                                                   group_name=group_name)

    def put_resharding_recv_task(self, uuid, tasks, group_name):
        self.recv_tasks[uuid] = ReshardingRecvTask(recv_specs=tasks,
                                                   group_name=group_name)

    def run_resharding_send_task(self, uuid, ary_uuid):
        task: ReshardingSendTask = self.send_tasks[uuid]
        group_name = task.group_name
        if global_config.enable_overlapping:
            col.wait_events(group_name, [ary_uuid], self.num_devices, True)

        for send_tile_spec in task.tile_specs:
            send_tile_spec: ReshardingSendSpec
            self.send_tile(ary_uuid, send_tile_spec.device_id,
                           send_tile_spec.tile_spec.offset,
                           send_tile_spec.tile_spec.rank,
                           send_tile_spec.tile_spec.gpu_idx, task.group_name)

    def run_resharding_recv_task(self, uuid, ary_uuid, set_empty_buffer=True):
        task: ReshardingRecvTask = self.recv_tasks[uuid]
        group_name = task.group_name
        if set_empty_buffer and ary_uuid not in self.buffers:
            assert not global_config.enable_overlapping, "Unsupported."
            self.buffers[ary_uuid] = [None] * self.num_devices

        if global_config.enable_overlapping:
            col.wait_events(group_name, [ary_uuid], self.num_devices, False)

        buffers = self.buffers[ary_uuid]
        for recv_spec in task.recv_specs:
            recv_spec: ReshardingRecvSpec
            device_id = recv_spec.device_id
            if set_empty_buffer:
                buffers[device_id] = self.backend.buffer_from_pyval(
                    np.full(recv_spec.shape, 1e-8, recv_spec.dtype),
                    self.local_devices[device_id])

            for recv_tile_spec in recv_spec.tile_specs:
                recv_tile_spec: ReshardingTileSpec
                self.recv_tile(ary_uuid, device_id, recv_tile_spec.offset,
                               recv_tile_spec.rank, recv_tile_spec.gpu_idx,
                               task.group_name)

        if global_config.enable_overlapping:
            col.record_events(group_name, [ary_uuid], self.num_devices, False)

    def send_tile(self, uuid: int, device_id: int, offset: Sequence[slice],
                  dst_rank: int, dst_gpu_idx: int, group_name: str):
        if global_config.pipeline_use_signal_send_recv:
            signal = self.signal_buffers[device_id]
            col.send_multigpu(signal,
                              dst_rank,
                              dst_gpu_idx,
                              group_name,
                              start_pos=0,
                              n_elements=1)
        else:
            worker_nccl_util.send_tile(self, uuid, device_id, offset, dst_rank,
                                       dst_gpu_idx, group_name)

    def recv_tile(self, uuid: int, device_id: int,
                  indices_in_dst_tile: Sequence[slice], src_rank: int,
                  src_gpu_idx: int, group_name: str):
        if uuid not in self.buffers:
            raise RuntimeError("Buffer has not been created.")

        if global_config.pipeline_use_signal_send_recv:
            signal = self.signal_buffers[device_id]
            col.recv_multigpu(signal,
                              src_rank,
                              src_gpu_idx,
                              group_name,
                              start_pos=0,
                              n_elements=1)
        else:
            worker_nccl_util.recv_tile(self, uuid, device_id,
                                       indices_in_dst_tile, src_rank,
                                       src_gpu_idx, group_name)

    def put_resharding_broadcast_task(self, uuid, tasks, group_name):
        self.broadcast_tasks[uuid] = ReshardingBroadcastTask(
            broadcast_specs=tasks, group_name=group_name)

    def run_resharding_broadcast_task(self,
                                      uuid,
                                      ary_uuid,
                                      set_empty_buffer=True):
        task: ReshardingBroadcastTask = self.broadcast_tasks[uuid]
        group_name = task.group_name
        broadcast_specs = task.broadcast_specs
        if set_empty_buffer and ary_uuid not in self.buffers:
            assert not global_config.enable_overlapping, "Unsupported."
            picked_spec = list(broadcast_specs.values())[0]
            shape = picked_spec.recv_tile_shape
            dtype = picked_spec.dtype
            self.buffers[ary_uuid] = [
                self.backend.buffer_from_pyval(np.full(shape, 1e-8, dtype),
                                               self.local_devices[device_id])
                for device_id in range(self.num_devices)
            ]

        has_recv = False
        for group_idx in broadcast_specs:
            broadcast_spec: ReshardingBroadcastSpec = broadcast_specs[group_idx]
            is_send = broadcast_spec.devices_global_rank[0] == 0
            has_recv = has_recv or not is_send
            if global_config.enable_overlapping:
                col.wait_events(group_name, [ary_uuid], self.num_devices,
                                is_send)

            worker_nccl_util.broadcast(self, ary_uuid, broadcast_spec.comm_key,
                                       broadcast_spec.world_size,
                                       broadcast_spec.devices_ids,
                                       broadcast_spec.devices_global_rank,
                                       broadcast_spec.tensor_slices,
                                       task.group_name)
        if global_config.enable_overlapping and has_recv:
            col.record_events(group_name, [ary_uuid], self.num_devices, False)

    ##### Profiling and Debugging Related Functions #####
    def profile_hlo_ops(self, op_infos: Sequence[Any], cache_filename: str,
                        single_timeout: float):
        num_devices = self.num_hosts * len(self.local_devices)
        return mesh_profiling.profile_hlo_ops(op_infos, self.backend,
                                              self.local_devices, self.host_id,
                                              num_devices, cache_filename,
                                              single_timeout)

    def profile_executable_with_dummy_inputs(self, uuid: int, **kwargs):
        return self.executables[uuid].profile_with_dummy_inputs(
            self.backend, self.local_devices, **kwargs)

    def profile_resharding_send_task(self,
                                     uuid,
                                     buf_uuids,
                                     warmup=1,
                                     repeat=3,
                                     number=3,
                                     sync=False):
        # TODO(yonghao): the sync function should be carefully reconsidered
        def run_fn():
            self.run_resharding_send_task(uuid, buf_uuids)

        sync_fn = self.sync if sync else None
        costs = benchmark_func(run_fn, sync_fn, warmup, repeat, number)
        return np.mean(costs)

    def profile_resharding_recv_task(self,
                                     uuid,
                                     buf_uuids,
                                     warmup=1,
                                     repeat=3,
                                     number=3,
                                     sync=False):
        set_empty_buffer = True

        def run_fn():
            nonlocal set_empty_buffer
            self.run_resharding_recv_task(uuid, buf_uuids, set_empty_buffer)
            set_empty_buffer = False

        sync_fn = self.sync if sync else None
        costs = benchmark_func(run_fn, sync_fn, warmup, repeat, number)
        return np.mean(costs)

    @staticmethod
    def get_timer(name: str):
        return timers(name)

    @staticmethod
    def reset_timer(name: str):
        timers(name).reset()

    @staticmethod
    def get_tracer():
        return tracer

    def get_live_buffer_uuids(self):
        return list(self.buffers.keys())

    ##### Other Functions #####
    def sync(self, sync_all_devices=False):
        # We sync one device instead of all for smaller runtime overhead.
        # This is correct because of SPMD.
        if sync_all_devices:
            for device in self.local_devices:
                device.synchronize_all_activity()
        else:
            self.local_devices[0].synchronize_all_activity()

    def sync_all(self):
        for device in self.local_devices:
            device.synchronize_all_activity()

    @staticmethod
    def check_alive():
        return True

    def shutdown(self):
        self.sync()
        self.buffers.clear()
        self.executables.clear()
        self.distributed_client.shutdown()
        # sync & shutdown DaemonMoveWorker
        self.sync_move_worker()
        ray.kill(self.move_worker)
        self.move_worker = None


"""
Definition of the customized virtual device mesh.
-------------------------------------------------
we modify the API of getting physical mesh to implement device-independent mesh definition. 
Thus, we can instantiate multiple MeshHostWorkers on the single host with even only one GPU 
(to provide pre-allocated GPU memory for XLA Python client).
"""

class VirtualPhysicalMesh:
    """
    A virtual physical mesh used for pipeline parallel compilation.

    VirtualPhysicalMesh is used during compile time. We don't allocate actual
    workers for it. When compilation is finished, we instantiated it as a
    PhysicalDeviceMesh and launch workers.

    A VirtualPhysicalMesh can also be sliced into multiple VirtualPhysicalMesh.
    After slicing, each sliced VirtualPhysicalMesh can be instantiated as a
    PhysicalDeviceMesh. These sliced PhysicalDeviceMesh together can form a
    PhysicalDeviceMeshGroup for pipeline parallelism.
    """

    def __init__(self,
                 host_ids: Sequence[int],
                 host_info: Sequence[dict],
                 num_devices_per_host,
                 parent: "VirtualPhysicalMesh" = None,
                 devices: Sequence[Sequence[int]] = None):
        # host_ids are the indices of hosts in the global DeviceCluster
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_devices_per_host = num_devices_per_host
        self.parent = parent

        self.launched_physical_mesh = None
        self.launched_physical_mesh_group = None

        #################################
        #       Modified by Crius       #
        #################################

        self.devices = [None for _ in host_ids]
        self.device_strs = []
        
        # if devices is not None:
        #     if len(devices) != len(host_ids):
        #         raise RuntimeError(
        #             "Please specify the gpu IDs used on each host.")
        #     if not all(len(ids) == num_devices_per_host for ids in devices):
        #         raise RuntimeError(
        #             "Device IDs specified for each host does not align "
        #             "with `num_devices_per_host`.")
        # else:
        #     devices = [list(range(num_devices_per_host)) for _ in host_ids]

        # self.devices = devices
        # # Depending on gpu_ids, generate device strs and ask Ray to allocate.
        # self.device_strs = []

        # (Modified by Crius) Distable underlying device info check. 
        # for i in range(self.num_hosts):
        #     ip = self.host_info[i]["NodeManagerAddress"]
        #     self.device_strs.extend(
        #         [device_id_to_str(ip, j) for j in devices[i]])

        #################################
        #       Modification End        #
        #################################

    @property
    def shape(self):
        return (len(self.host_ids), self.num_devices_per_host)

    @property
    def num_devices(self):
        """Return the total number of GPUs on this mesh."""
        return len(self.host_ids) * self.num_devices_per_host

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    def slice_1d(self, dim: int, indices: Sequence[int]):
        """
        Slice a mesh given the slicing config.

        Args:
            dim: which dimension to slice from, 0 is host or 1 is the gpu
            indices: indices to include along this dimension.

        Returns:
            mesh (PhysicalDeviceMesh)
        """
        if dim == 0:
            # slicing along the host dimension
            host_ids = [self.host_ids[x] for x in indices]
            host_info = [self.host_info[x] for x in host_ids]
            return VirtualPhysicalMesh(
                host_ids=host_ids,
                host_info=host_info,
                num_devices_per_host=self.num_devices_per_host,
                parent=self)
        else:
            # slicing along the device dimension

            # (Modified by Crius)
            # # Check the validity of device_indices
            # for i in range(len(indices)):
            #     for x in indices[i]:
            #         assert x in self.devices[i]

            return VirtualPhysicalMesh(host_ids=self.host_ids,
                                       host_info=self.host_info,
                                       num_devices_per_host=len(indices[0]),
                                       parent=self,
                                       devices=indices)

    def slice_2d(self, host_indices, device_indices):
        host_ids = [self.host_ids[x] for x in host_indices]
        host_info = [self.host_info[x] for x in host_indices]

        # (Modified by Crius)
        # # Check the validity of device_indices
        # for i in range(len(device_indices)):
        #     for x in device_indices[i]:
        #         assert x in self.devices[i]

        return VirtualPhysicalMesh(host_ids=host_ids,
                                   host_info=host_info,
                                   num_devices_per_host=len(device_indices[0]),
                                   parent=self,
                                   devices=device_indices)

    def slice_profiling_submeshes(self, submesh_num_hosts,
                                  submesh_num_devices_per_host):
        num_hosts = len(self.host_ids)
        num_devices_per_host = self.num_devices_per_host
        num_host_submeshes = num_hosts // submesh_num_hosts
        num_device_submeshes = (num_devices_per_host //
                                submesh_num_devices_per_host)
        all_submeshes = []
        for i in range(num_host_submeshes):
            for j in range(num_device_submeshes):
                host_indices = range(i * submesh_num_hosts,
                                     (i + 1) * submesh_num_hosts)
                device_indices = [
                    range(j * submesh_num_devices_per_host,
                          (j + 1) * submesh_num_devices_per_host)
                    for _ in host_indices
                ]
                all_submeshes.append(self.slice_2d(host_indices,
                                                   device_indices))
        return all_submeshes

    def get_logical_mesh(self,
                         mesh_shape: Optional[Sequence[int]] = None,
                         mesh_alpha: Optional[float] = None,
                         mesh_beta: Optional[float] = None):
        """
        Return a logical mesh and parameters of the alpha-beta communication
        cost model. The logical view is used for auto-sharding.
        """
        if mesh_shape is None:
            mesh_shape = (self.num_hosts, self.num_devices_per_host)

        id_mesh = np.arange(self.num_devices).reshape(mesh_shape)
        mesh_alpha = mesh_alpha or (1, 1)
        mesh_beta = mesh_beta or (1, 0.1)
        return LogicalDeviceMesh(None, id_mesh, mesh_alpha, mesh_beta)

    def get_physical_mesh(self, mesh_id: int = 0):
        """Launch a physical mesh (which will request resources from Ray)."""
        assert self.launched_physical_mesh is None, \
            "Physical mesh can only be launched once."

        self.launched_physical_mesh = DistributedPhysicalDeviceMesh(
            host_ids=self.host_ids,
            host_info=self.host_info,
            num_devices_per_host=self.num_devices_per_host,
            parent=self,
            devices=self.devices,
            mesh_id=mesh_id)
        return self.launched_physical_mesh

    def get_physical_mesh_group(self, sliced_virtual_meshes):
        """Launch a physical mesh group (which will request resources from
        Ray)."""
        assert self.launched_physical_mesh_group is None, \
            "Physical mesh group can only be launched once."
        
        # Launch physical meshes in parallel
        physical_meshes = [None] * len(sliced_virtual_meshes)

        def launch_func(i):
            physical_meshes[i] = sliced_virtual_meshes[i].get_physical_mesh(i)

        print("Launching physical meshes...")
        
        threads = []
        for i in range(len(sliced_virtual_meshes)):
            t = threading.Thread(target=launch_func, args=(i,))
            t.start()
            threads.append(t)
        for i in range(len(sliced_virtual_meshes)):
            threads[i].join()
        
        print("Physical meshes have been launched.")

        self.launched_physical_mesh_group = (PhysicalDeviceMeshGroup(
            physical_meshes, self))
        
        print("Physical mesh group has been initialized.")
        
        return self.launched_physical_mesh_group


"""
Definition of the customized physical device mesh.
-------------------------------------------------
We modify the initialization of the DistributedPhysicalDeviceMesh class, enabling instantiating 
multiple ray actors on the single GPU.
"""

class DistributedPhysicalDeviceMesh(PhysicalDeviceMesh):
    """
    A multi-host physical device mesh to run computation distributedly.
    It uses ray actors and the distributed XLA runtime.
    """

    def __init__(self,
                 host_ids: Sequence[int],
                 host_info: Sequence[dict],
                 num_devices_per_host: int,
                 parent: Optional["VirtualPhysicalMesh"] = None,
                 devices: Optional[Sequence[Sequence[int]]] = None,
                 mesh_id: Optional[int] = None,
                 namespace: Optional[str] = None):
        # host_ids are the indices of hosts in the global DeviceCluster
        self.host_ids = host_ids
        self.host_info = host_info
        self.num_hosts = len(host_ids)
        self.num_devices_per_host = num_devices_per_host
        self.parent = parent
        self.mesh_id = mesh_id
        self.workers = None
        self.service_server = None
        self.operation_executables = {}
        self.one_replica_ids = {}
        self.namespace = namespace

        #################################
        #       Modified by Crius       #
        #################################

        if devices is not None:
            # if len(devices) != len(host_ids):
            #     raise RuntimeError(
            #         "Please specify the gpu IDs used on each host.")
            # if not all(len(ids) == num_devices_per_host for ids in devices):
            #     raise RuntimeError(
            #         "Devices specified for each host does not align "
            #         "with `num_devices_per_host`.")
            self.devices = devices
        else:
            # devices = [list(range(num_devices_per_host)) for _ in host_ids]
            self.devices = [None for _ in host_ids]

        # self.devices = devices
        self.device_strs = []
        self.node_ips = [] 

        for i in range(self.num_hosts):
            # ip = self.host_info[i]["NodeManagerAddress"]
            ip = "fake_ip_addr"
            # self.device_strs.extend(
            #     [device_id_to_str(ip, j) for j in devices[i]])
            self.node_ips.append(ip)
        
        self.service_servers, self.workers = self.launch_xla_servers()
        self.launched = True

        print("XLA servers have been launched.")
        
        # found_existing_workers = False
        # if self.namespace:
        #     try:
        #         ray.get_actor(self.get_host_worker_name(0))
        #         found_existing_workers = True
        #     except ValueError:
        #         pass

        # if found_existing_workers:
        #     self.service_server = None
        #     self.workers = self.connect_to_existing_workers()
        #     self.launched = False
        # else:
        #     self.service_server, self.workers = self.launch_xla_servers()
        #     self.launched = True

        #################################
        #       Modification End        #
        #################################

        self.to_delete_remote_refs = []
        self.to_delete_remote_ref_ct = 0

    def get_host_worker_name(self, host_id):
        if self.namespace:
            return f"mesh_{self.mesh_id}_host_{host_id}"
        else:
            return None

    def connect_to_existing_workers(self):
        workers = []
        for i in range(self.num_hosts):
            workers.append(ray.get_actor(self.get_host_worker_name(i)))
        return workers
    
    def create_virtual_placement_group(self, num_hosts: int, num_devices_per_host: int):
        """ 
        Modified by Crius.
        ----------------------------------------------------------------
        Create virtual placement group in Ray without device check, 
        which is ray.wait([placement_group.ready()], timeout=timeout)). 
        """
        # Resource bundles
        bundles = [{
            "CPU": int(num_hosts * 2),
            "GPU": num_devices_per_host,
        } for _ in range(SINGLE_HOST_NUM)]
        # Create Ray placement group
        placement_group = ray.util.placement_group(bundles, strategy="SPREAD", name="test_pg")
        
        return placement_group
    
    def launch_multi_xla_servers_on_single_host(self):
        """ Launch multiple XLA servers on the single host. """
        service_servers, server_addresses = list(), list()

        for i in range(self.num_hosts):
            # Launch distributed xla runtime
            port = None
            while port in used_port_set:
                port = np.random.randint(global_config.xla_server_port_start,
                                        global_config.xla_server_port_end)
                if check_server_port(ray.util.get_node_ip_address(), port):
                    port = None
            used_port_set.add(port)

            server_address = f"{ray.util.get_node_ip_address()}:{port}"
            print(f"Trying to start XLA gRPC server on port: {port}...")
            service_server = xla_client._xla.get_distributed_runtime_service(
                server_address, self.num_hosts, use_coordination_service=False)
            print(f"Success to start XLA gRPC server on port: {port}...")
            time.sleep(0.4)

            service_servers.append(service_server)
            server_addresses.append(server_address)
        
        return service_servers, server_addresses

    def launch_xla_servers(self):

        #################################
        #       Modified by Crius       #
        #################################

        multiple_servers = True

        if not multiple_servers:
            # Launch distributed xla runtime
            port = None
            while port in used_port_set:
                port = np.random.randint(global_config.xla_server_port_start,
                                        global_config.xla_server_port_end)
                if check_server_port(ray.util.get_node_ip_address(), port):
                    port = None
            used_port_set.add(port)

            server_address = f"{ray.util.get_node_ip_address()}:{port}"
            logger.debug(f"Trying to start XLA gRPC server on port: {port}...")
            service_server = xla_client._xla.get_distributed_runtime_service(
                server_address, self.num_hosts, use_coordination_service=False)
            logger.debug(f"Success to start XLA gRPC server on port: {port}...")
            time.sleep(0.4)

            service_servers = [service_server]
            server_addresses = [server_address]
        else:
            service_servers, server_addresses = self.launch_multi_xla_servers_on_single_host()

        # Launch workers
        workers = []

        # Apply virtual placement group
        apply_virtual_pg = True
        if apply_virtual_pg:
            # Virtual physical devices
            num_hosts = int(os.environ.get("CRIUS_NUM_HOSTS"))
            num_devices_per_host = int(os.environ.get("CRIUS_NUM_DEVICES_PER_HOST"))
            assert num_hosts is not None and num_devices_per_host is not None, \
                "Environment variables of the virtual cluster specs are not properly set."
            assert num_devices_per_host == 1, \
                "Device num per host must be initialized to 1 to enable single-GPU profiling by creating multiple Ray actors."
            print(f"[I] Virtual physical device info: (1) Host num: {num_hosts}; (2) Device num per host: {num_devices_per_host}.")
            # Define virtual placement group instead of directly accessing the underlying Ray resources.
            placement_group = self.create_virtual_placement_group(num_hosts, num_devices_per_host)
            # Default bundle index list
            device_bundle_idx_list = [i for i in range(SINGLE_HOST_NUM)]
        else:
            # Retrieve the placement group
            placement_group = retrieve_placement_group()
            # Get the sorted bundle index list
            device_bundle_idx_list = get_bundle_idx(placement_group, self.node_ips)

        #################################
        #       Modification End        #
        #################################

        def lanuch_func(_i):
            # Launch the DaemonMoveWorker
            cls = ray.remote(num_cpus=0)(DaemonMoveWorker)
            move_worker = cls.options(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_index).remote()
            
            print(f"Launching MeshHostWorker {_i}...")

            # Launch the MeshHostWorker
            cls = ray.remote(num_cpus=0,
                            num_gpus=self.num_devices_per_host)(MeshHostWorker)
            worker = cls.options(placement_group=placement_group,
                                placement_group_bundle_index=bundle_index,
                                name=host_worker_name,
                                runtime_env={
                                    "env_vars": env_vars
                                }).remote(server_addresses[_i], self.num_hosts, _i,
                                        self.mesh_id, move_worker,
                                        global_config.runtime_random_seed)
            workers.append(worker)

            print(f"MeshHostWorker on host {_i} has been launched.")
        
        threads = []

        for i in range(self.num_hosts):
            # Set XLA environment variables
            env_vars = {
                "ALPA_IS_WORKER":
                    "True",
                "NCCL_USE_MULTISTREAM":
                    "False",
                # "NCCL_USE_MULTISTREAM":
                #     "True",
                "XLA_PYTHON_CLIENT_MEM_FRACTION":
                    str(global_config.xla_client_mem_fraction),
                "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "") +
                              f" --xla_gpu_autotune_level"
                              f"={global_config.xla_gpu_autotune_level}"),
                "XLA_PYTHON_CLIENT_PREALLOCATE":
                    global_config.xla_client_client_preallocate,
                # "NCCL_LAUNCH_MODE": "PARALLEL",
                # "XLA_FLAGS": "--xla_dump_to=hlo --xla_dump_hlo_pass_re=.*"
                # "NCCL_DEBUG": "INFO" if i == 0 else "VERSION",
                # "NCCL_DEBUG_SUBSYS": "ALL",
                # "RAY_IGNORE_UNHANDLED_ERRORS": "True",
            }

            if global_config.resharding_mode == "broadcast":
                env_vars["NCCL_ALGO"] = "Ring"
                env_vars["NCCL_PROTO"] = "Simple"

            if "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ:
                env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = os.environ[
                    "XLA_PYTHON_CLIENT_ALLOCATOR"]

            if "NCCL_DEBUG" in os.environ:
                env_vars["NCCL_DEBUG"] = os.environ[
                    "NCCL_DEBUG"] if i == 0 else "VERSION"

            if global_config.use_aws_efa:
                env_vars.update({
                    "FI_PROVIDER": "efa",
                    "FI_EFA_USE_DEVICE_RDMA": "1",
                    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH",
                                                      ""),  # For libnccl-net.so
                    "NCCL_PROTO": "simple",
                })

            # bundle_index = device_bundle_idx_list[i]
            bundle_index = device_bundle_idx_list[0]

            host_worker_name = self.get_host_worker_name(i)

            #################################
            #       Modified by Crius       #
            #################################

            t = threading.Thread(target=lanuch_func, args=(i, ))
            t.start()
            threads.append(t)
            
        for i in range(self.num_hosts):
            threads[i].join()

            #################################
            #       Modification End        #
            #################################

        # return service_server, workers
        return service_servers, workers

    @property
    def host_ips(self):
        ips = [
            self.host_info[i]["NodeManagerAddress"]
            for i, _ in enumerate(self.host_ids)
        ]
        return ips

    def get_virtual_physical_mesh(self):
        return VirtualPhysicalMesh(
            host_ids=self.host_ids,
            host_info=self.host_info,
            num_devices_per_host=self.num_devices_per_host,
            parent=self,
            devices=self.devices)

    def _split_ids_to_host(self, host_local_ids: Sequence[Tuple[int, int]]):
        if host_local_ids is None:
            full_local_id = [
                range(self.num_devices_per_host) for _ in range(self.num_hosts)
            ]
            full_id_local_idx = [(i, j)
                                 for i in range(self.num_hosts)
                                 for j in range(self.num_devices_per_host)]
            return tuple(full_local_id), full_id_local_idx
        per_host_id = [[] for _ in range(self.num_hosts)]
        host_id_local_idx = []
        for id_pair in host_local_ids:
            host_id, device_id = id_pair
            host_id_local_idx.append((host_id, len(per_host_id[host_id])))
            per_host_id[host_id].append(device_id)
        return per_host_id, host_id_local_idx

    ##### Buffer Related Functions #####
    def get_remote_buffers(
            self,
            ary_refs: Union[List["RemoteArrayRef"], "RemoteArrayRef"],
            host_local_ids: Sequence[Sequence[Tuple[int, int]]] = None,
            batching=False,
            return_ray_ref=False):
        """
        Get values of remote buffers.

        Args:
            host_local_ids: For each RemoteArrayRef, we can fetch a list of
              buffers from multiple devices on multiple hosts. This variable
              defines a list of (host_id, local_id) pair for each
              RemoteArrayRef. If it is None, fetch all remote buffers.
            batching: Whether batch remote calls by host ids. This can reduce
              ray overhead.
        """
        return_list = True
        if not isinstance(ary_refs, Iterable):
            return_list = False
            ary_refs = [ary_refs]
        if host_local_ids is None:
            host_local_ids = [None] * len(ary_refs)
        elif not isinstance(host_local_ids, Iterable):
            assert not return_list
            host_local_ids = [host_local_ids]

        if batching:
            # Batch the remote calls by host ids
            ary_ids = np.array([ref.uuid for ref in ary_refs])
            per_host_ids = np.empty((self.num_hosts, len(ary_ids)),
                                    dtype=object)
            host_id_local_indices = []
            for arg_id, id_pairs in enumerate(host_local_ids):
                tmp_ids, tmp_indices = self._split_ids_to_host(id_pairs)
                host_id_local_indices.append(tmp_indices)
                for host_id, tmp_per_host in enumerate(tmp_ids):
                    per_host_ids[host_id][arg_id] = np.array(tmp_per_host)

            # [host_id-> (buf_idx-> (local_device_id->device_buffer))]
            obj_refs = []
            for host_id in range(self.num_hosts):
                obj_refs.append(self.workers[host_id].get_buffers.remote(
                    ary_ids, per_host_ids[host_id]))
            per_host_results = ray.get(obj_refs)
            # [buf_id -> (flatten_id -> device_buffer)]
            ret = []
            for ref_idx, id_pairs in enumerate(host_id_local_indices):
                buffers = []
                for id_pair in id_pairs:
                    host_id, local_idx = id_pair
                    buffers.append(
                        per_host_results[host_id][ref_idx][local_idx])
                ret.append(buffers)
        else:
            obj_refs = []
            for ary_ref, id_pairs in zip(ary_refs, host_local_ids):
                ary_obj_refs = []
                for id_pair in id_pairs:
                    host_id, local_id = id_pair
                    ary_obj_refs.append(
                        self.workers[host_id].get_buffers.remote(
                            ary_ref.uuid, local_id))
                obj_refs.append(ary_obj_refs)
            if return_ray_ref:
                ret = obj_refs
            else:
                ret = [ray.get(refs) for refs in obj_refs]
        return ret if return_list else ret[0]

    def delete_remote_buffers(self, ary_refs: List["RemoteArrayRef"]):
        """Delete remote buffers."""
        if not self.workers or not ray or not ray_worker or not np.array:
            return

        # Put delete requests into a buffer
        for ary_ref in ary_refs:
            self.to_delete_remote_refs.append(ary_ref.uuid)
        self.to_delete_remote_ref_ct += len(ary_refs)

        # Execute the delete requests if there are enough requests
        if (self.to_delete_remote_ref_ct >
                global_config.delete_remote_arrays_threshold):
            to_delete_remote_refs = np.array(self.to_delete_remote_refs)
            try:
                for host_id in range(self.num_hosts):
                    self.workers[host_id].delete_buffers.remote(
                        to_delete_remote_refs)
            except AttributeError:
                pass
            self.to_delete_remote_refs = []
            self.to_delete_remote_ref_ct = 0

    def block_until_ready_remote_buffers(self,
                                         ary_refs: List["RemoteArrayRef"]):
        """Block until the remote buffers are ready."""
        tasks = []
        ary_uuids = np.array([ref.uuid for ref in ary_refs])
        for worker in self.workers:
            tasks.append(worker.block_until_ready_buffers.remote(ary_uuids))
        ray.get(tasks)

    ##### Executable Related Functions #####
    def shard_args_to_bufs(self, shard_indices: Sequence[Sequence[Index]],
                           donated_invars: Sequence[bool],
                           batch_invars: Sequence[bool], num_micro_batches: int,
                           args: Sequence[Any]):
        ret_bufs = []
        total_bytes = 0
        time_start = time.time()

        for arg, indices, donated, is_batch_var in zip(args, shard_indices,
                                                       donated_invars,
                                                       batch_invars):
            tic = time.time()
            slow_path = False

            if is_batch_var:
                if (isinstance(arg, DistributedArray) and
                        arg.skip_shard_args_check is True):
                    assert num_micro_batches == 1
                    ret_bufs.append([arg.remote_ref])
                else:
                    slow_path = True
                    if not isinstance(arg, ShapedArray):
                        arg = np.asarray(arg)
                    refs = _shard_array(arg, self, indices, num_micro_batches)
                    ret_bufs.append(refs)
            else:
                if (isinstance(arg, DistributedArray) and
                        arg.device_mesh == self and arg.indices == indices):
                    # Fast path for DistributedArray
                    ret_bufs.append(arg.remote_ref)
                elif isinstance(arg, ReplicatedDistributedArray):
                    replica = arg.get_replica_on_mesh(self)
                    assert replica.indices == indices
                    ret_bufs.append(replica.remote_ref)
                else:  # Slow path
                    slow_path = True
                    if type(arg) not in [ShapedArray, ShapeDtypeStruct]:
                        arg = xla.canonicalize_dtype(arg)
                    ref = shard_arg_handlers[type(arg)](arg, self, indices)[0]
                    ret_bufs.append(ref)
                    if donated and hasattr(arg, "delete"):
                        # shard_arg_handler always creates new buffers,
                        # so we can delete the old buffers
                        arg.delete()

            if False and slow_path:  # pylint: disable=condition-evals-to-constant
                # Print debug info
                size = np.prod(arg.shape) * arg.dtype.itemsize
                bandwidth = size / (time.time() - tic)
                total_bytes += size
                print("Slow path. "
                      f"shape: {arg.shape}, "
                      f"bandwidth: {bandwidth/1024**2:.2f} MB/s "
                      f"total_bytes: {total_bytes/1024**2:.2f} MB "
                      f"total_time: {time.time() - time_start:.2f}")

        return ret_bufs

    def shard_args_to_arrays(self, avals: Sequence[ShapedArray],
                             shard_indices: Sequence[Sequence[Index]],
                             sharding_specs: Sequence[ShardingSpec],
                             args: Sequence[np.array]):
        arrays = []
        for i in range(len(avals)):
            remote_ref = _shard_array(args[i], self, shard_indices[i])[0]
            arrays.append(
                DistributedArray(self, avals[i], sharding_specs[i], remote_ref,
                                 shard_indices[i]))
        return arrays

    def get_outputs_handler(self, avals: Sequence[ShapedArray],
                            sharding_specs: Sequence[ShardingSpec]):
        indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, sharding_specs)
        ]

        def outs_handler(refs):
            ret = []
            for i, aval in enumerate(avals):
                dis_array = DistributedArray(device_mesh=self,
                                             aval=aval,
                                             sharding_spec=sharding_specs[i],
                                             remote_ref=refs[i],
                                             indices=indices[i])
                ret.append(dis_array)
            return ret

        return outs_handler

    def delete_remote_executable(self, exec_uuid: int):
        """Delete remote worker executables of a driver executable."""
        if not self.workers or not ray or not ray_worker or not np.array:
            return

        try:
            for w in self.workers:
                w.delete_executable.remote(exec_uuid)
        except AttributeError:
            pass

    def set_runtime_random_seed(self, seed: int):
        for w in self.workers:
            w.set_runtime_random_seed.remote(seed)

    ##### Profiling and Debugging Related Functions #####
    def profile_hlo_ops(self,
                        op_infos: Sequence[Tuple],
                        cache_filename: str,
                        single_timeout: Optional[float] = None,
                        batch_timeout: Optional[float] = None):
        tasks = []
        for w in self.workers:
            tasks.append(
                w.profile_hlo_ops.remote(op_infos, cache_filename,
                                         single_timeout))
        return ray.get(tasks, timeout=batch_timeout)[0]

    def get_remote_timer(self, timer_name: str):
        return ray.get(self.workers[0].get_timer.remote(timer_name))

    def reset_remote_timer(self, timer_name: str):
        for worker in self.workers:
            ray.get(worker.reset_timer.remote(timer_name))

    def get_remote_tracer(self):
        return ray.get(self.workers[0].get_tracer.remote())

    def get_memory_allocated(self):
        return max(
            ray.get([w.get_memory_allocated.remote() for w in self.workers]))

    def get_max_memory_allocated(self):
        return max(
            ray.get([w.get_max_memory_allocated.remote() for w in self.workers
                    ]))

    def get_available_memory(self):
        return min(
            ray.get([w.get_available_memory.remote() for w in self.workers]))

    def reset_memory_stats(self):
        for worker in self.workers:
            ray.get(worker.reset_memory_stats.remote())

    ##### Other Functions #####
    def sync_workers(self, sync_all_devices=False):
        ray.get([w.sync.remote(sync_all_devices) for w in self.workers])

    def sync_move_workers(self):
        ray.get([w.sync_move_worker.remote() for w in self.workers])

    def shutdown(self, forced=False):
        self.operation_executables.clear()
        if not self.launched:
            return
        if not forced:
            ray.get([w.shutdown.remote() for w in self.workers])
        for worker in self.workers:
            ray.kill(worker)
        self.workers = None

        multiple_servers = True
        if not multiple_servers:
            # shutdown grpc server
            if self.service_server:
                self.service_server.shutdown()
                self.service_server = None
        else:
            for _server in self.service_servers:
                if _server:
                    _server.shutdown()
            self.service_servers = None
        self.launched = False


class PhysicalDeviceMeshGroup:
    """A list of physical devices that forms a pipeline."""

    def __init__(self, meshes: Sequence[DistributedPhysicalDeviceMesh],
                 parent: VirtualPhysicalMesh):
        self.meshes = list(meshes)
        self.parent = parent
        self.collective_groups: List[List[Any]] = [
            [None for _ in range(len(self))] for _ in range(len(self))
        ]

    def __getitem__(self, index):
        return self.meshes[index]

    def __len__(self):
        return len(self.meshes)

    def index(self, *args, **kwargs):
        return self.meshes.index(*args, **kwargs)

    def establish_nccl_group(self,
                             src_mesh_id: int,
                             dst_mesh_id: int,
                             instantiate=True):
        """Establish NCCL group between two meshes."""
        # pylint: disable=import-outside-toplevel
        from alpa.pipeline_parallel.cross_mesh_resharding import CollectiveGroup

        assert src_mesh_id < dst_mesh_id
        if self.collective_groups[src_mesh_id][dst_mesh_id] is not None:
            # Already established
            return
        src_mesh = self.meshes[src_mesh_id]
        dst_mesh = self.meshes[dst_mesh_id]
        device_strs = OrderedSet(src_mesh.device_strs + dst_mesh.device_strs)
        cg = CollectiveGroup(device_strs, src_mesh, dst_mesh)
        self.collective_groups[src_mesh_id][dst_mesh_id] = cg
        self.collective_groups[dst_mesh_id][src_mesh_id] = cg
        if instantiate:
            self._instantiate_nccl_group(cg)

    def instantiate_nccl_group(self, src_mesh_id: int, dst_mesh_id: int):
        cg = self.collective_groups[src_mesh_id][dst_mesh_id]
        self._instantiate_nccl_group(cg)

    def shard_args_to_arrays(self, placement_specs: PlacementSpec,
                             args: Sequence[Any]):
        rets = []

        for info, arg in zip(placement_specs, args):
            aval = info.aval
            if len(info.mesh_ids) == 1:
                mesh = self.meshes[info.mesh_ids[0]]
                spec = info.sharding_specs[0]
                indices = pxla.spec_to_indices(aval.shape, spec)
                rets.append(
                    mesh.shard_args_to_arrays((aval,), (indices,), (spec,),
                                              (arg,))[0])
            else:
                meshes, arrays = [], []
                for mesh_id, spec in zip(info.mesh_ids, info.sharding_specs):
                    mesh = self.meshes[mesh_id]
                    meshes.append(mesh)
                    indices = pxla.spec_to_indices(aval.shape, spec)
                    arrays.append(
                        mesh.shard_args_to_arrays((aval,), (indices,), (spec,),
                                                  (arg,))[0])
                rets.append(ReplicatedDistributedArray(meshes, arrays))

        return rets

    def set_runtime_random_seed(self, seed: int):
        for m in self.meshes:
            m.set_runtime_random_seed(seed)

    def sync_workers(self):
        """Sync device activities on all workers."""
        all_workers = [w for mesh in self.meshes for w in mesh.workers]
        ray.get([w.sync.remote() for w in all_workers])

    def sync_move_workers(self):
        """Sync moveworkers on all meshes."""
        for mesh in self.meshes:
            mesh.sync_move_workers()

    def get_memory_allocated(self):
        """Get the current size of allocated memory."""
        calls = []
        for mesh in self.meshes:
            for worker in mesh.workers:
                calls.append(worker.get_memory_allocated.remote())
        return max(ray.get(calls))

    def get_max_memory_allocated(self):
        """Get the maximal size of memory allocated so far."""
        calls = []
        for mesh in self.meshes:
            for worker in mesh.workers:
                calls.append(worker.get_max_memory_allocated.remote())
        return max(ray.get(calls))

    def get_max_memory_allocated_per_mesh(self):
        """Get the maximal size of memory allocated for each mesh so far."""
        return [mesh.get_max_memory_allocated() for mesh in self.meshes]

    def reset_memory_stats(self):
        for mesh in self.meshes:
            mesh.reset_memory_stats()

    def destroy_collective_groups(self):
        for i in range(len(self)):
            for j in range(len(self)):
                if i < j and self.collective_groups[i][j] is not None:
                    self.collective_groups[i][j].destroy()

    def shutdown(self):
        self.destroy_collective_groups()
        for mesh in self.meshes:
            mesh.shutdown()

    def exception_shutdown(self):
        """In this shutdown, some actors might have died."""
        # recycle collective group info
        for i in range(len(self)):
            for j in range(len(self)):
                if i < j and self.collective_groups[i][j]:
                    group_name = self.collective_groups[i][j].group_name
                    # TODO(Hao): move this part of recycling to
                    #   ray.util.collective instead of here.
                    name = "info_" + group_name
                    try:
                        store = ray.get_actor(name)
                        ray.kill(store)
                    except ValueError:
                        pass
        # TODO(Hao): recycle the NCCLUniqueID named actor. Their name is MD5
        #  hashed. each of them will take 1 CPU.
        # recycle info actors
        for mesh in self.meshes:
            mesh.shutdown(forced=True)

    @staticmethod
    def _instantiate_nccl_group(cg):
        if global_config.eagerly_create_communicators:
            cg.instantiate_now()
        else:
            cg.instantiate()
