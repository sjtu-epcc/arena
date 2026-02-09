#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to the offline profiler of communication operators on intra/inter-node devices. """

import time
from typing import Any, Sequence, List
import argparse
import pickle
import uuid
import numpy as np
import multiprocessing
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.lib import xla_client
import cupy
from cupy.cuda import Device
from cupy.cuda.nccl import (
    NcclCommunicator, get_unique_id)
import ray
import ray.util.collective as col
from ray.util.placement_group import get_current_placement_group
from alpa.util import (
    is_ray_node_resource, get_bundle_idx, create_placement_group)
from alpa.monkey_patch import override_get_backend
from alpa.collective.worker_nccl_util_cupy import xla_buffer_to_cupy
from alpa.collective.collective import (
    _check_single_tensor_input, _check_rank_valid
)
from alpa.collective.collective_group.nccl_collective_group import _check_gpu_tensors
from alpa.collective.collective_group.nccl_util import (
    get_tensor_ptr, get_tensor_n_elements, get_nccl_tensor_dtype,
    get_nccl_runtime_version, get_tensor_device
)
# import cupy.cuda.nccl as nccl

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxpr.utils import (
    CollectiveCommConfigs, P2PCommConfigs, is_power_of, remove_all, XLA_AUTO_TUNE_LEVEL, 
    NCCL_USE_MULTISTREAM, TOTAL_COMM_SIZE, MIN_COMM_SIZE, MAX_COMM_SIZE, 
    MAX_COMM_SIZE_P2P, MAX_COMM_SIZE_LOW_BW, THRE_COMM_SIZE, THRE_COMM_SIZE_P2P, 
    THRE_COMM_SIZE_LOW_BW, MAX_COMM_SIZE_INTERVAL, MAX_COMM_SIZE_INTERVAL_P2P, 
    MAX_COMM_SIZE_INTERVAL_LOW_BW, LOW_BW_GPU_TYPES, REPEAT_TIMES_EACH_COMM_SIZE, 
    BARRIER_INTERVAL_INTRA_HOST, BARRIER_INTERVAL_INTER_HOSTS, MAX_DEVICE_NUM_PER_HOST)


class HostCommProfileWorker:
    """ A ray actor in the communication group on each host to profile communication operators. """
    def __init__(self, server_addr: str, num_hosts: int, num_devices_per_host: int, host_id: int):
        # Address of the xla grpc server
        self.server_addr = server_addr
        # Global device cluster
        self.num_hosts = num_hosts
        self.num_global_devices = num_hosts * num_devices_per_host
        # Host id of this worker
        self.host_id = host_id
        # Distributed xla client of this host
        self.distributed_client = (
            xla_client._xla.get_distributed_runtime_client(server_addr, host_id, 
                                                           use_coordination_service=False)
        )
        print(f"[I] Connecting to XLA runtime with the address of {server_addr}...")
        self.distributed_client.connect()
        print(f"[I] The XLA runtime has been connected.")
        # Backend
        self.backend = xla_client.make_gpu_client(self.distributed_client, node_id=host_id)
        assert num_devices_per_host <= len(self.backend.local_devices()), \
            f"There are {len(self.backend.local_devices())} devices on the host, " + \
            f"while {num_devices_per_host} devices are requested."
        self.local_devices = self.backend.local_devices()[:num_devices_per_host]
        self.num_local_devices = len(self.local_devices)
    
    def init_nccl_communicator(self, comm_cfgs: CollectiveCommConfigs):
        """ Profile communication operators in the communication group. """
        print(f"[I] Initializing NCCL communicator on host (id = {self.host_id})...")
        return init_nccl_communicator(comm_cfgs, self.backend, self.local_devices, self.num_global_devices)
    
    def _profile(self, comm_cfgs: CollectiveCommConfigs):
        """ Profile communication operators in the communication group. """
        print(f"[I] Profiling on host (id = {self.host_id})...")
        return profile_comm_op_one_config_collective(comm_cfgs, self.backend, self.local_devices, self.num_global_devices)

    def profile(self, comm_cfgs_list: Sequence[CollectiveCommConfigs]):
        """ Profile communication operators in the communication group. """
        # print(f"[I] Profiling on host (id = {self.host_id})...")
        return profile_comm_op_multi_configs_collective(comm_cfgs_list, self.backend, self.local_devices, self.num_global_devices)
    
    def sync(self):
        """ Sync activities on local devices of the underlying host. """
        # Only need to sync one device due to spmd
        self.local_devices[0].synchronize_all_activity()

    def shutdown(self):
        """ Shutdown the lanuched profiling worker. """
        self.sync()
        self.distributed_client.shutdown()


class DeviceP2PCommProfileWorker:
    """ A ray actor on each device to profile p2p send/recv communication operators. """
    def __init__(self, num_hosts: int, num_devices: int, rank: int, server_addr: str = None):
        self.num_hosts = num_hosts
        self.num_devices = num_devices
        self.p2p_rank = rank
        self.is_sender = (rank == 0)
        # Buffer to be translated
        self.buffer = None

        if num_hosts == 1:
            # Intra-host
            self.backend = override_get_backend()
            # self.local_devices = self.backend.local_devices()[:num_devices] if self.is_sender \
            #                         else self.backend.local_devices()[num_devices:]
            self.local_devices = self.backend.local_devices()
            assert len(self.local_devices) == num_devices, \
                f"Expected {num_devices} devices allocated, got {len(self.local_devices)} devices."
        else:
            # Inter-hosts
            self.distributed_client = (
                xla_client._xla.get_distributed_runtime_client(server_addr, rank, 
                                                            use_coordination_service=False)
            )
            print(f"[I] Connecting to XLA runtime with the address of {server_addr}...")
            self.distributed_client.connect()
            print(f"[I] The XLA runtime has been connected.")
            # Backend
            self.backend = xla_client.make_gpu_client(self.distributed_client, node_id=rank)
            assert num_devices <= len(self.backend.local_devices()), \
                f"There are {len(self.backend.local_devices())} devices on the host, " + \
                f"while {num_devices} devices are requested."
            self.local_devices = self.backend.local_devices()[:num_devices]

    def buffer_from_pyval(self, data_shape: Sequence[int], ndtype: Any, gpu_idx: int):
        """ Write in the local buffer. """
        _xla_buffer = self.backend.buffer_from_pyval(
            np.empty(data_shape, ndtype), self.local_devices[gpu_idx])
        # To cupy array
        _take_ownership = not self.is_sender
        self.buffer = xla_buffer_to_cupy(_xla_buffer, _take_ownership)
    
    def init(self, world_size: int):
        """ Init communication group. """
        assert self.p2p_rank == 0 or self.p2p_rank == 1, \
            f"Worker rank should be either 0 (sender) or 1 (receiver), got {self.p2p_rank}."
        col.init_collective_group(world_size, self.p2p_rank, backend="nccl", group_name="p2p_group") 
    
    def send(self, dst_rank: int, dst_gpu_idx: int = 0):
        """ Send cupy buffer to the dst gpu rank with blocking. """
        assert self.buffer is not None, \
            "Call obj.buffer_from_pyval() first to buffer the target tensor before communication."
        if self.num_devices == 1:
            col.send(self.buffer, dst_rank, "p2p_group")
        else:
            col.send_multigpu(self.buffer, dst_rank, dst_gpu_idx, "p2p_group")
    
    def recv(self, src_rank: int, src_gpu_idx: int = 0):
        """ Receive cupy buffer from the dst gpu rank with blocking. """
        assert self.buffer is not None, \
            "Call obj.buffer_from_pyval() first to buffer the target tensor before communication."
        if self.num_devices == 1:
            col.recv(self.buffer, src_rank, "p2p_group")
        else:
            col.recv_multigpu(self.buffer, src_rank, src_gpu_idx, "p2p_group")
    
    def sync(self):
        """ Sync activities on local devices of the underlying host. """
        for _d in self.local_devices:
            _d.synchronize_all_activity()
    
    def flush_buffer(self):
        self.buffer = None


############################################
#            Utility Functions             #
############################################

def _create_channel_id(backend: Any):
    channel_id = backend.create_channel_handle()
    channel_id.type = xe.ChannelHandle_ChannelType.DEVICE_TO_DEVICE
    channel_id.handle = 1
    return channel_id


def _bound(value, minimum, maximum):
    return max(min(value, maximum), minimum)


def init_backend():
    """ Initializing Ray Cluster & Alpa backend. """
    # Connect to or construct a ray cluster
    if not ray.is_initialized():
        try:
            ray.init(address="auto")
        except:
            raise ConnectionError("Currently only support pre-init multi-hosts Ray cluster.")


def init_nccl_communicator(comm_cfgs: CollectiveCommConfigs, backend: Any, 
                           local_devices: Sequence[Any], num_devices: int):
    """
    Initialize a nccl communicator based on the given replica groups with dummy input 
    to avoid nccl/xla deadlock described in alpa/mesh_profiling/profile_one_hlo_op(). 
    """
    replica_groups = comm_cfgs.replica_groups
    data_shape = comm_cfgs.data_shape
    data_type = comm_cfgs.data_type
    print(f"[I] Initializing NCCL communicator for replica groups: {replica_groups}")
    # Builder
    builder = xc.XlaBuilder("nccl_init")
    # Operands
    shape = xc.Shape.array_shape(data_type, data_shape)
    _replicated = list()
    operands = [xops.Parameter(builder, 0, shape.with_major_to_minor_layout_if_absent(), 
                            "op_1", _replicated),
                xops.Parameter(builder, 1, shape.with_major_to_minor_layout_if_absent(), 
                            "op_2", _replicated)
                ]
    
    def op_impl(_operands):
        # Replica groups
        _replica_groups = xc.make_replica_groups(replica_groups)
        # Channel id
        _channel_id = _create_channel_id(backend)
        # AllReduce operator
        xla_computation = xc.XlaBuilder("add")
        xops.Add(xops.Parameter(xla_computation, 0, shape), 
                xops.Parameter(xla_computation, 1, shape))
        _manual_partition = True
        allreduce_op = xops.AllReduce(_operands[0], xla_computation.build(), _replica_groups, 
                                    _channel_id, None, _manual_partition)
        _operands[-1] = allreduce_op
    
    # Sharding
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])

    # Sharding tensors and build computation
    builder.set_sharding(sharding)
    op_impl(operands)
    builder.clear_sharding()
    # XLA exploits tuple to store results before compilation
    xops.Tuple(builder, operands)
    for _i in range(len(operands)):
        builder.setup_alias((_i,), _i, ())
    xla_computation = builder.build()

    # Compile
    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )
    compiled = backend.compile(xla_computation, compile_options)

    # Dispatch inputs to each device
    device_inputs = list()
    for _ in range(len(operands)):
        device_inputs.append([
            backend.buffer_from_pyval(
                np.ones(data_shape, xc.XLA_ELEMENT_TYPE_TO_DTYPE[data_type]), 
                _device,
            )
            for _device in local_devices
        ])

    # Execute
    for _device in local_devices:
        _device.synchronize_all_activity()
    device_outputs = compiled.execute_sharded_on_local_devices(device_inputs)
    for _device in local_devices:
        _device.synchronize_all_activity()
    print("[I] NCCL communicator has been initialized.")


def get_crossed_host_num_of_replica_groups(replica_groups: Sequence[Sequence[int]], 
                                           replica_to_device_mapping: Sequence[Sequence[int]]):
    """ Get the number of crossed hosts for each group in the given replica groups. """
    host_occupied_flags = [0 for _ in range(len(replica_to_device_mapping))]
    for _replica_id in replica_groups[0]:
        # Only consider the first group since each group should be the same when 
        # considering hosts crossing situations.
        for _i, _replica_ids_one_host in enumerate(replica_to_device_mapping):
            # Locate which host this replica belongs to
            if _replica_id in _replica_ids_one_host:
                host_occupied_flags[_i] = 1
                break
    return np.sum(host_occupied_flags)


def enumerate_all_communication_size():
    """ Enumerate all communication size to profile communication operators. """
    all_comm_sizes = list()
    _gpu_type = args.devices_name.split("_")[1]
    _s = MIN_COMM_SIZE

    if args.profile_collective:
        _max_comm_size = MAX_COMM_SIZE if (args.num_hosts == 1 or 
                                        _gpu_type not in LOW_BW_GPU_TYPES) \
                                        else MAX_COMM_SIZE_LOW_BW
        _threshold_comm_size = THRE_COMM_SIZE if (args.num_hosts == 1 or 
                                                _gpu_type not in LOW_BW_GPU_TYPES) \
                                        else THRE_COMM_SIZE_LOW_BW
        _interval = MAX_COMM_SIZE_INTERVAL if (args.num_hosts == 1 or 
                                            _gpu_type not in LOW_BW_GPU_TYPES) \
                                        else MAX_COMM_SIZE_INTERVAL_LOW_BW
    elif args.profile_p2p:
        (_max_comm_size, _threshold_comm_size, 
         _interval) = (MAX_COMM_SIZE_P2P, THRE_COMM_SIZE_P2P, 
                       MAX_COMM_SIZE_INTERVAL_P2P)
    else:
        raise NotImplementedError()

    while _s < _max_comm_size:
        all_comm_sizes.append((_s,))
        _s = _s * 2 if _s < _threshold_comm_size else _s + _interval
    
    return all_comm_sizes


def generate_normalized_replica_groups(replica_to_device_mapping: Sequence[Sequence[int]], 
                                       num_hosts_crossed: int, intra_host_size: int):
    """ Generate one normalized replica groups. """
    replica_to_device_mapping = np.array(replica_to_device_mapping)
    # Tile sizes for each axis
    intra_host_tile_size = len(replica_to_device_mapping[0]) // intra_host_size
    inter_hosts_tile_size = len(replica_to_device_mapping) // num_hosts_crossed
    # Slice inter hosts
    inter_hosts_sliced = np.split(replica_to_device_mapping, 
                                  inter_hosts_tile_size, 
                                  axis=0)
    # Slice intra host
    intra_host_sliced = list()
    for _sliced in inter_hosts_sliced:
        intra_host_sliced.extend(
            np.split(_sliced, intra_host_tile_size, axis=1)
        )
    return [list(_s.reshape((-1,))) for _s in intra_host_sliced]


def enumerate_all_replica_groups(num_hosts: int, num_devices_per_host: int, only_best_locality: bool = False):
    """ Enumerate all possible replica groups with all gpus within single host. 
    
    Args:
        num_hosts (int): The number of hosts.
        num_devices_per_host (int): The number of devices per host.
        only_best_locality (bool): Whether to only consider the best locality of replica groups (e.g., [[0, 1], [2, 3]]).
    """

    num_global_devices = num_hosts * num_devices_per_host
    # Generate the device mapping from replica ids to devices. The definition of device mapping
    # is explained in `./runtime_profiler/_get_stage_submeshes_and_map_replicas_to_devices()`
    replica_to_device_mapping = np.arange(num_global_devices).reshape((num_hosts, num_devices_per_host))

    def __is_best_locality(groups: List[List[int]]):
        """ Check whether the replica groups are the best locality. """
        if len(groups[0]) > num_devices_per_host:
            # Cross-host group is not best-locality
            return False
        
        for group in groups:
            if any(group[_i + 1] != group[_i] + 1 for _i in range(len(group) - 1)):
               return False
        return True

    # Enumerate
    replica_groups_list = list()
    _group_size = 2
    while _group_size <= num_global_devices:
        _groups_list = list()
        if num_hosts == 1:
            # Symmetric
            _group_num = num_global_devices // _group_size
            _groups_list.append(
                [[_i * _group_size + _j for _j in range(_group_size)] for _i in range(_group_num)]
            )
        else:
            # Asymmetric due to varied bandwidth intra/inter hosts
            num_subgroups = min(num_hosts, _group_size)
            for _num_hosts_crossed in range(1, num_subgroups + 1, 1):
                # Replica num intra host
                _intra_host_size = _group_size // _num_hosts_crossed
                if not is_power_of(2, _num_hosts_crossed) or _intra_host_size > num_devices_per_host:
                    # Skip if not the power of 2 or 1
                    continue
                
                # Generate replica groups based on the number of hosts crossed
                groups = generate_normalized_replica_groups(replica_to_device_mapping, _num_hosts_crossed, _intra_host_size)
                if not only_best_locality or __is_best_locality(groups):
                    _groups_list.append(groups)
        
        replica_groups_list.extend(_groups_list)
        _group_size *= 2

    return replica_groups_list


def gen_replica_id_to_device_mapping(num_hosts: int, num_devices_per_host: int):
    """ Generate the mapping from replica id to devices in hosts. """
    return [
        list(_arr) 
            for _arr in list(
                np.arange(
                    num_hosts * num_devices_per_host
                ).reshape((num_hosts, num_devices_per_host))
            )
    ]


def _get_comm_group_from_one_subgroup(group: Sequence[int], num_hosts: int, 
                                      replica_to_device_mapping: Sequence[Sequence[int]]):
    """ Get comm group from one subgroup inside the replica groups. """
    _replica_num_host = [0 for _ in range(num_hosts)]
    for _replica_id in group:
        for _i, _host in enumerate(replica_to_device_mapping):
            if _replica_id in _host:
                _replica_num_host[_i] += 1
                break
    _replica_num_host = remove_all(_replica_num_host, [0])
    return (len(_replica_num_host), _replica_num_host[0])


def estimate_nonprof_comm(num_hosts: int, intra_host_size: int, 
                          gpu_type: str, comm_data_pth: str):
    """ 
    Estimate non-profiled multi-hosts comm_time_table based on corresponding subgroups 
    from larger comm_group. 
    """
    # Load profiled data to estimate
    num_devices_per_host = intra_host_size
    while num_devices_per_host <= MAX_DEVICE_NUM_PER_HOST and not os.path.exists(
        os.path.join(comm_data_pth, 
                     f"{num_hosts}_{gpu_type}_{num_hosts}_n_{num_devices_per_host}_d.pkl")
    ):
        num_devices_per_host *= 2
    assert num_devices_per_host <= MAX_DEVICE_NUM_PER_HOST, \
        f"Seems that no communication data is offline profiled with {num_hosts} hosts."
    _comm_file_name = f"{num_hosts}_{gpu_type}_{num_hosts}_n_{num_devices_per_host}_d.pkl"
    _comm_data_pth = os.path.join(comm_data_pth, _comm_file_name)
    print(f"[WARN] To estimate non-profiled data ({num_hosts}, {intra_host_size}), " +
          f"Loading offline profiled communication data of GPU type `{gpu_type}` " + 
          f"with {num_hosts} hosts and {num_devices_per_host} devices per host...")
    with open(_comm_data_pth, "rb") as f:
        _comm_time_table = pickle.load(f)
    
    # Candidate replica groups
    replica_groups_list = enumerate_all_replica_groups(num_hosts, intra_host_size)
    # Replica id to device mapping
    prof_r2d_mapping = gen_replica_id_to_device_mapping(num_hosts, num_devices_per_host)
    nonprof_r2d_mapping = gen_replica_id_to_device_mapping(num_hosts, intra_host_size)
    # Estimated comm time table
    comm_time_table = dict()

    def __parse_key(key: str):
        """ Parse key of comm time table to get op type and replica groups. """
        # ('send', [[0, 1], [2, 3]]) -> 'send', [[0, 1], [2, 3]]
        strs = key.split("',")
        op_type = strs[0].split("'")[1]
        tmp = strs[1].replace(" ", "").replace("[", "").replace("]])", "").split("],")
        replica_groups = [[int(_c) for _c in _s.split(",")] for _s in tmp]
        return op_type, replica_groups

    for _dst_replica_groups in replica_groups_list:
        # Match the replica groups in profiled comm time table with the same comm group
        _dst_comm_group = _get_comm_group_from_one_subgroup(_dst_replica_groups[0], num_hosts, 
                                                            nonprof_r2d_mapping)
        for _key in _comm_time_table:
            _op_type, _replica_groups = __parse_key(_key)
            _comm_group = _get_comm_group_from_one_subgroup(_replica_groups[0], num_hosts, 
                                                            prof_r2d_mapping)
            if _dst_comm_group == _comm_group:
                _dst_key = str((_op_type, _dst_replica_groups))
                assert _dst_key not in comm_time_table, \
                    f"Rewrite key `{_dst_key}` in the target comm time table."
                comm_time_table[_dst_key] = list()
                for _rec in _comm_time_table[_key]:
                    comm_time_table[_dst_key].append([_rec[0], _rec[1], _rec[2], 
                                                      _dst_replica_groups])
    
    return comm_time_table


def _init_xla_grpc_server(num_hosts: int, num_devices_per_worker: int):
    """ Initialize xla grpc server in multi-hosts scenarios. """
    # Address of xla grpc server
    server_port = np.random.randint(20000, 25000)
    server_addr = f"{ray.util.get_node_ip_address()}:{server_port}"
    print(f"[I] Initializing gRPC server of XLA runtime on address: {server_addr}...")
    service_server = xla_client._xla.get_distributed_runtime_service(server_addr, num_hosts, 
                                                                     use_coordination_service=False)
    print(f"[I] The XLA gRPC server has been initialized.")
    time.sleep(1)
    # Init backend
    print(f"[I] Initializing Ray cluster...")
    init_backend()
    print(f"[I] The device cluster has been initialized.")
    # Host infos
    host_infos, host_ips = list(), list()
    for _node in ray.nodes():
        for _key in _node["Resources"]:
            if (is_ray_node_resource(_key)):
                host_infos.append(_node)
                host_ips.append(_key.split("node:")[-1])
    # Placement group
    pg_name = "default_pg"
    try:
        pg = ray.util.get_placement_group(pg_name)
    except ValueError:
        pg = None
    placement_group = pg if pg \
        else create_placement_group(num_hosts, [num_devices_per_worker] * num_hosts, pg_name)
    # Bundle index list
    device_bundle_idx_list = get_bundle_idx(placement_group, host_ips)

    return server_addr, service_server, placement_group, device_bundle_idx_list


def _shutdown_all(num_hosts: int, prof_workers: Sequence[Any], service_server: Any, is_collective: bool):
    """ Shutdown all workers and grpc server. """
    # Shutdown all workers
    print("")
    print(f"[I] Shutting down profiling workers on {num_hosts} hosts and XLA runtime...")
    if is_collective and not args.force_shutdown:
        ray.get([_w.shutdown.remote() for _w in prof_workers])
    for _worker in prof_workers:
        ray.kill(_worker)
    # Shutdown grpc server
    if service_server:
        service_server.shutdown()
    print("[I] All profiling workers and the XLA runtime have been shutted down.")


############################################
#       P2P Send/Recv Communication        #
############################################

def profile_comm_op_one_config_p2p(comm_cfgs: P2PCommConfigs, 
                                   send_worker: Any, recv_worker: Any):
    """ Profile one p2p communication operator with the given config. """
    global global_comm_time_table
    # Communication configs
    data_shape = comm_cfgs.data_shape
    comm_size = np.prod(data_shape)
    data_type = comm_cfgs.data_type
    src_rank, dst_rank = comm_cfgs.src_rank, comm_cfgs.dst_rank
    src_gpu_idx, dst_gpu_idx = comm_cfgs.src_gpu_idx, comm_cfgs.dst_gpu_idx
    
    if data_type == xc.PrimitiveType.F32:
        ndtype = np.float32
    
    # Repeat times to call the communication operator
    repeat_times = _bound(int(TOTAL_COMM_SIZE / max(comm_size * \
                        xc.XLA_ELEMENT_TYPE_TO_DTYPE[data_type].itemsize, 1)), 
                    10, 1 << 5)
    
    # # Buffer to be transfered
    # tensors = [
    #     send_backend.buffer_from_pyval(
    #         np.empty(data_shape, ndtype), send_local_devices[src_gpu_idx]), 
    #     recv_backend.buffer_from_pyval(
    #         np.empty(data_shape, ndtype), recv_local_devices[dst_gpu_idx]), 
    # ]
    # send_buffer, recv_buffer = tensors[0], tensors[1]
    # _global_src_gpu_idx = src_gpu_idx
    # _global_dst_gpu_idx = dst_gpu_idx + dst_rank * len(send_local_devices)
    # assert _global_src_gpu_idx == xe.get_buffer_device_id(send_buffer), \
    #     f"Mismatched src GPU rank {_global_src_gpu_idx} with the device ID ({xe.get_buffer_device_id(send_buffer)}) " + \
    #     f"of the transmitted buffer."
    # assert _global_dst_gpu_idx == xe.get_buffer_device_id(recv_buffer), \
    #     f"Mismatched dst GPU rank {_global_dst_gpu_idx} with the device ID ({xe.get_buffer_device_id(recv_buffer)}) " + \
    #     f"of the transmitted buffer."
    # # Cupy array
    # to_send = xla_buffer_to_cupy(send_buffer)
    # to_recv = xla_buffer_to_cupy(recv_buffer, take_ownership=True)
    
    # # Warmup
    # warmup = _bound(repeat_times // 10, 2, 10)
    # for _ in range(warmup):
    #     ray.get([
    #         send_worker.send.remote(to_send, dst_rank, dst_gpu_idx),
    #         recv_worker.recv.remote(to_recv, src_rank, src_gpu_idx)
    #     ])    
    # for _device in local_devices:
    #     _device.synchronize_all_activity()

    # # Send/recv operations
    # _time_marker = time.time()
    # for _ in range(repeat_times):
    #     ray.get([
    #         send_worker.send.remote(to_send, dst_rank, dst_gpu_idx),
    #         recv_worker.recv.remote(to_recv, src_rank, src_gpu_idx)
    #     ])
    # for _device in local_devices:
    #     _device.synchronize_all_activity()
    # comm_time = (time.time() - _time_marker) / repeat_times

    # Buffer in
    ray.get([
        send_worker.buffer_from_pyval.remote(data_shape, ndtype, src_gpu_idx),
        recv_worker.buffer_from_pyval.remote(data_shape, ndtype, dst_gpu_idx)
    ])
    
    # Warmup
    warmup = _bound(repeat_times // 10, 10, 1 << 5)
    for _ in range(warmup):
        ray.get([
            send_worker.send.remote(dst_rank, dst_gpu_idx),
            recv_worker.recv.remote(src_rank, src_gpu_idx)
        ])    
    ray.get([_w.sync.remote() for _w in [send_worker, recv_worker]])

    # Send/recv operations
    _time_marker = time.time()
    # for _ in range(1):   
    for _ in range(repeat_times):
        ray.get([
            send_worker.send.remote(dst_rank, dst_gpu_idx),
            recv_worker.recv.remote(src_rank, src_gpu_idx)
        ])   
        ray.get([_w.sync.remote() for _w in [send_worker, recv_worker]])
    comm_time = (time.time() - _time_marker) / repeat_times
    # comm_time = (time.time() - _time_marker)

    # Flush buffer
    ray.get([_w.flush_buffer.remote() for _w in [send_worker, recv_worker]])

    return (data_shape, xc.XLA_ELEMENT_TYPE_TO_DTYPE[data_type], comm_time, None)


def profile_comm_intra_host_p2p():
    """ Profile p2p send/recv operators intra host. """
    # Data type
    data_type = xc.PrimitiveType.F32
    # Nccl configs
    world_size = 2
    src_rank, dst_rank = 0, 1                       # Src and dst rank
    src_gpu_idx, dst_gpu_idx = 0, 0                 # Dst gpu index in each worker
    num_devices_per_worker = args.num_devices_per_host if args.num_hosts > 1 else args.num_devices_per_host // 2
    print(f"[I] Allocate {num_devices_per_worker} devices on each worker.")
    # Read existed comm data
    comm_log_pth = "./jaxpr/comm_data"
    if not os.path.exists(comm_log_pth):
        os.mkdir(comm_log_pth)
    _file_name = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d.pkl"
    pth = os.path.join(comm_log_pth, _file_name)
    if os.path.exists(pth):
        print(f"[TMP] Existed profiled communication data in `{pth}`, updating it...")
        with open(pth, "rb") as f:
            global_comm_time_table = pickle.load(f)
    else:
        print(f"[TMP] Profiled communication data not found in `{pth}`, creating it...")
        global_comm_time_table = dict()
    if "send" not in global_comm_time_table.keys() or args.overwrite_data:
        global_comm_time_table["send"] = list()
    else:
        print("[TMP] Cache is loaded. If you want to overwrite previously profiled data, use `--overwrite_data`.")
        return
    # Init backend
    print(f"[I] Initializing Ray cluster...")
    init_backend()
    print(f"[I] The device cluster has been initialized.")
    
    # Send/recv workers
    print("[I] Lanuching Ray actors as device P2P send/recv communication profiler...")
    cls = ray.remote(num_cpus=0, 
                     num_gpus=num_devices_per_worker)(DeviceP2PCommProfileWorker)
    send_worker = cls.remote(1, num_devices_per_worker, src_rank)
    recv_worker = cls.remote(1, num_devices_per_worker, dst_rank)
    # Init comm group
    init_events = list()
    for _worker in [send_worker, recv_worker]:
        init_events.append(_worker.init.remote(world_size))
    ray.get(init_events)
    # # Backend
    # backend = override_get_backend()
    # send_local_devices, recv_local_devices = backend.local_devices()[:num_devices_per_worker], \
    #                                          backend.local_devices()[num_devices_per_worker:]
    print("[I] Workers have been lanuched and initialized.")
    
    # All communication sizes
    all_comm_sizes = enumerate_all_communication_size()
    
    # Send/recv operations
    for _i, _comm_size in enumerate(all_comm_sizes):
        avg_comm_times = list()
        _shape, _dtype = None, None
        _comm_cfgs = P2PCommConfigs(data_shape=_comm_size, data_type=data_type, 
                                    src_rank=src_rank, dst_rank=dst_rank,
                                    src_gpu_idx=src_gpu_idx, dst_gpu_idx=dst_gpu_idx)
        
        for _ in range(REPEAT_TIMES_EACH_COMM_SIZE):
            (_shape, _dtype, _comm_time, _) = profile_comm_op_one_config_p2p(_comm_cfgs, 
                                                                             send_worker, recv_worker)
            avg_comm_times.append(_comm_time)
    
        print(f"[TMP] Profiled comm times: {avg_comm_times}")
        print(f"[TMP] Average comm time (s): {np.mean(avg_comm_times)}")

        global_comm_time_table["send"].append([_shape, str(_dtype), np.mean(avg_comm_times), None])

        print(f"[I] ({_i}/{len(all_comm_sizes)}) OP type: send | Data shape: {_shape} | " + 
              f"Sender: worker {src_rank} (GPU {src_gpu_idx}) | Receiver: worker {dst_rank} (GPU {dst_gpu_idx}) | " +
              f"| Time cost (s): {np.mean(avg_comm_times)}")
    
    # Destroy actors
    ray.kill(send_worker)
    ray.kill(recv_worker)

    # Store communication time data
    print("")
    print(f"[TMP] Writing profiled communication data to '{pth}'...")
    with open(pth, "wb") as f:
        pickle.dump(global_comm_time_table, f)
    
    print("")
    print("[I] All communication operators have been profiled.")


def profile_comm_inter_hosts_p2p():
    """ Profile p2p send/recv operators inter hosts. """
    # Data type
    data_type = xc.PrimitiveType.F32
    # Nccl configs
    world_size = 2
    src_rank, dst_rank = 0, 1                       # Src and dst rank
    src_gpu_idx, dst_gpu_idx = 0, 0                 # Dst gpu index in each worker
    num_devices_per_worker = args.num_devices_per_host if args.num_hosts > 1 else args.num_devices_per_host // 2
    print(f"[I] Allocate {num_devices_per_worker} devices on each worker.")
    # Read existed comm data
    comm_log_pth = "./jaxpr/comm_data"
    if not os.path.exists(comm_log_pth):
        os.mkdir(comm_log_pth)
    _file_name = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d.pkl"
    pth = os.path.join(comm_log_pth, _file_name)
    if os.path.exists(pth):
        print(f"[TMP] Existed profiled communication data in `{pth}`, updating it...")
        with open(pth, "rb") as f:
            global_comm_time_table = pickle.load(f)
    else:
        print(f"[TMP] Profiled communication data not found in `{pth}`, creating it...")
        global_comm_time_table = dict()
    # Overwrite or return
    if "send" not in global_comm_time_table.keys() or args.overwrite_data:
        global_comm_time_table["send"] = list()
    else:
        print("[TMP] Cache is loaded. If you want to overwrite previously profiled data, use `--overwrite_data`.")
        return
    # Device cluster
    num_hosts = int(args.num_hosts)
    # Init xla grpc server
    (server_addr, service_server, placement_group, 
     device_bundle_idx_list) = _init_xla_grpc_server(num_hosts, num_devices_per_worker)
    
    # Send/recv workers
    print("[I] Lanuching Ray actors as device P2P send/recv communication profiler...")
    prof_workers = list()
    backends, local_devices_list = list(), list()
    for _i in range(num_hosts):
        # Worker
        worker_name = f"host_worker_{_i}"
        # assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ, \
        #     f"XLA_PYTHON_CLIENT_MEM_FRACTION is not set in os.environ."
        # Set XLA environment variables
        env_vars = {
            "NCCL_USE_MULTISTREAM":
                "False",
            "XLA_PYTHON_CLIENT_MEM_FRACTION":
                os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8"),
            "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "") +
                            f" --xla_gpu_autotune_level"
                            f"={os.environ.get('XLA_AUTO_TUNE_LEVEL')}"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        }
        # Bundle index
        bundle_index = device_bundle_idx_list[_i]
        # Lanuch
        cls = ray.remote(num_cpus=0, 
                         num_gpus=num_devices_per_worker)(DeviceP2PCommProfileWorker)
        worker = cls.options(placement_group=placement_group, 
                             placement_group_bundle_index=bundle_index,
                             name=worker_name,
                             runtime_env={"env_vars": env_vars}
                             ).remote(num_hosts, num_devices_per_worker, _i, server_addr)
        prof_workers.append(worker)
    # Init comm group
    init_events = list()
    for _worker in prof_workers:
        init_events.append(_worker.init.remote(world_size))
    ray.get(init_events)
    print("[I] Workers have been lanuched and initialized.")

    # All communication sizes
    all_comm_sizes = enumerate_all_communication_size()
    
    # Send/recv operations
    for _i, _comm_size in enumerate(all_comm_sizes):
        avg_comm_times = list()
        _shape, _dtype = None, None
        _comm_cfgs = P2PCommConfigs(data_shape=_comm_size, data_type=data_type, 
                                    src_rank=src_rank, dst_rank=dst_rank,
                                    src_gpu_idx=src_gpu_idx, dst_gpu_idx=dst_gpu_idx)
        for _ in range(REPEAT_TIMES_EACH_COMM_SIZE):
            (_shape, _dtype, _comm_time, _) = profile_comm_op_one_config_p2p(_comm_cfgs, 
                                                                             prof_workers[0],
                                                                             prof_workers[1])
            avg_comm_times.append(_comm_time)
    
        print(f"[TMP] Profiled comm times: {avg_comm_times}")
        print(f"[TMP] Average comm time (s): {np.mean(avg_comm_times)}")

        global_comm_time_table["send"].append([_shape, str(_dtype), np.mean(avg_comm_times), None])

        print(f"[I] ({_i}/{len(all_comm_sizes)}) OP type: send | Data shape: {_shape} | " + 
              f"Sender: worker {src_rank} (GPU {src_gpu_idx}) | Receiver: worker {dst_rank} (GPU {dst_gpu_idx}) | " +
              f"| Time cost (s): {np.mean(avg_comm_times)}")
    
    # Shutdown
    for _local_devices in local_devices_list:
        _local_devices[0].synchronize_all_activity()
    _shutdown_all(num_hosts, prof_workers, service_server, is_collective=False)

    # Store communication time data
    print("")
    print(f"[TMP] Writing profiled communication data to '{pth}'...")
    with open(pth, "wb") as f:
        pickle.dump(global_comm_time_table, f)
    
    print("")
    print("[I] All communication operators have been profiled.")


############################################
#         Collective Communication         #
############################################

def profile_comm_op_one_config_collective(comm_cfgs: CollectiveCommConfigs, backend: Any, 
                                          local_devices: Sequence[Any], num_devices: int):
    """ Profile one collective communication operator with the given config. """
    global global_comm_time_table
    # Communication configs
    op_type = comm_cfgs.op_type
    replica_groups = comm_cfgs.replica_groups
    data_shape = comm_cfgs.data_shape
    data_type = comm_cfgs.data_type
    # Builder
    builder = xc.XlaBuilder("profile_comm")
    # Communication size (in output shape)
    comm_size = np.prod(data_shape)
    # Int type
    if data_type == xc.PrimitiveType.F32:
        int_type = np.int32
    elif data_type == xc.PrimitiveType.F16:
        int_type = np.int16
    else:
        raise RuntimeWarning(f"Current data type ({data_type}) is not supported.")
    
    # Operand shapes
    shapes = None
    # if op_type == "send":
    #     # TODO(chunyu): Temporarily implement as an allgather, further implementation
    #     #               needs to exploit cupy.cuda.nccl.NcclCommunicator.
    #     #               When estimating with these offline profiled data, we should divide the 
    #     #               estimated comm time of "send" with 2 since p2p send operation is one-way
    #     #               while all-gather operation is two-way.
    #     comm_size = comm_size // len(replica_groups[0]) * len(replica_groups[0])
    #     shapes = [((comm_size // len(replica_groups[0]),), data_type), ((comm_size,), data_type)]
    # elif op_type == "all-reduce":
    if op_type == "all-reduce":
        shapes = [((comm_size,), data_type), ((comm_size,), data_type)]
    elif op_type == "reduce-scatter":
        comm_size = comm_size // len(replica_groups[0]) * len(replica_groups[0])
        shapes = [((comm_size,), data_type), ((comm_size // len(replica_groups[0]),), data_type)]
    elif op_type == "all-gather":
        comm_size = comm_size // len(replica_groups[0]) * len(replica_groups[0])
        shapes = [((comm_size // len(replica_groups[0]),), data_type), ((comm_size,), data_type)]
    elif op_type == "all-to-all":
        comm_size = comm_size // (len(replica_groups[0])**2) * (len(replica_groups[0])**2)
        shapes = [((comm_size,), data_type), ((comm_size,), data_type)]
        # shapes = [((comm_size // len(replica_groups[0]),), data_type),
        #           ((comm_size // len(replica_groups[0]),), data_type)]
    elif op_type == "barrier":
        shapes = [((1,), data_type), ((1,), data_type)]
    
    # Repeat times to call the communication operator
    _gpu_type = args.devices_name.split("_")[1]
    _max_repeat_times = 1 << 8 if _gpu_type not in LOW_BW_GPU_TYPES else 1 << 4
    _min_repeat_times = 10 if _gpu_type not in LOW_BW_GPU_TYPES else 5
    repeat_times = _bound(int(TOTAL_COMM_SIZE / max(comm_size * \
                        xc.XLA_ELEMENT_TYPE_TO_DTYPE[data_type].itemsize, 1)), 
                    _min_repeat_times, _max_repeat_times)

    # Input tuple
    in_tuple_shape = xc.Shape.tuple_shape(
        [xc.Shape.array_shape(np.dtype(int_type), ())] +
        [xc.Shape.array_shape(dtype, shape) for shape, dtype in shapes])
    in_tuple = xops.Parameter(builder, 0, in_tuple_shape)
    # Counter
    counter = xops.GetTupleElement(in_tuple, 0)
    counter = xops.Sub(counter, xops.Constant(builder, int_type(1)))
    # Operands
    operands = [
        xops.GetTupleElement(in_tuple, i + 1) for i in range(len(shapes))
    ]
    
    def op_impl(_operands):
        # Replica groups
        _replica_groups = xc.make_replica_groups(replica_groups)
        # Channel id
        _channel_id = _create_channel_id(backend)

        # if op_type == "send":
        #     # Send operator
        #     # TODO(chunyu): Temporarily implement as an allgather, further implementation
        #     #               needs to exploit cupy.cuda.nccl.NcclCommunicator.
        #     if shapes[0][0][0] == 0:
        #         return 
        #     _xla_op = xops.AllGather(_operands[0], 0, len(replica_groups[0]), 
        #                              _replica_groups, _channel_id, None, True)
        # elif op_type == "all-reduce":
        if op_type == "all-reduce":
            # AllReduce operator
            xla_computation = xc.XlaBuilder("add")
            _shape = xc.Shape.array_shape(data_type, ())
            xops.Add(xops.Parameter(xla_computation, 0, _shape), 
                    xops.Parameter(xla_computation, 1, _shape))
            _manual_partition = True
            _xla_op = xops.AllReduce(_operands[0], xla_computation.build(), 
                                     _replica_groups, _channel_id, None, 
                                     _manual_partition)
        elif op_type == "reduce-scatter":
            # ReduceScatter operator
            if shapes[0][0][0] == 0:
                return
            xla_computation = xc.XlaBuilder("add")
            _shape = xc.Shape.array_shape(data_type, ())
            xops.Add(xops.Parameter(xla_computation, 0, _shape), 
                    xops.Parameter(xla_computation, 1, _shape))
            _manual_partition = True
            _xla_op = xops.ReduceScatter(_operands[0], xla_computation.build(), 0, 
                                         len(replica_groups[0]), _replica_groups, 
                                         _channel_id, None, _manual_partition)
        elif op_type == "all-gather":
            # Allgather operator
            if shapes[0][0][0] == 0:
                return 
            _xla_op = xops.AllGather(_operands[0], 0, len(replica_groups[0]), 
                                     _replica_groups, _channel_id, None, True)
        elif op_type == "all-to-all":
            # All-to-all operator
            if shapes[0][0][0] // len(replica_groups[0]) == 0:
                return 
            _xla_op = xops.AllToAll(_operands[0], 0, 0, len(replica_groups[0]),
                                    _replica_groups, _channel_id, None, True)
        elif op_type == "barrier":
            # Barrier to sync 
            xla_computation = xc.XlaBuilder("add")
            _shape = xc.Shape.array_shape(data_type, ())
            xops.Add(xops.Parameter(xla_computation, 0, _shape), 
                    xops.Parameter(xla_computation, 1, _shape))
            _manual_partition = True
            _xla_op = xops.AllReduce(_operands[0], xla_computation.build(), 
                                     _replica_groups, _channel_id, None, 
                                     _manual_partition)
        _operands[-1] = _xla_op

    # Sharding
    sharding = xc.OpSharding()
    sharding.type = sharding.type.REPLICATED
    sharding.tile_assignment_dimensions.extend([1])
    sharding.tile_assignment_devices.extend([0])

    # Sharding tensors and build computation
    builder.set_sharding(sharding)
    op_impl(operands)
    builder.clear_sharding()
    # XLA exploits tuple to store results before compilation
    xops.Tuple(builder, [counter] + operands)
    xla_computation = builder.build()

    # Condition
    cond = xc.XlaBuilder("condition")
    in_tuple = xops.Parameter(cond, 0, in_tuple_shape)
    counter = xops.GetTupleElement(in_tuple, 0)
    # While loop until counter is smaller than 0
    xops.Gt(counter, xops.Constant(cond, int_type(0)))
    cond_computation = cond.Build()

    # While loop
    loop = xc.XlaBuilder("loop")
    _shape = xc.Shape.array_shape(np.dtype(int_type), ())
    _replicated = list()
    counter = xops.Parameter(loop, 0, _shape, "", _replicated)
    operands = [
        xops.Parameter(loop, _i + 1, xc.Shape.array_shape(_dtype, _shape), 
                       "", _replicated)
        for _i, (_shape, _dtype) in enumerate(shapes)
    ]
    while_init = xops.Tuple(loop, [counter] + operands)
    xops.While(cond_computation, xla_computation, while_init)
    for _i in range(len(shapes) + 1):
        loop.setup_alias((_i,), _i, ())
    loop_computation = loop.Build()

    # Compile
    compile_options = xb.get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
    )

    # compile_options = xb.get_compile_options(
    #     num_replicas=num_devices,
    #     num_partitions=1,
    #     device_assignment=np.arange(num_devices).reshape((-1, 1)),
    #     use_spmd_partitioning=True,
    # )

    compiled = backend.compile(loop_computation, compile_options)

    # print(compiled.hlo_modules()[0].to_string(xe.HloPrintOptions.short_parsable()))

    # exit(0)

    # Shapes of counter and data
    all_shapes = [(1, int_type)] + shapes

    # Warmup 
    warmup_times = _bound(repeat_times // 10, 2, 10)

    # Dispatch inputs to each device
    device_inputs = list()
    for _i, (shape, dtype) in enumerate(all_shapes):
        if _i == 0:
            # Counter
            device_inputs.append([
                backend.buffer_from_pyval(
                    int_type(warmup_times),
                    _device,
                )
                for _device in local_devices
            ])
        else:
            # Data
            device_inputs.append([
                backend.buffer_from_pyval(
                    np.ones(shape, xc.XLA_ELEMENT_TYPE_TO_DTYPE[dtype]),
                    _device,
                )
                for _device in local_devices
            ])

    # Warmup execute
    for _device in local_devices:
        _device.synchronize_all_activity()
    device_inputs = compiled.execute_sharded_on_local_devices(device_inputs)
    for _device in local_devices:
        _device.synchronize_all_activity()

    # Profiling input
    device_inputs[0] = [
            backend.buffer_from_pyval(int_type(repeat_times), _device)
            for _device in local_devices
        ]

    # Execute
    for _device in local_devices:
        _device.synchronize_all_activity()
    _time_marker = time.time()
    device_outputs = compiled.execute_sharded_on_local_devices(device_inputs)
    for _device in local_devices:
        _device.synchronize_all_activity()
    comm_time = (time.time() - _time_marker) / repeat_times

    return (data_shape, xc.XLA_ELEMENT_TYPE_TO_DTYPE[data_type], comm_time, replica_groups)


def profile_comm_op_multi_configs_collective(comm_cfgs_list: Sequence[CollectiveCommConfigs], backend: Any, 
                                             local_devices: Sequence[Any], num_devices: int):
    """ Profile multiple communication operators with varying configurations. """
    results = list()
    for _i, _comm_cfgs in enumerate(comm_cfgs_list):
        if _i % BARRIER_INTERVAL_INTER_HOSTS == 0:
            print("      ------------ Barrier -------------")
            _barrier_comm_cfgs = CollectiveCommConfigs(op_type="barrier", replica_groups=[[_j for _j in range(num_devices)]], 
                                             data_shape=(1,), data_type=xc.PrimitiveType.F32)
            profile_comm_op_one_config_collective(_barrier_comm_cfgs, backend, local_devices, num_devices)
        
        print(f"[TMP] ({_i + 1}/{len(comm_cfgs_list)}) OP type: {comm_cfgs_list[0].op_type} | Replica groups: {comm_cfgs_list[0].replica_groups}")
        
        avg_comm_times = list()
        _shape, _dtype, _replica_groups = None, None, None
        for _ in range(REPEAT_TIMES_EACH_COMM_SIZE):
            (_shape, _dtype, _comm_time, _replica_groups) = profile_comm_op_one_config_collective(_comm_cfgs, backend, 
                                                                                       local_devices, num_devices)
            avg_comm_times.append(_comm_time)
        
        print(f"[TMP] Profiled comm times: {avg_comm_times}")
        print(f"[TMP] Average comm time (s): {np.mean(avg_comm_times)}")
        
        results.append((_shape, _dtype, np.mean(avg_comm_times), _replica_groups))
    
    return results


def profile_comm_intra_host_collective():
    """ Offline profile collective communication operators intra host. """
    global global_comm_time_table
    # Data type
    data_type = xc.PrimitiveType.F32
    # Read existed comm data
    comm_log_pth = "./jaxpr/comm_data"
    if not os.path.exists(comm_log_pth):
        os.mkdir(comm_log_pth)
    _file_name = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d.pkl"
    pth = os.path.join(comm_log_pth, _file_name)
    if os.path.exists(pth):
        print(f"[TMP] Existed profiled communication data in `{pth}`, updating it...")
        with open(pth, "rb") as f:
            global_comm_time_table = pickle.load(f)
    else:
        print(f"[TMP] Profiled communication data not found in `{pth}`, creating it...")
        global_comm_time_table = dict()
    # Backend
    assert args.num_devices_per_host <= len(xb.local_devices()), \
        f"There are {len(xb.local_devices())} devices on the host, " + \
        f"while {args.num_devices_per_host} devices are requested."
    local_devices = xb.local_devices()[:args.num_devices_per_host]
    backend = xb.get_device_backend(local_devices[0])
    # backend = xb.get_backend("gpu")
    # local_devices = backend.local_devices()
    num_devices = len(local_devices)
    
    # Enumerate all possible replica groups with all gpus within single host
    replica_groups_list = enumerate_all_replica_groups(1, num_devices)
    # All communication sizes
    all_comm_sizes = enumerate_all_communication_size()

    for _replica_groups in replica_groups_list:
        # Init nccl communicator
        _comm_cfgs = CollectiveCommConfigs(op_type="", replica_groups=_replica_groups, data_shape=(1024,),
                                data_type=data_type)
        init_nccl_communicator(_comm_cfgs, backend, local_devices, num_devices)
        
        # Profile communication operators
        for _op_type in ("all-reduce", "reduce-scatter", "all-gather", "all-to-all"):
            print("")
            print(f"=========== Profiling Operator: {_op_type} | Replica Groups: {_replica_groups} ===========")
            _key = str((_op_type, _replica_groups))
            if _key not in global_comm_time_table.keys() or args.overwrite_data:
                global_comm_time_table[_key] = list()
            else:
                continue

            # if _op_type == "all-gather" and str(("send", _replica_groups)) in global_comm_time_table.keys():
            #     # Use results of send operation to estimate all-gather operation.
            #     print("[WARN] Since we temporarily exploit all-gather to estimate send, directly use the " + 
            #           f"profiling results of send as all-gather, with replica groups = {_replica_groups}.")
            #     for _rec in global_comm_time_table[str(("send", _replica_groups))]:
            #         global_comm_time_table[_key].append(_rec)
            # else:
            #     # Normal case
            for _i, _comm_size in enumerate(all_comm_sizes):
                if _i % BARRIER_INTERVAL_INTRA_HOST == 0:
                    print("      ------------ Barrier -------------")
                    _barrier_comm_cfgs = CollectiveCommConfigs(op_type="barrier", replica_groups=[[_j for _j in range(num_devices)]], 
                                                    data_shape=(1,), data_type=xc.PrimitiveType.F32)
                    profile_comm_op_one_config_collective(_barrier_comm_cfgs, backend, local_devices, num_devices, )

                avg_comm_times = list()
                _shape, _dtype = None, None
                _comm_cfgs = CollectiveCommConfigs(op_type=_op_type, replica_groups=_replica_groups, 
                                        data_shape=_comm_size, data_type=data_type)
                for _ in range(REPEAT_TIMES_EACH_COMM_SIZE):
                    (_shape, _dtype, _comm_time, _replica_groups) = profile_comm_op_one_config_collective(_comm_cfgs, backend, 
                                                                                            local_devices, num_devices)
                    avg_comm_times.append(_comm_time)
                
                print(f"[TMP] Profiled comm times: {avg_comm_times}")
                print(f"[TMP] Average comm time (s): {np.mean(avg_comm_times)}")
                
                global_comm_time_table[_key].append([_shape, str(_dtype), np.mean(avg_comm_times), _replica_groups])

                print(f"[I] ({_i}/{len(all_comm_sizes)}) OP type: {_op_type} | Data shape: {_shape} | " + 
                    f"Mesh shape: (1, {num_devices}) | Replica groups: {_replica_groups} " +
                    f"| Time cost (s): {np.mean(avg_comm_times)}")
        
            # Store communication time data
            print("")
            print(f"[TMP] Writing profiled communication data to '{pth}'...")
            with open(pth, "wb") as f:
                pickle.dump(global_comm_time_table, f)
    
    print("")
    print("[I] All communication operators have been profiled.")


def profile_comm_inter_hosts_collective(only_best_locality: bool = False):
    """ 
    Offline profile collective communication operators inter hosts by exploiting ray framework. 
    """
    global global_comm_time_table
    # Data type
    data_type = xc.PrimitiveType.F32
    # Read existed comm data
    comm_log_pth = "./jaxpr/comm_data"
    if not os.path.exists(comm_log_pth):
        os.mkdir(comm_log_pth)
    _file_name = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d.pkl"
    pth = os.path.join(comm_log_pth, _file_name)
    if os.path.exists(pth):
        print(f"[TMP] Existed profiled communication data in `{pth}`, updating it...")
        with open(pth, "rb") as f:
            global_comm_time_table = pickle.load(f)
    else:
        print(f"[TMP] Profiled communication data not found in `{pth}`, creating it...")
        global_comm_time_table = dict()
    # Device cluster
    num_hosts = int(args.num_hosts)
    num_devices_per_host = int(args.num_devices_per_host)
    num_global_devices = num_hosts * num_devices_per_host
    # Init xla grpc server
    (server_addr, service_server, placement_group, 
     device_bundle_idx_list) = _init_xla_grpc_server(num_hosts, num_devices_per_host)
    
    # Lanuch profile worker on each host
    print("[I] Lanuching Ray actors as profiling workers...")
    prof_workers = list()
    for _i in range(num_hosts):
        worker_name = f"host_worker_{_i}"
        # assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ, \
        #     f"XLA_PYTHON_CLIENT_MEM_FRACTION is not set in os.environ."
        # Set XLA environment variables
        env_vars = {
            "NCCL_USE_MULTISTREAM":
                "False",
            "XLA_PYTHON_CLIENT_MEM_FRACTION":
                os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8"),
            "XLA_FLAGS": (os.environ.get("XLA_FLAGS", "") +
                            f" --xla_gpu_autotune_level"
                            f"={os.environ.get('XLA_AUTO_TUNE_LEVEL')}"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        }
        # Bundle index
        bundle_index = device_bundle_idx_list[_i]
        # Lanuch
        cls = ray.remote(num_cpus=0, 
                         num_gpus=num_devices_per_host)(HostCommProfileWorker)
        worker = cls.options(placement_group=placement_group, 
                             placement_group_bundle_index=bundle_index,
                             name=worker_name,
                             runtime_env={"env_vars": env_vars}
                             ).remote(server_addr, num_hosts, 
                                      num_devices_per_host, _i)
        prof_workers.append(worker)
    print(f"[I] All profiling workers has been lanuched.")

    # Enumerate all possible replica groups with all gpus among multiple hosts
    replica_groups_list = enumerate_all_replica_groups(num_hosts, num_devices_per_host, only_best_locality)
    # All communication sizes
    all_comm_sizes = enumerate_all_communication_size()

    for _replica_groups in replica_groups_list:    
        # Initialize nccl communicator
        print("")
        print(f"[I] Initialzing NCCL communicator among {num_hosts} hosts with " +
            f"totally {num_global_devices} GPUs...")
        init_tasks = list()
        _comm_cfgs = CollectiveCommConfigs(op_type="", replica_groups=_replica_groups, data_shape=(1024,),
                                data_type=data_type)
        for _worker in prof_workers:
            init_tasks.append(
                _worker.init_nccl_communicator.remote(_comm_cfgs)
            )
        ray.get(init_tasks)
        # # Sync all workers
        # ray.get([_w.sync.remote() for _w in prof_workers])
        print("[I] NCCL communicator has been initialized.")
        
        # Profile communication operators on workers    
        print("")
        print(f"[I] Profiling communication operators among {num_hosts} hosts with " +
              f"totally {num_global_devices} GPUs...")
        for _op_type in ("all-reduce", "reduce-scatter", "all-gather", "all-to-all"):
            print("")
            print(f"=========== Profiling Operator: {_op_type} | Replica Groups: {_replica_groups} ===========")
            _key = str((_op_type, _replica_groups))
            if _key not in global_comm_time_table.keys() or args.overwrite_data:
                global_comm_time_table[_key] = list()
            else:
                continue
            
            # Lanuch all tasks with same op type and replica groups
            comm_cfgs_list = list()
            for _i, _comm_size in enumerate(all_comm_sizes):
                comm_cfgs_list.append(
                    CollectiveCommConfigs(op_type=_op_type, replica_groups=_replica_groups, 
                                data_shape=_comm_size, data_type=data_type)
                )
            prof_tasks = list()
            for _worker in prof_workers:
                prof_tasks.append(
                    _worker.profile.remote(comm_cfgs_list)
                )
            results = ray.get(prof_tasks)[0]
            # Record
            for _i, (_shape, _dtype, _comm_time, _replica_groups) in enumerate(results):
                global_comm_time_table[_key].append([_shape, str(_dtype), _comm_time, _replica_groups])
                print(f"[I] ({_i + 1}/{len(all_comm_sizes)}) OP type: {_op_type} | Data shape: {_shape} | " + 
                    f"Mesh shape: ({num_hosts}, {num_devices_per_host}) | Replica groups: {_replica_groups} " +
                    f"| Time cost (s): {_comm_time}")

            # for _i, _comm_size in enumerate(all_comm_sizes):
            #     _comm_cfgs = CollectiveCommConfigs(op_type=_op_type, replica_groups=_replica_groups, 
            #                             data_shape=_comm_size, data_type=data_type)
            #     prof_tasks = list()
            #     for _worker in prof_workers:
            #         prof_tasks.append(
            #             _worker.profile.remote(_comm_cfgs)
            #         )
            #     (_shape, _dtype, _comm_time, _replica_groups) = ray.get(prof_tasks)[0]
            #     global_comm_time_table[_key].append([_shape, str(_dtype), _comm_time, _replica_groups])
            #     print(f"[I] ({_i}/{len(all_comm_sizes)}) OP type: {_op_type} | Data shape: {_shape} | " + 
            #         f"Mesh shape: ({num_hosts}, {num_devices_per_host}) | Replica groups: {_replica_groups} " +
            #         f"| Time cost (s): {_comm_time}")
    
            # Store communication time data
            print("")
            print(f"[TMP] Writing profiled communication data to '{pth}'...")
            with open(pth, "wb") as f:
                pickle.dump(global_comm_time_table, f)
    
    print("[I] All communication operators have been profiled.")
    
    # Shutdown
    _shutdown_all(num_hosts, prof_workers, service_server, is_collective=True)


############################################
#           Dummy Test Functions           #
############################################

def test_init_nccl_communicator():
    """ Dummy test for initializing nccl communicator. """
    # Backend
    backend = xb.get_backend("gpu")
    local_devices = backend.local_devices()
    # Use all GPUs in one replica group
    replica_groups = [[_i for _i in range(len(local_devices))]]
    # Test
    comm_cfgs = CollectiveCommConfigs(op_type="", replica_groups=replica_groups, data_shape=(1024,),
                            data_type=xc.PrimitiveType.F32)
    init_nccl_communicator(comm_cfgs, backend, local_devices)


def test_inspect_profiled_comm_data():
    """ Dummy test for loading offline profiled communication data. """
    pth = "./jaxpr/comm_data/2_a100_2_n_4_d.pkl"
    with open(pth, "rb") as f:
        comm_time_table = pickle.load(f)
    
    print(comm_time_table.keys())
    for _key in comm_time_table.keys():
        print("")
        print(_key)
        print(comm_time_table[_key])

    # op_type = "all-reduce"
    # replica_groups = [[0, 1]]
    # key = str((op_type, replica_groups))
    # _comm_size = 1
    # while _comm_size <= MAX_COMM_SIZE:
    #     for _rec in comm_time_table[key]:
    #         if _rec[0] == (_comm_size,):
    #             print(f"Comm shape: {_rec[0]} | Comm time: {_rec[2]}")
    #             break
    #     _comm_size *= 2
    
    # print(MAX_COMM_SIZE)

    # key = str(("send", [[_i for _i in range(2)]]))
    # print(comm_time_table[key][-1])
    # print(comm_time_table[key][-2])
    # print(comm_time_table[key][-3])

    # key = str(("send", [[0, 1], [2, 3]]))

    # strs = key.split("',")
    # op_type = strs[0].split("'")[1]
    # tmp = strs[1].replace(" ", "").replace("[", "").replace("]])", "").split("],")
    # replica_groups = [[int(_c) for _c in _s.split(",")] for _s in tmp]

    # print(op_type)
    # print(replica_groups)

    # print(comm_time_table)


def main():
    """ Entrypoint. """
    # Environmental variables
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "True"
    os.environ["XLA_FLAGS"] = f"--xla_gpu_autotune_level={XLA_AUTO_TUNE_LEVEL}"
    os.environ["NCCL_USE_MULTISTREAM"] = NCCL_USE_MULTISTREAM

    if args.test_enum_replica_groups:
        # Test replica group enumeration
        replica_groups_list = enumerate_all_replica_groups(args.num_hosts, args.num_devices_per_host, args.only_best_locality)
        print("[I] All possible replica groups:")
        for groups in replica_groups_list:
            print(f" - {groups}")
        print(f"[I] Totally {len(replica_groups_list)} replica groups.")
        return

    if args.num_hosts == 1:
        if args.profile_collective:
            profile_comm_intra_host_collective()
        elif args.profile_p2p:
            profile_comm_intra_host_p2p()
    else:
        if args.profile_collective:
            profile_comm_inter_hosts_collective(only_best_locality=args.only_best_locality)
        elif args.profile_p2p:
            profile_comm_inter_hosts_p2p()

    # test_init_nccl_communicator()
    # test_inspect_profiled_comm_data()


if __name__ == "__main__":
    # Args 
    parser = argparse.ArgumentParser()
    # Configurations of communication configurations
    parser.add_argument("--devices_name", default="1_a40", type=str)
    parser.add_argument("--num_devices_per_host", default=2, type=int)
    parser.add_argument("--num_hosts", default=1, type=int)
    parser.add_argument("--force_shutdown", default=False, action='store_true', 
                            help="Whether to forcibly shutdown the workers after profiling.")
    parser.add_argument("--overwrite_data", default=False, action='store_true', 
                            help="Whether to overwrite profiled communication data.")
    parser.add_argument("--profile_collective", default=False, action='store_true', 
                            help="Whether to profile collective communication.")
    parser.add_argument("--profile_p2p", default=False, action='store_true', 
                            help="Whether to profile p2p communication.")
    parser.add_argument("--only_best_locality", default=False, action='store_true', 
                            help="Only profile the replica groups (assume 2-GPU node) with the best locality (e.g., " + 
                                 "[[0, 1], [2, 3]]), dropping those with suboptimal locality (e.g., [[0, 2], [1, 3]]). " + 
                                 "Cross-host group ([[0,1,2,3]]) is also dropped.")
    parser.add_argument("--test_enum_replica_groups", default=False, action='store_true', 
                            help="Enumerate all candidate replica groups (for check) without real execution.")
    args = parser.parse_args()

    xops = xc.ops

    # Global nccl communicator set that contains all replica groups corresponded to 
    # created communicators. 
    global_nccl_communicator_set = set()
    # Global communication time table
    global_comm_time_table = None

    main()

    # num_hosts = 4
    # num_devices_per_host = 1
    # replica_groups_list = enumerate_all_replica_groups(num_hosts, num_devices_per_host)

    # for _groups in replica_groups_list:
    #     print(_groups)

    # num_hosts = 4
    # num_devices_per_host = 4

    # arr = np.arange(num_hosts * num_devices_per_host).reshape((num_hosts, num_devices_per_host))

    # group_size = 8
    # num_hosts_crossed = 4
    # assert num_hosts_crossed <= num_hosts, f"Attempt to cross {num_hosts_crossed} hosts while only {num_hosts} hosts existed."

    # _intra_host_size = group_size // num_hosts_crossed
    # _num_tile_inter_hosts = num_hosts // num_hosts_crossed
    # _num_tile_intra_host = num_devices_per_host // _intra_host_size

    # _tmp = np.split(arr, _num_tile_inter_hosts, axis=0)

    # print(_tmp, type(_tmp))

    # _tmp_2 = list()
    # for _rec in _tmp:
    #     _tmp_2.extend(np.split(_rec, _num_tile_intra_host, axis=1))

    # for _rec in _tmp_2:
    #     _rec = list(_rec.reshape((-1,)))
    #     print(_rec)

    # pth = "./jaxpr/comm_data/_deprecated/1_v100_1_n_2_d.pkl"
    # with open(pth, "rb") as f:
    #     global_comm_time_table = pickle.load(f)

    # # _op_type = "send"
    # # replica_groups = [[0, 1]]
    # # print(str((_op_type, replica_groups)))
    
    # new_table = dict()
    # for _key in global_comm_time_table.keys():
    #     _new_key = f"('{_key}', {[[0, 1]]})"
    #     new_table[_new_key] = global_comm_time_table[_key]
    
    # pth = "./jaxpr/comm_data/1_v100_1_n_2_d.pkl"
    # with open(pth, "wb") as f:
    #     pickle.dump(new_table, f)

    # num_devices = 8

    # # Enumerate all possible replica groups with all gpus
    # replica_groups_list = list()
    # _group_size = 2
    # while _group_size <= num_devices:
    #     _group_num = num_devices // _group_size
    #     replica_groups_list.append(
    #         [[_i * _group_size + _j for _j in range(_group_size)] for _i in range(_group_num)]
    #     )
    #     _group_size *= 2
    
    # print(replica_groups_list)
