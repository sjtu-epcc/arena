#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to the runtime profiler on the input DNN model. """

import os
import argparse
import time
import threading
import traceback
from multiprocessing import Process
import pickle
import logging as logger
from typing import (
    Sequence, Tuple, Any, Optional, List, Dict)
import numpy as np
import ray

# Enable multi-thread CPU loading
num_cpus_for_jax = os.environ.get("NUM_CPUS_FOR_JAX", "16")
os.environ["XLA_FLAGS"] = f" --xla_force_host_platform_device_count={num_cpus_for_jax}"     # Overwrite
# num_threads_for_load = os.environ.get("NUM_THREADS_FOR_LOAD", "4")
# os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
#                               f" --xla_force_host_platform_device_count=16 --xla_cpu_multi_thread_eigen=true " + 
#                               f" intra_op_parallelism_threads={num_threads_for_load} inter_op_parallelism_threads=1")

# TODO(chunyu): A good idea: since we only focus on getting all operators of a single microbatch, we can overwrite
# the global batch size to global-bs/num_microbatches for profiling, and restore it for e2e latency modeling.


import jax
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.interpreters import pxla
# from jax._src.lib.xla_extension import (
#     HloModule, Shape, LoadedExecutable)

import alpa
from alpa.global_env import global_config
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, XlaShardedPipelineComputation)
from alpa.wrapped_hlo import WrappedHlo, HloStatus
from alpa.pipeline_parallel.compile_executable import (
    generate_sharded_xla_computations, generate_sharded_xla_computations_arguments)
from alpa.pipeline_parallel.stage_profiling import (CompileWorkerPool, BaseWorkerPoolWrapper, run_auto_sharding_pass)
from alpa.pipeline_parallel.computation import get_donatable_intermediate, generate_computations_from_modules
from alpa.shard_parallel.auto_sharding import AutoShardingOption, run_spmd_partitioner_pass
from alpa.util import OrderedSet
from ray.util import ActorPool

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell.cell import (
    Cell, gen_submesh_physical_shapes)
from pipeline.planner import PipelinePlanner
from jaxpr.preprocess import (
    load_trainer_configs, prepare_flax_model)
from jaxpr.transform_jaxpr import (
    prepare_model_and_transform_jaxpr, generate_virtual_physical_mesh)
from jaxpr.transform_hlo import (
    transform_jaxpr_stages_to_hlo_modules, compile_one_sharded_hlo_module, 
    shard_one_hlo_module)
from jaxpr.communication import (
    estimate_nonprof_comm, gen_replica_id_to_device_mapping
)
from jaxpr.hlo_ops import ParamOperator

if os.environ.get("ENABLE_CRIUS_PROFILER", "false") == "true":
    # Import functions of crius profiler
    print("")
    print("[I] Loading Crius kernel-level profiler...")
    from jaxpr.hlo_profiler import (
        reconstr_hlo_entry_and_stat_comm, constr_one_xla_op, 
        profile_one_compiled_executable, compile_and_profile_one_xla_op,
        _estimate_comm_op_time)

from jaxpr.utils import (
    XLA_AUTO_TUNE_LEVEL, NCCL_USE_MULTISTREAM, AVG_GEMM_KERNEL_TIME, 
    BASE_GPU_TYPE, MAX_DEVICE_NUM_PER_HOST, CAND_GPU_TYPES, GB,
    MAX_TRAIN_TIMEOUT, XLA_PRIMITIVE_TYPE_NUM_BYTES, ParallelMethod, 
    HardwareConfigs, ProfileConfigs, Computation, IterTimeCollection, InputConfigs, 
    CellConfigs, load_device_info_table, unique_append, remove_all, is_power_of, 
    gen_hashkey_with_model_configs, load_tuning_database, store_tuning_database)
from jaxpr.testing import (
    get_dummy_input_cfgs, get_dummy_trainer_cfgs, 
    test_create_virtual_placement_group)

# Current absolute path of this script
CUR_PATH = os.path.dirname(os.path.abspath(__file__))


def init_backend():
    """ Initializing Ray Cluster & Alpa backend. """
    # Connect to or construct a ray cluster
    if not ray.is_initialized():
        try:
            # In this case, profiling tools such as nsys cannot track cuda api calls by 
            # `nsys profile python script.py`, since the running ray instance is not a 
            # child process of this python script.
            ray.init(address=args.ray_address)
            # Change to this can modify ray.util.get_node_ip_adress() from default ip
            # to the specified ip (for infiniband).
            # ray.init(address="auto", _node_ip_address="192.168.1.66")
        except:
            print("[I] No local Ray cluster is found, constructing a new one.")
            # ray.init(address="local")
            ray.init()
    # Get envion
    num_hosts = int(os.environ.get("CRIUS_NUM_HOSTS"))
    num_devices_per_host = int(os.environ.get("CRIUS_NUM_DEVICES_PER_HOST"))
    # Init alpa
    alpa.init(cluster="ray", num_devices_per_node=num_devices_per_host, num_nodes=num_hosts)

    print("[I] Ray Cluster & Alpa backend initialization is completed.")
    print("[I] Device Info:")
    print(f"    - Devices num per node: {num_devices_per_host}")
    print(f"    - Nodes num: {num_hosts}")


def init_ray_cluster():
    """ Initialize Ray cluster for parallel compilation for single-device profiling. """
    # Connect to or construct a ray cluster
    if not ray.is_initialized():
        try:
            ray.init(address=args.ray_address)
        except:
            print("[I] No local Ray cluster is found, constructing a new one.")
            # ray.init(address="local")
            ray.init()


def _init_donation_mapping(num_meshes: int, schedule: Any, jax_all_stages: Sequence[JaxPipelineComputation], 
                           global_invars: Any, accumulator_mapping: Any):
    """ 
    Decide the mapping from the co-located stages to mesh and initialize donation. 
    Modified from: alpa/pipeline_parallel/compile_executable/shard_each_stage()
    """
    # Initialize donation mapping
    stage_dict = [[] for _ in range(num_meshes)]
    stage_id_dict = [[] for _ in range(num_meshes)]
    dummy_stage_id_dict = [[] for _ in range(num_meshes)]
    donatable_dict = [[] for _ in range(num_meshes)]
    mesh_stage_mapping = schedule.mesh_stage_mapping
    
    donatable_list = get_donatable_intermediate(
        jax_all_stages, mesh_stage_mapping, 
        OrderedSet(global_invars).union(accumulator_mapping.keys()))
    # Map stages to meshes, gathering co-located stages (forward, backward, apply_grad)
    for i, stage in enumerate(jax_all_stages):
        mesh_indices = list(schedule.stage_placement(i))
        assert len(mesh_indices) == 1
        mesh_idx = mesh_indices[0]
        if len(stage.outvars) == 0:
            # This is a dummy stage, we don't need to shard it
            dummy_stage_id_dict[mesh_idx].append(i)
            continue
        stage_id_dict[mesh_idx].append(i)
        stage_dict[mesh_idx].append(stage)
        donatable_dict[mesh_idx].append(donatable_list[i])
    
    return stage_dict, stage_id_dict, dummy_stage_id_dict, donatable_dict, mesh_stage_mapping


def _parse_parallel_degrees(profile_cfgs: ProfileConfigs):
    """ 
    Parse the user-specified parallelism degrees, support symmetric 
    mesh slice and need specified logical shapes for asymmetric cases. 
    """

    assert (profile_cfgs.parallel_degrees != "none" or 
            profile_cfgs.force_logical_shapes != "none" or 
            profile_cfgs.force_plan_shape_hashkey != "none"), \
        "When cell profile is disabled, user need to specify parallelism."

    num_hosts = profile_cfgs.num_hosts
    num_devices_per_host = profile_cfgs.num_devices_per_host
    num_devices = num_hosts * num_devices_per_host

    # Logical submesh shapes (i.e., parallelism) 
    if profile_cfgs.force_logical_shapes != "none":
        # Forced parallelism (mostly for asymmetric case)
        _stage_strs = profile_cfgs.force_logical_shapes.split(",")
        submesh_logical_shapes = [
            (int(_c.split("_")[0]), int(_c.split("_")[1])) 
                for _c in _stage_strs
        ]
        pp_degree = len(submesh_logical_shapes)

    elif profile_cfgs.force_plan_shape_hashkey != "none":
        # Forced plan_shape_hashkey
        (plan_hashkey, shape_hashkeys) = profile_cfgs.force_plan_shape_hashkey.split("::")
        # Submesh logical/physical shapes
        submesh_logical_shapes = [
            Cell.gen_hashkey_with_stage_shape(shape_hashkey=_s, decode=True).stage_shape
                for _s in shape_hashkeys.split("__")
        ]
        pp_degree = len(submesh_logical_shapes)

    else:
        # Construct symmetric parallelism based on degrees
        if isinstance(profile_cfgs.parallel_degrees, str):
            para_degrees = tuple([
                int(_c) for _c in profile_cfgs.parallel_degrees.split(",")
            ])
        else:
            assert isinstance(profile_cfgs.parallel_degrees, tuple), \
                f"Parallel degrees should be either tuple or str object."
            para_degrees = profile_cfgs.parallel_degrees
        
        (pp_degree, dp_degree, mp_degree) = para_degrees
        # # Environmental variables
        # os.environ["CRIUS_GLOBAL_PARA_DEGREES"] = f"{pp_degree}_{dp_degree}_{mp_degree}"
        assert is_power_of(2, pp_degree) and np.prod(para_degrees) == num_devices, \
            f"User-specified parallel degrees ({para_degrees}) is with the wrong format."
        submesh_logical_shapes=[(dp_degree, mp_degree) for _ in range(pp_degree)]
    
    # Physical submesh shapes (i.e., gpu allocation)
    gpu_sharding = [np.prod(_submesh) for _submesh in submesh_logical_shapes]
    assert np.sum(gpu_sharding) == num_devices, \
        f"Mismatch between GPU sharding ({gpu_sharding}) and total #GPUs ({num_devices})"
    submesh_physical_shapes = gen_submesh_physical_shapes(gpu_sharding, num_hosts, 
                                                          num_devices_per_host)
    
    return (submesh_physical_shapes, submesh_logical_shapes, pp_degree)


def _get_cross_stages_comm_vars(src: XlaShardedPipelineComputation, dst: XlaShardedPipelineComputation):
    """ Get the communication variables between two stages. """
    comm_vars = []
    for _i, _var in enumerate(src.outvars):
        if _var in dst.invars:
            comm_vars.append((_var.aval.shape, _var.aval.dtype))
    return comm_vars


def _get_hlo_output_sharding_specs(hlo_sharding: xc.OpSharding):
    """ 
    Parse the sharding specs of the target HLO sharding in the tuple format. 
    Refer to: `alpa/shard_parallel/auto_sharding/hlo_sharding_to_sharding_spec()`
    """
    sharding_specs = list()     # [(sharding_type, num_dims, dim_sharding), ...]
    # Protobuf for describing data structure
    proto = hlo_sharding.to_proto()
    sharding_type, tuple_shardings = proto.type, proto.tuple_shardings
    assert sharding_type == xc.OpSharding.Type.TUPLE, \
        "Only support tuple-style output HLO shardings."

    for _proto in tuple_shardings:
        sharding_type, tile_assign_dims, tile_assign_devs = (
            _proto.type, _proto.tile_assignment_dimensions,
            _proto.tile_assignment_devices)

        if sharding_type == xc.OpSharding.Type.OTHER:
            # Sharded across devices
            sharding_specs.append(
                (sharding_type, len(tile_assign_dims), tuple(tile_assign_dims))
            )
        elif sharding_type == xc.OpSharding.Type.REPLICATED:
            # Replicated across devices
            sharding_specs.append(
                (sharding_type, len(tile_assign_dims), tuple([1 for _ in range(len(tile_assign_dims))]))
            )
        else:
            raise NotImplementedError()
            
            # for _i, _dim in enumerate(tile_assign_dims):
            #     # For each shape dim
            #     if _dim == 1:
            #         sharding.append(pxla.NoSharding())
            #     else:
            #         sharding.append(pxla.Chunked(_dim))
        
        # # Only consider abstract dim sharding and ignore mesh mapping
        # sharding_specs.append(pxla.ShardingSpec(sharding, None))
    
    return sharding_specs


def _get_hlo_sharded_outvars(hlo: xe.HloModule, sharded: bool = True):
    """ 
    Get the sharded output tensors of the HLO module on the single device.
    """
    outvars = list()
    # Shapes of output tensors
    result_shape = hlo.program_shape().result_shape()
    
    if not sharded:
        # Hlo not sharded
        for _i, _shape in enumerate(result_shape.tuple_shapes()):
            # Output tensor shape 
            outvars.append((_shape.dimensions(), str(_shape.element_type())))
    else:
        # Hlo sharded
        # Sharding specs of output tensors
        sharding_specs = _get_hlo_output_sharding_specs(hlo.spmd_output_sharding())
        assert len(result_shape.tuple_shapes()) == len(sharding_specs), \
            f"Mismatched number of output tensors ({len(result_shape.tuple_shapes())}) " + \
            f"and sharding specs ({len(sharding_specs)}). "
        for _i, _shape in enumerate(result_shape.tuple_shapes()):
            # Output tensor shape
            (sharding_type, num_dims, dim_sharding) = sharding_specs[_i]
            if sharding_type == xc.OpSharding.Type.OTHER:
                # Sharded, _shape.dimensions() has already been sharded by dim_sharding as the
                # initial shape is _shape.dimensions() * dim_sharding.

                # TODO(chunyu): For inter-hosts communication, we should use _shape.dimensions() * dim_sharding
                #               as the total inter-stages communication amount for sharded output tensors.
                #               For intra-hosts communication, we use _shape.dimensions() if the p2p nvlink 
                #               among each GPU pair is enabled.
                
                _sharded_shape = tuple([
                    _shape.dimensions()[_j] * dim_sharding[_j] 
                        for _j in range(num_dims)
                ])

                # _sharded_shape = _shape.dimensions()


            elif sharding_type == xc.OpSharding.Type.REPLICATED:
                # Replicated
                _sharded_shape = _shape.dimensions()

                # TODO(chunyu): For inter-hosts communication, we should use _shape.dimensions() * num_gpus
                #               as the total inter-stages communication amount for sharded output tensors.
                #               For intra-hosts communication, we use _shape.dimensions() if the p2p nvlink 
                #               among each GPU pair is enabled.

                # TODO(chunyu): This tensor should be the output of the allreduce for gradient sync 
                #               (during backward), which should only be transferred to the next 
                #               backward stage for the last microbatch.

            else:
                raise NotImplementedError()

            outvars.append((_sharded_shape, str(_shape.element_type())))
        
            print(sharding_type, _shape.dimensions(), _sharded_shape, sharding_specs[_i])

    return outvars


def _module_to_stage_mapping(num_stages: int = 1):
    """ 
    Generate the index mapping from modules to each stages. The mapping should be:
    [fp_1, fp_2, ..., fp_n, bp_n, bp_{n-1}, ..., bp_1, apply_grad_1, apply_grad_2, ..., apply_grad_n]
    """
    forward_module_idxs = list(range(num_stages))
    backward_module_idxs = [num_stages + _i for _i in forward_module_idxs]
    apply_grad_module_idxs = [2 * num_stages + _i for _i in forward_module_idxs]

    module_idx_to_stage_idx_mapping = {}
    for _i, _idx in enumerate(forward_module_idxs):
        module_idx_to_stage_idx_mapping[_idx] = _i
    for _i, _idx in enumerate(backward_module_idxs):
        module_idx_to_stage_idx_mapping[_idx] = len(backward_module_idxs) - _i - 1
    for _i, _idx in enumerate(apply_grad_module_idxs):
        module_idx_to_stage_idx_mapping[_idx] = _i

    return forward_module_idxs, backward_module_idxs, apply_grad_module_idxs, module_idx_to_stage_idx_mapping


def _shard_stages_and_gen_hlos(
    jax_all_stages: Sequence[JaxPipelineComputation],
    submesh_logical_shapes: Sequence[Sequence[int]],
    submesh_physical_shapes: Sequence[Sequence[int]],
    backend: Any, 
    schedule: Any, 
    global_invars: Any,
    accumulator_mapping: Any, 
    donate_invars_dict,
    acc_grad_outvars: Any, 
    num_micro_batches: int,
    sliced_virtual_meshes: Any, 
    parallel_idx: int,
):
    """ Shard all Jax stages and generate HLO texts. """
    
    print("")
    print(f"[I] (Parallel plan idx: {parallel_idx}) Sharding stages...")
    num_forward_stages = len(submesh_logical_shapes)
    
    # Initialize donation mapping
    (stage_dict, stage_id_dict, dummy_stage_id_dict, 
     donatable_dict, mesh_stage_mapping) = _init_donation_mapping(num_forward_stages, schedule, jax_all_stages, 
                                                                  global_invars, accumulator_mapping)
    # Enable ILP in intra-stage sharding
    as_option = AutoShardingOption(enable_auto_sharding=True, force_batch_dim_to_mesh_dim=0, prefer_reduce_scatter=True,
                                   allow_mixed_mesh_shape=True)
    
    # @ray.remote(num_cpus=1)
    # class __SharedMemoryActor:
    #     """ Ray actor for shared memory. """
    #     def __init__(self):
    #         self.jax_all_stages = None
        
    #     def set_jax_all_stages(self, jax_all_stages_):
    #         self.jax_all_stages = jax_all_stages_

    #     def get_jax_all_stages(self):
    #         return self.jax_all_stages

    # shared_mem_actor = __SharedMemoryActor.remote()
    # ray.get(shared_mem_actor.set_jax_all_stages.remote(jax_all_stages))

    # exit(0)

    # @ray.remote(num_cpus=1)
    # def __ray_actor_stage_sharding_func(mesh_index_, stage_dict_ref):
    #     """ Ray actor function for parallel XLA stage sharding. """
        
    #     stage_dict_ = ray.get(stage_dict_ref)
    #     print(stage_dict_.__class__)

        # stage_donate_invars_ = [donate_invars_dict[_stage_idx] for _stage_idx in stage_id_dict[mesh_index_]]
        # ray_sharded_xla_stages , ray_flops = generate_sharded_xla_computations(
        #     name=f"mesh_{mesh_index_}", 
        #     jaxpr_computations=stage_dict[mesh_index_],
        #     computation_donate_invars=stage_donate_invars_,
        #     donatable_lists=donatable_dict[mesh_index_],
        #     acc_grad_outvars=acc_grad_outvars,
        #     num_micro_batches=num_micro_batches,
        #     logical_mesh=sliced_virtual_meshes[mesh_index_].get_logical_mesh(submesh_logical_shapes[mesh_index_]),
        #     autosharding_option=as_option,
        #     input_sharding_dict=None, 
        #     output_sharding_dict=None, 
        #     stage_input_sharding=None,
        # )
        # return ray_sharded_xla_stages, ray_flops

        # return 0

    # exported = export.export(jax.jit(jax_all_stages[0]))(jax.ShapeDtypeStruct((), jax.numpy.float32))
    # serialized_jaxpr = exported.serialize()
    
    # def __stage_sharding_func(mesh_index_):
    #     """ A process function for parallel XLA stage sharding. """
        
    #     print(stage_dict.__class__)
    #     print("In a multiprocess function, global stage_dict is available!")

    #     return mesh_index_

    class StageShardingWorker:
        """ A Ray actor for stage sharding. """

        @staticmethod
        def run_auto_sharding_pass(stage_id, hlo, other_kwargs):
            """Run auto-sharding pass on a WrappedHlo."""
            assert other_kwargs["return_mode"] == "stages"
            # pylint: disable=unbalanced-tuple-unpacking
            hlo_stage_names, hlo_stages, stage_plan = run_auto_sharding_pass(
                hlo, **other_kwargs)
            return stage_id, (hlo_stage_names, hlo_stages, stage_plan)        

    class StageShardingWorkerPool(BaseWorkerPoolWrapper):
        """ A pool of StageShardingWorker actors. """
        def __init__(self, num_cpus):
            super().__init__()
            worker_cls = ray.remote(num_cpus=1)(StageShardingWorker)
            self.actors = [worker_cls.remote() for _ in range(num_cpus)]
            self.pool = ActorPool(self.actors)   

    # Shard stages
    use_ray_for_parallel_compile = os.environ.get("USE_RAY_FOR_PARALLEL_COMPILE", "false") == "true"
    if use_ray_for_parallel_compile:
        compile_workers = StageShardingWorkerPool(num_forward_stages)
        compile_fn = lambda w, v: w.run_auto_sharding_pass.remote(*v)
        compile_intermediate = [None] * num_forward_stages

    sharded_xla_stages = [None] * len(jax_all_stages)
    _time_marker = time.time()

    # Repeated stages in the model to reduce compiling/profiling times
    repeated_stages = os.environ.get("REPEATED_STAGES_IN_MODEL", "none")
    if repeated_stages != "none":
        repeated_stages = [int(_layer) for _layer in repeated_stages.split(",")]
        assert 0 not in repeated_stages, "Stage 0 should not be in repeated stages."
        print(f"\n\n[I] Repeated stages in the model: {repeated_stages}")
    else:
        repeated_stages = []
    
    if not use_ray_for_parallel_compile:
        # Sequential compile each stage
        for mesh_idx in range(num_forward_stages):
            stage_donate_invars = [donate_invars_dict[stage_idx] for stage_idx in stage_id_dict[mesh_idx]]
            # Auto-sharding pass to annotate sharding specs
            _sharded_xla_stages, flops = generate_sharded_xla_computations(name=f"mesh_{mesh_idx}", 
                                                                        jax_computations=stage_dict[mesh_idx], 
                                                                        computation_donate_invars=stage_donate_invars, 
                                                                        donatable_lists=donatable_dict[mesh_idx], 
                                                                        acc_grad_outvars=acc_grad_outvars, 
                                                                        num_micro_batches=num_micro_batches,
                                                                        logical_mesh=sliced_virtual_meshes[mesh_idx] \
                                                                                        .get_logical_mesh(submesh_logical_shapes[mesh_idx]), 
                                                                        autosharding_option=as_option, 
                                                                        input_sharding_dict=None, output_sharding_dict=None,
                                                                        stage_input_sharding=None)
            for i, _stage in zip(stage_id_dict[mesh_idx], _sharded_xla_stages):
                sharded_xla_stages[i] = _stage
        print(f"[I] Sharding stages takes {time.time() - _time_marker} s.")
    
    else:

        # def __gen_stage_sharding_args(mesh_index_):
        #     """ A thread function for parallel geting XLA stage sharding arguments. """
        #     stage_donate_invars = [donate_invars_dict[_stage_idx] for _stage_idx in stage_id_dict[mesh_index_]]
        #     # Auto-sharding pass to annotate sharding specs
        #     _sharded_xla_stages, flops = generate_sharded_xla_computations(name=f"mesh_{mesh_index_}", 
        #                                                                 jax_computations=stage_dict[mesh_index_], 
        #                                                                 computation_donate_invars=stage_donate_invars, 
        #                                                                 donatable_lists=donatable_dict[mesh_index_], 
        #                                                                 acc_grad_outvars=acc_grad_outvars, 
        #                                                                 num_micro_batches=num_micro_batches,
        #                                                                 logical_mesh=sliced_virtual_meshes[mesh_index_] \
        #                                                                                 .get_logical_mesh(submesh_logical_shapes[mesh_index_]), 
        #                                                                 autosharding_option=as_option, 
        #                                                                 input_sharding_dict=None, output_sharding_dict=None,
        #                                                                 stage_input_sharding=None)
        #     for i, _stage in zip(stage_id_dict[mesh_index_], _sharded_xla_stages):
        #         sharded_xla_stages[i] = _stage         

        # threads = [threading.Thread(target=__gen_stage_sharding_args, args=(_i, )) for _i in range(num_forward_stages)]
        # for t in threads:
        #     t.start()
        # for t in threads:
        #     t.join()

        for mesh_idx in range(num_forward_stages):
            if mesh_idx in repeated_stages and repeated_stages.index(mesh_idx) != 0:
                # Skip the repeated stages
                print(f"[I] (_shard_stages_and_gen_hlos(), compile_workers) Skip sharding stage {mesh_idx} as it is a repeated stage.")
                continue
            
            stage_donate_invars = [donate_invars_dict[stage_idx] for stage_idx in stage_id_dict[mesh_idx]]
            logical_mesh = sliced_virtual_meshes[mesh_idx].get_logical_mesh(submesh_logical_shapes[mesh_idx])

            hlo, flops = (generate_sharded_xla_computations_arguments(
                f"mesh_{mesh_idx}", stage_dict[mesh_idx],
                stage_donate_invars, None, None, None))
            other_kwargs = {
                "logical_mesh": logical_mesh,
                "return_mode": "stages",
                "as_option": as_option,
                "num_micro_batches": num_micro_batches,
            }
            compile_workers.submit(compile_fn, (mesh_idx, hlo, other_kwargs))
            compile_intermediate[mesh_idx] = (stage_dict[mesh_idx], stage_donate_invars)

        # print(f"[I] Get sharding arguments takes {time.time() - _t} s.")

        num_real_forward_stages = num_forward_stages - len(repeated_stages) + 1
        for _ in range(num_real_forward_stages):
            mesh_idx, (computation_names, computation_hlos,
                       stage_plan) = compile_workers.get_next_unordered()
            jax_computations, computation_donate_invars = compile_intermediate[mesh_idx]
            _sharded_xla_stages = generate_computations_from_modules(
                jax_computations, computation_names, computation_hlos, computation_donate_invars, 
                donatable_dict[mesh_idx], acc_grad_outvars, stage_plan)
            for i, xla_stage in zip(stage_id_dict[mesh_idx], _sharded_xla_stages):
                sharded_xla_stages[i] = xla_stage
        
        # Put all objects into shared memory
        # stage_dict_ref = ray.put(stage_dict)
        
        # Use ray actors for parallel compilation
        # ray_funcs = [
        #     __ray_actor_stage_sharding_func.options(name=f"xla_stage_sharding_actor_{_mesh_idx}").remote(
        #         _mesh_idx, stage_dict_ref, 
        #     ) for _mesh_idx in range(num_forward_stages)
        # ]
        # results = ray.get(ray_funcs)

        # from jax import export
        # from multiprocessing import Pool

        # with Pool(num_forward_stages) as p:
        #     results = p.map(__stage_sharding_func, range(num_forward_stages))
        
        print(f"[I] Sharding stages takes {time.time() - _time_marker} s.")

        compile_workers.shutdown()

    # Module -> stage mapping
    _, _, _, module_idx_to_stage_idx_mapping = _module_to_stage_mapping(num_forward_stages)
    
    # SPMD partition and compile
    print(f"[I] (Parallel plan idx: {parallel_idx}) SPMD partitioning and " + 
          f"compiling (kernel fusing) HLO stages...")
    assert len(sharded_xla_stages) % 3 == 0, \
        f"The number of sharded XLA stages should be divisible by 3 (forward + backward " + \
        f"+ apply_grad), got {len(sharded_xla_stages)}."
    
    class SPMDPartitionWorker:
        """ A Ray actor for stage SPMD partitioning. 
        
        In single-GPU profiling, Ray actors multiplex a single GPU device.
        """

        @staticmethod
        def run_spmd_partition_pass(
            sharded_stage_id, sharded_stage_hlo: WrappedHlo, 
            output_acc_grad_indices: Sequence[int],
        ):
            """Run SPMD pass on a sharded stage (seperated forward/backward/apply_grad)."""
            assert not sharded_stage_hlo.is_spmd_partitioned()
            stage_idx = module_idx_to_stage_idx_mapping[sharded_stage_id]
            num_devices_cur_mesh = np.prod(submesh_physical_shapes[stage_idx])
            # Only-sharded hlo modules
            hlo_module: WrappedHlo = run_spmd_partitioner_pass(
                sharded_stage_hlo, num_devices_cur_mesh, 
                rewrite_for_grad_acc=(len(output_acc_grad_indices) > 0),
                rewrite_grad_acc_indices=output_acc_grad_indices,
            )
            # Optimized hlo modules
            local_devices_ = [xb.local_devices(backend='gpu')[0]]     # Only one xla device
            backend_ = xb.get_device_backend(local_devices_[0])
            compiled = compile_one_sharded_hlo_module(hlo_module, num_devices_cur_mesh, 
                                                    backend_, bypass_device=True)
            # Wrap into serializable object
            wrapped_opt_hlo_module = WrappedHlo(compiled.hlo_modules()[0], HloStatus.FULLY_OPTIMIZED)
            return sharded_stage_id, (hlo_module, wrapped_opt_hlo_module)

    class SPMDPartitionWorkerPool(BaseWorkerPoolWrapper):
        """ A pool of SPMDPartitionWorker actors. """
        def __init__(self, num_cpus):
            super().__init__()
            print(f"\n\n ===============> Number of SPMD partition workers: {num_cpus}")
            worker_cls = ray.remote(num_cpus=1, num_gpus=(1 / num_cpus))(SPMDPartitionWorker)
            self.actors = [worker_cls.remote() for _ in range(num_cpus)]
            self.pool = ActorPool(self.actors)   

    if not use_ray_for_parallel_compile:
        hlo_modules, optimized_modules = [], []
        cross_stages_comm_vars_table = {}   # stage idx (output) -> comm vars list ([(shape, dtype), ...])
        _time_marker = time.time()
        for _i, _stage in enumerate(sharded_xla_stages):
            stage_idx = module_idx_to_stage_idx_mapping[_i]
            num_devices_cur_mesh = np.prod(submesh_physical_shapes[stage_idx])
            # Process hlo modules
            hlo_module = _stage.get_spmd_partitioned()
            # Only-sharded hlo modules
            hlo_modules.append(hlo_module.get_module())
            # Optimized hlo modules
            compiled = compile_one_sharded_hlo_module(hlo_module, num_devices_cur_mesh, 
                                                    backend, bypass_device=True)
            opt_hlo = compiled.hlo_modules()[0]
            optimized_modules.append(opt_hlo)

            # print("")
            # print(f"Module idx: {_i}")

            # Add additional analysis to get inter-stages communication tensors of compiled 
            # (graph-optimized) modules. Use the output tensors of stage i as communication 
            # tensors, which are one-to-one mapped with input tensors of stage i+1. We 
            # should also identify forward/backward/apply_grad stages here by analyzing 
            # different phases seperately.
            # Note that the output tensors of a hlo can be sharded onto multiple devices
            # as DistributedArray, while each device only sends its own partition without 
            # considering partitions on other devices. Since we model inter-stages communication
            # with p2p send/recv, the stage outvars should be captured with sharded shapes 
            # on the single device.
            if _i < 2 * num_forward_stages - 1 and _i != num_forward_stages - 1:
                # Apply_grad stage not has inter-stages communication, 
                # the same for the last forward stage and the last backward stage.
                # Record the communicated variables from stage i to stage i + 1 with
                # the format of (shape, dtype).
                cross_stages_comm_vars_table[_i] = _get_cross_stages_comm_vars(_stage, 
                                                                            sharded_xla_stages[_i + 1])

        print(f"[I] SPMD partitioning and pre-compiling HLO modules concurrently takes {time.time() - _time_marker} s.")

    else:
        _time_marker = time.time()
        hlo_modules = [None] * len(sharded_xla_stages)
        optimized_modules = [None] * len(sharded_xla_stages)
        spmd_partition_workers = SPMDPartitionWorkerPool(len(sharded_xla_stages))
        spmd_partition_fn = lambda w, v: w.run_spmd_partition_pass.remote(*v)
        for i, stage in enumerate(sharded_xla_stages):
            if stage is None:
                # Skip the repeated stages
                print(f"[I] (_shard_stages_and_gen_hlos(), spmd_partition_workers) Skip SPMD partitioning stage {i} as it is a repeated stage.")
                continue
            
            # Submit the SPMD partitioning task
            hlo_: WrappedHlo = stage.hlo
            output_acc_grad_indices_: Sequence[int] = stage.output_acc_grad_indices
            spmd_partition_workers.submit(spmd_partition_fn, (i, hlo_, output_acc_grad_indices_))
        
        num_real_sharded_xla_stages = len(sharded_xla_stages) - len(repeated_stages) * 3 + 3
        for _ in range(num_real_sharded_xla_stages):
            sharded_stage_id_, (hlo_module_, opt_hlo_module_) = spmd_partition_workers.get_next_unordered()
            hlo_modules[sharded_stage_id_] = hlo_module_.get_module()
            optimized_modules[sharded_stage_id_] = opt_hlo_module_.get_module()

        print(f"[I] SPMD partitioning and pre-compiling HLO modules concurrently takes {time.time() - _time_marker} s.")

        cross_stages_comm_vars_table = {}
        for i, stage in enumerate(sharded_xla_stages):
            if i < 2 * num_forward_stages - 1 and i != num_forward_stages - 1:
                if stage is None or sharded_xla_stages[i + 1] is None:
                    # Skip the repeated stages
                    # FIXME(chunyu): We ignore the inter-stages communication of repeated stages.
                    cross_stages_comm_vars_table[i] = []
                    continue

                cross_stages_comm_vars_table[i] = _get_cross_stages_comm_vars(stage, sharded_xla_stages[i + 1])            

    return hlo_modules, optimized_modules, cross_stages_comm_vars_table


def load_model_and_generate_sharded_hlo(
    input_cfgs: InputConfigs, 
    cell: Cell,
    pipeline_planner: PipelinePlanner,
    enable_cell_profile: bool = False,
    cell_prof_strategy: str = "auto",
    # enable_auto_pipeline: bool = False,
) -> None:
    """ 
    Given a cell, load model and generate sharded HLO text with specified 
    or enumerated parallel plans on the given hardware configuration. 

    Args: 
     - `input_cfgs`: Described in docstrings of `estimate_one_cell()`.
     - `cell`: The cell to be pipeline-partitioned and profiled.
     - `pipeline_planner`: Pipeline planner to perform stage partition and GPU allocation 
                           the the cell.
     - `enable_cell_profile`: Enable cell-style profiling that automatically (1) allocate
                              per-layer GPU fraction based on layer FLOPs; (2) cluster layers
                              into stages (#stages is specified) based on minimal inter-stages
                              communication; (3) Shard per-stage GPUs based on layer GPU 
                              fraction; (4) Enumerate and profile several parallel plans within 
                              cell based on profiling strategy. 
     - `cell_prof_strategy`: The strategy of cell profiling. Options: ["minimal", "uniform", "auto"]. 
                             Detailed descriptions are given in `../cell/enum_parallel_plans()`.
    """

    print("")
    print("[I] Loading model and generating sharded HLO module...")
    
    # Job
    job_id = input_cfgs.job_id if input_cfgs.job_id is not None else "default"

    # Backend
    # Once the XLA Python client is started, the GPU memory is pre-allocated until the 
    # program is ended. The profiler only exploits one xla device to profile operators.
    _time_marker = time.time()
    local_devices = [xb.local_devices()[0]]     # Only one xla device
    backend = xb.get_device_backend(local_devices[0])
    virtual_mesh = generate_virtual_physical_mesh(cell)
    print(f"[I] Initializing XLA devices takes {time.time() - _time_marker} s.")

    # ------------------ Step 1. Prepare model and determine pipeline partition ------------------
    
    # First, prepare model and transform to jaxpr stages, deciding submesh physical 
    # shapes (i.e., gpu allocation of each stage) if auto pipeline is enabled.
    num_micro_batches = input_cfgs.trainer_cfgs["num_micro_batches"]
    _time_marker = time.time()
    (jax_all_stages, sliced_virtual_meshes, 
     schedule, _, accumulator_mapping, 
     global_invars, num_micro_batches, 
     donate_invars_dict, acc_grad_outvars, 
     submesh_physical_shapes) = prepare_model_and_transform_jaxpr(
         input_cfgs, 
         cell, 
         pipeline_planner,
         use_remat=True, 
         virtual_mesh=virtual_mesh,
     )
    print(f"[I] Loading model and transform to Jaxpr stages " + 
          f"takes {time.time() - _time_marker} s.")

    # exit(0)
    
    assert cell.is_pipeline_partitioned(), \
        f"Cell is not pipeline-partitioned."
    print(f"[I] The GPU sharding of pipeline " + 
          f"stages is: {cell.pipeline_plan.gpu_sharding}")

    # ------------------ Step 1.5. Generate to-tune plans and update tuning database ------------------
    
    if os.environ.get("DISABLE_PLAN_SET", "false") == "false":
        # Generate pareto plan set for further job tuning, where each elite plan is represented as:
        #   f"{plan_hashkey}::{shape_hashkey_stage_1}__{shape_hashkey_stage_2}"
        plan_set = cell.gen_pareto_plan_set_for_tuning(num_micro_batches)
        
        # Load the global tuning database and update
        tuning_database = load_tuning_database()
        model_cfgs_hashkey = gen_hashkey_with_model_configs(
            model_name=input_cfgs.trainer_cfgs["model_name"],
            param_num=input_cfgs.trainer_cfgs["param_num"],
            batch_size=input_cfgs.trainer_cfgs["batch_size"],
            num_micro_batches=input_cfgs.trainer_cfgs["num_micro_batches"],
            gpu_type=cell.gpu_type,
            num_hosts=cell.num_hosts,
            num_devices_per_host=cell.num_devices_per_host,
            num_stages=cell.num_stages,
        )
        tuning_database["plan_set"][model_cfgs_hashkey] = plan_set

        print(tuning_database)
        
        # Store the global tuning database
        store_tuning_database(tuning_database)

    # ------------------ Step 2. Enumerate and determine cell's parallel plans ------------------

    # Then, we need to decide submesh logical shapes (i.e., how each stage-submesh 
    # pair is data/tensor parallelized).
    if not enable_cell_profile:
        # Use user-specified parallelism to profile one parallelism
        assert input_cfgs.parallel_method.submesh_logical_shapes is not None, \
            "The parallelism must be specified when cell profile is disabled."
        cell.preset_parallel_plans([input_cfgs.parallel_method.submesh_logical_shapes])
    else:
        # Generate multiple parallelisms in cell to profile
        cell.enum_parallel_plans(cell_prof_strategy)

    assert cell.is_parallel_enumerated(), \
        f"Cell is not parallel-enumerated."
    _mode = "auto" if enable_cell_profile else "uniform"
    print(f"[I] Pipeline partition mode: {_mode} | Parallel enum mode: {cell_prof_strategy} | " + 
          f"Parallel plans: {cell.parallel_plans}")

    # ------------------ Step 3. Shard stages and generate hlos ------------------

    # Finally, shard stages with each parallel plan and generate hlos
    hlo_text_table, opt_hlo_text_table, cross_stages_comm_vars_table = {}, {}, {}
    for i, stage_shapes in enumerate(cell.parallel_plans):
        # Trasform `StageShape` into `List[int]`
        submesh_logical_shapes = [_s.stage_shape for _s in stage_shapes]

        (hlo_modules, opt_modules, 
         comm_vars_table) = _shard_stages_and_gen_hlos(
             jax_all_stages, submesh_logical_shapes, submesh_physical_shapes, backend, 
             schedule, global_invars, accumulator_mapping, donate_invars_dict,
             acc_grad_outvars, num_micro_batches, sliced_virtual_meshes, parallel_idx=i,
         )

        hashkey = Cell.gen_hashkey_with_parallelism(submesh_logical_shapes)
        # Hlo text
        hlo_text_table[hashkey], opt_hlo_text_table[hashkey] = {}, {}
        hlo_print_option = xe.HloPrintOptions.short_parsable()
        for j in range(len(hlo_modules)):
            hlo_text_table[hashkey][str(j)] = hlo_modules[j].to_string(hlo_print_option) \
                if hlo_modules[j] is not None else None
            opt_hlo_text_table[hashkey][str(j)] = opt_modules[j].to_string(hlo_print_option) \
                if opt_modules[j] is not None else None

            # For debug
            if j == 2 and hlo_modules[j] is not None and opt_modules[j] is not None:
                with open("./tmp_module_0.txt", "w") as f:
                    f.write(hlo_modules[j].to_string(hlo_print_option))
                with open("./tmp_module_0_opt.txt", "w") as f:
                    f.write(opt_modules[j].to_string(hlo_print_option))
            
            if j == 3 and hlo_modules[j] is not None and opt_modules[j] is not None:
                with open("./tmp_module_1.txt", "w") as f:
                    f.write(hlo_modules[j].to_string(hlo_print_option))
                with open("./tmp_module_1_opt.txt", "w") as f:
                    f.write(opt_modules[j].to_string(hlo_print_option))

        # Comm vars
        cross_stages_comm_vars_table[hashkey] = comm_vars_table
    
    print("")
    print("[I] All parallel plans in the cell have been sharded and compiled.")
    hlo_pth = os.environ.get("HLO_LOG_PATH")
    if not os.path.exists(hlo_pth):
        os.mkdir(hlo_pth)
    # Only-sharded
    pth = os.path.join(hlo_pth, f"{job_id}_sharded_stages.pkl")
    print(f"[TMP] Writing only-sharded HLO text to '{pth}'...")
    with open(pth, "wb") as f:
        pickle.dump(hlo_text_table, f)
    # Optimized
    pth = os.path.join(hlo_pth, f"{job_id}_optimized_stages.pkl")
    print(f"[TMP] Writing optimized HLO text to '{pth}'...")
    with open(pth, "wb") as f:
        pickle.dump(opt_hlo_text_table, f)
    # Inter-stages comm vars
    pth = os.path.join(hlo_pth, f"{job_id}_inter_stages_comm_vars.pkl")
    print(f"[TMP] Writing inter-stages communicated vars to '{pth}'")
    with open(pth, "wb") as f:
        pickle.dump(cross_stages_comm_vars_table, f)


def _get_cand_comm_groups_and_r2d_map(submesh_physical_shapes: Sequence[Tuple[int]],
                                      submesh_logical_shapes: Sequence[Tuple[int]], 
                                      cross_stages_intra_node: bool = False,
                                      cross_stages_inter_nodes: bool = False):
    """ Get submesh for each stage, map replicas to devices in each stage. """
    # [(1, 2), ...] means we need to query offline profiled data with 1 node and 2 devices per node.
    cand_comm_groups = list()
    # stage_idx -> device mapping (from replica ids to device)
    replica_to_device_mapping_table = dict()
    # NOTE: We describe the definition of device mapping in two cases:
    #       - Case 1: When specify #dp=2 and #mp=4 with 8 gpus and 2 hosts, we got replica groups 
    #                 {{0,1,2,3},{4,5,6,7}} for mp and {{0,4},{1,5},{2,6},{3,7}} for dp. 
    #       - Case 2: When specify #dp=4 and #mp=2 with 8 gpus and 2 hosts, we got replica groups 
    #                 {{0,2,4,6},{1,3,5,7}} for dp and {{0,1},{2,3},{4,5},{6,7}} for mp.
    #       Since model parallelism leads to larger communication amount, we believe that the mapping 
    #       from replica ids to device should be like: ((0, 1, 2, 3), (4, 5, 6, 7)), which means 
    #       replica 0-3 in node 1, replica 4-7 in node 2. This is corresponded to the definition of 
    #       `global_device_ids()` in xla.

    for _stage_idx, _submesh in enumerate(submesh_physical_shapes):
        (dp_degree, mp_degree) = submesh_logical_shapes[_stage_idx]
        if dp_degree == 1 and mp_degree == 1:
            continue

        cand_comm_groups = unique_append(cand_comm_groups, _submesh)

        # if dp_degree == 1 or mp_degree == 1:
        #     # Vanilla parallelism inside this stage
        #     cand_comm_groups = unique_append(cand_comm_groups, _submesh)
        # else:
        #     # Hybrid parallelism inside this stage
        #     if mp_degree <= _submesh[1]:
        #         # Perform model parallelism inside each node
        #         cand_comm_groups = unique_append(cand_comm_groups, (1, mp_degree))
        #         # Perform data parallelism across nodes
        #         cand_comm_groups = unique_append(cand_comm_groups, (_submesh[0], _submesh[1] // mp_degree))
        #     else:
        #         # Perform model parallelism across nodes whiling fully occupying these nodes
        #         cand_comm_groups = unique_append(cand_comm_groups, (mp_degree // _submesh[1], _submesh[1]))
        #         # Perform data parallelism across nodes
        #         cand_comm_groups = unique_append(cand_comm_groups, (dp_degree, 1))
        # Replica to device mapping
        replica_to_device_mapping_table[_stage_idx] = gen_replica_id_to_device_mapping(_submesh[0], _submesh[1])
    
    # Add communication group for cross-stages
    if cross_stages_intra_node:
        cand_comm_groups = unique_append(cand_comm_groups, (1, 2))
    if cross_stages_inter_nodes:
        cand_comm_groups = unique_append(cand_comm_groups, (2, 1))
    
    return cand_comm_groups, replica_to_device_mapping_table


def _get_cross_nodes_module_idxs(cell: Cell,
                                 submesh_physical_shapes: Sequence[Tuple[int]],
                                 forward_module_idxs: Sequence[int], 
                                 backward_module_idxs: Sequence[int], 
                                 module_idx_to_stage_idx_mapping: dict):
    """ 
    Get the indexs of all modules that need to transmit output to its 
    successor across nodes. 
    """
    # Cross-nodes stage idxs
    cross_nodes_stage_idxs = []
    _cur_d = 0
    for _i, _submesh in enumerate(submesh_physical_shapes):
        _cur_d += np.prod(_submesh)
        if _cur_d % cell.num_devices_per_host == 0:
            # The currently traversed submeshes can fulfill the 
            # first k hosts, thus this submesh (stage) must be
            # a cross-stages one.
            cross_nodes_stage_idxs.append(_i)
            
    # # Global allocated devices
    # num_hosts = int(os.environ.get("CRIUS_NUM_HOSTS"))
    # num_devices_per_host = int(os.environ.get("CRIUS_NUM_DEVICES_PER_HOST"))

    # # Cross-nodes stage idxs
    # assert all([_submesh == submesh_physical_shapes[0] for _submesh in submesh_physical_shapes]), \
    #     f"Currently we only support even slicing on physical meshes, but got {submesh_physical_shapes}."
    # num_devices_per_stage = np.prod(submesh_physical_shapes[0])
    # assert num_devices_per_host % num_devices_per_stage == 0 or num_devices_per_stage % num_devices_per_host == 0, \
    #     f"Currently we only support #devices_per_host ({num_devices_per_host}) can be divisible by " + \
    #     f"#devices_per_stage ({num_devices_per_stage}) or reversed."
    # if num_devices_per_stage <= num_devices_per_host:
    #     # The last stage on each node is the cross-nodes stage (e.g., stage 2 if stage 0, 1, 2 are on node 1)
    #     _num_stages_per_host = num_devices_per_host // num_devices_per_stage
    #     cross_nodes_stage_idxs = [(_i * _num_stages_per_host - 1) for _i in range(1, num_hosts)]
    # else:
    #     # Each stage should be cross-nodes stage
    #     cross_nodes_stage_idxs = [_i for _i in range(len(submesh_physical_shapes))]
    
    # Cross-nodes module idxs
    cross_nodes_module_idxs = list()
    for _stage_idx in cross_nodes_stage_idxs:
        # Forward module in this stage (not the last stage) and backward module in the next stage (if existed)
        # should be the cross-nodes modules
        for _module_idx in module_idx_to_stage_idx_mapping:
            if module_idx_to_stage_idx_mapping[_module_idx] == _stage_idx \
                and _module_idx in forward_module_idxs[:-1]:
                # Forward
                cross_nodes_module_idxs.append(_module_idx)
            if (module_idx_to_stage_idx_mapping[_module_idx] == _stage_idx + 1) \
                and _module_idx in backward_module_idxs:
                # Backward
                cross_nodes_module_idxs.append(_module_idx)

    return cross_nodes_module_idxs


def estimate_computation_memory_footprint(
    computation: Computation,
) -> float:
    """ 
    Estimate memory footprint of the computation. 

    Note that each HLO module corresponds to one per-GPU 
    replica/sharding. For instance, if a module is 
    one of the two-way data parallelized replicas, then
    its batch dimension has already be divided by 2.

    We only consider memory footprint of `param` operator
    in HLO module, since both module input and operand
    of any operator (e.g., `cudnn-conv`, `cublas-gemm`) are
    constructed with `param` operator.

    For model weights, we estimate by multiplying #elements
    and #bytes-per-element. For gradients and other 
    intermediate results, since in backward and apply_grad 
    computations they are all computed with computing 
    operators, we estimate in the same way.
    
    Finally, we add the memory footprints of forward, backward 
    and apply_grad modules together as the memory footprint
    of the corresponding stage. Note that since these three 
    computations are simultaneously deployed, their `param`s
    are intialized independently, thus the theoratical memory 
    footprint of model weights appear in these three computations
    individually.
    """

    # TODO(chunyu): Refine this by refering to `pipeline.planner.PipelinePlanner._analyze_layer_memory_access()`.
    
    if computation is None:
        return 0.0

    mem_bytes = 0
    for hlo_op in computation.op_group:
        if isinstance(hlo_op, ParamOperator):
            num_elms = np.prod(hlo_op.shape)
            mem_bytes += num_elms * XLA_PRIMITIVE_TYPE_NUM_BYTES[hlo_op.data_type]    
    return mem_bytes


def parse_hlo_and_profile_xla_ops(
    input_cfgs: InputConfigs, 
    cell: Cell,
    parallel_idx: int,
    disable_cupti: bool = False,
    warmup_num: int = 1,
    cross_stages_comm_vars_table: dict = None,
    hlo_text_table: dict = None,
    opt_hlo_text_table: dict = None,
    use_ray_for_parallel_compile: bool = False,
) -> Tuple[List[IterTimeCollection]]:
    """
    For each parallelism within cell, parse the HLO texts, statistically 
    analyzing computation operators and stage memory footprint.
    
    If the memory footprint not exceeds the available memory of XLA device,
    get the composed kernels (computations) and profile kernel performance. 
    """

    print("")
    print(f"[I] (Parallel plan idx: {parallel_idx}) Parsing HLO texts and profiling XLA operators...")
    # Job id
    job_id = input_cfgs.job_id if input_cfgs.job_id is not None else "default"
    # Backend
    local_devices = [xb.local_devices()[0]]     # Only one xla device
    backend = xb.get_device_backend(local_devices[0])
    # Compiling options
    gpu_type = cell.gpu_type
    num_micro_batches = input_cfgs.trainer_cfgs["num_micro_batches"]
    micro_batch_size = input_cfgs.trainer_cfgs["batch_size"] // num_micro_batches
    
    # Cell information
    num_stages = cell.num_stages
    submesh_physical_shapes = cell.gen_submesh_physical_shapes()
    stage_shapes = cell.parallel_plans[parallel_idx]
    # Trasform `StageShape` into `List[int]`
    submesh_logical_shapes = [_s.stage_shape for _s in stage_shapes]

    # Module to stage mapping
    (forward_module_idxs, backward_module_idxs, apply_grad_module_idxs, 
     module_idx_to_stage_idx_mapping) = _module_to_stage_mapping(num_stages)
    # Cross-nodes module idxs
    cross_nodes_module_idxs = _get_cross_nodes_module_idxs(cell,
                                                           submesh_physical_shapes,
                                                           forward_module_idxs,
                                                           backward_module_idxs,
                                                           module_idx_to_stage_idx_mapping)
    
    # Repeated stages in the model to reduce compiling/profiling times
    repeated_stages = os.environ.get("REPEATED_STAGES_IN_MODEL", "none")
    if repeated_stages != "none":
        repeated_stages = [int(_layer) for _layer in repeated_stages.split(",")]
        assert 0 not in repeated_stages, "Stage 0 should not be in repeated stages."
        cached_module_kernel_time = [None, None, None]   # (forward, backward, apply_grad)
        print(f"\n\n[I] Repeated stages in the model: {repeated_stages}")
    else:
        repeated_stages = []
        cached_module_kernel_time = None
    
    # Communication time table
    comm_time_tables = dict()
    # Module times (computation, intra-stage communication, grad-sync communication, cross-stages communication)
    forward_times, backward_times, apply_grad_times = [IterTimeCollection() for _ in range(num_stages)], \
                                                      [IterTimeCollection() for _ in range(num_stages)], \
                                                      [IterTimeCollection() for _ in range(num_stages)]
    
    # For each stage, specify (1) num_hosts and (2) num_devices_per_host to perform 
    # collective communication (within data or model parallelism). If DP x MP in one
    # stage, further decompose by specifying the hardware spec for each parallelism. 
    # Specifically, for each communication operator, identify its replica groups 
    # carefully and map each replica to GPUs (may with complex communication topologies, 
    # this is to decide which op communicates intra node or inter nodes). Then, we query 
    # corresponding offline-profiled communication data and add into the total 
    # communication time of this stage.
    _cross_stages_inter_nodes = len(cross_nodes_module_idxs) > 0
    _cross_stages_intra_node = len(submesh_logical_shapes) > 1
    (cand_comm_groups, 
     replica_to_device_mapping_table) = _get_cand_comm_groups_and_r2d_map(submesh_physical_shapes, 
                                                                          submesh_logical_shapes,
                                                                          _cross_stages_intra_node,
                                                                          _cross_stages_inter_nodes)
    # Offline-profiled communication data
    for comm_group in cand_comm_groups:
        # Load for each communicationa group
        key = f"{comm_group[0]}_n_{comm_group[1]}_d"
        if os.environ.get("USE_IB_COMM_DATA", "false") == "true":
            # Use comm data profiled with infiniband
            comm_file_name = f"{comm_group[0]}_{gpu_type}_{comm_group[0]}_n_{comm_group[1]}_d_ib.pkl"
        else:
            # Use comm data without infiniband
            comm_file_name = f"{comm_group[0]}_{gpu_type}_{comm_group[0]}_n_{comm_group[1]}_d.pkl"
        comm_data_pth = os.path.join(os.environ.get("COMM_LOG_PATH"), comm_file_name)

        if os.path.exists(comm_data_pth):
            print(f"[TMP] Loading offline profiled communication data: `{comm_file_name}`...")
            with open(comm_data_pth, "rb") as f:
                comm_time_tables[key] = pickle.load(f)
        else:
            # Estimate non-profiled multi-hosts comm group
            comm_time_tables[key] = estimate_nonprof_comm(comm_group[0], comm_group[1], 
                                                          gpu_type, comm_data_pth)
    
    # Memory footprint each stage
    stage_mem_bytes = {}    # stage_idx -> mem_bytes
    
    # Optimized hlos for statistically analyzing computation 
    # operators and stage memory footprint.
    for _module_idx in opt_hlo_text_table:
        module_idx = int(_module_idx)
        analyze_comm = True      
        stage_idx = module_idx_to_stage_idx_mapping[module_idx]
        dp_degree, mp_degree = (submesh_logical_shapes[stage_idx][0], 
                                submesh_logical_shapes[stage_idx][1])
        # Seperate grad_sync in the module if is the backward process 
        # with data parallelism existed.
        seperate_grad_sync = (dp_degree > 1 and 
                              module_idx in backward_module_idxs)
        # Environmental variables
        os.environ["DP_DEGREE"] = str(dp_degree)
        os.environ["MP_DEGREE"] = str(mp_degree)
        os.environ["MICRO_BATCH_SIZE"] = str(micro_batch_size // dp_degree)
        os.environ["NUM_DEVICES_CUR_STAGE"] = str(dp_degree * mp_degree)
        
        # Analyze cross-stage communication if #stage > 1 and not the 
        # last forward/backward stage and not apply_grad module.
        analyze_cross_comm = (analyze_comm and (num_stages > 1) and 
                              module_idx != len(forward_module_idxs) - 1 and 
                              module_idx != 2 * len(forward_module_idxs) - 1 and 
                              (module_idx not in apply_grad_module_idxs))

        cross_stages_comm_vars = cross_stages_comm_vars_table[module_idx] \
            if module_idx in cross_stages_comm_vars_table else None
        device_mapping = replica_to_device_mapping_table[stage_idx] \
            if stage_idx in replica_to_device_mapping_table.keys() else None

        backward_pass = (module_idx in backward_module_idxs)
        send_stage_shape = submesh_logical_shapes[stage_idx]
        if module_idx < num_stages - 1:
            recv_stage_shape = submesh_logical_shapes[stage_idx + 1]
        elif num_stages <= module_idx < 2 * num_stages - 1:
            recv_stage_shape = submesh_logical_shapes[stage_idx - 1]
        else:
            recv_stage_shape = None

        if opt_hlo_text_table[_module_idx] is not None:
            (entry, module_comm_time, grad_sync_comm_time,
            cross_stages_comm_time) = reconstr_hlo_entry_and_stat_comm(comm_time_tables, device_mapping,
                                                                        cross_stages_comm_vars, 
                                                                        opt_hlo_text_table[_module_idx], 
                                                                        send_stage_shape,
                                                                        recv_stage_shape,
                                                                        backward_pass,
                                                                        analyze_comm, 
                                                                        seperate_grad_sync, analyze_cross_comm, 
                                                                        cross_nodes_module=(module_idx in cross_nodes_module_idxs))
        else:
            entry = None
            module_comm_time, grad_sync_comm_time, cross_stages_comm_time = 0.0, 0.0, 0.0
        
        print(f"\Time of module {module_idx} (stage {stage_idx}):")
        print(f"Intra-stage comm time: {module_comm_time}")
        print(f"Grad sync time: {grad_sync_comm_time}")
        print("")
        
        # Estimate memory footprint of the computation.
        mem_bytes = estimate_computation_memory_footprint(entry)
        if stage_idx not in stage_mem_bytes:
            stage_mem_bytes[stage_idx] = mem_bytes
        else:
            stage_mem_bytes[stage_idx] += mem_bytes
        
        # Only statically analyze communication time
        print(f"[I] Optimized module {module_idx} for stage {stage_idx}, " + 
              f"comm time has been statically analyzed.")
        if module_idx in forward_module_idxs:
            # There can be communication in forward propagation of 
            # data parallelism probably due to parameter sync in forward phase.
            forward_times[stage_idx].intra_stage_comm_time = module_comm_time
            forward_times[stage_idx].cross_stage_comm_time = cross_stages_comm_time
        elif module_idx in backward_module_idxs:
            backward_times[stage_idx].intra_stage_comm_time = module_comm_time
            backward_times[stage_idx].grad_sync_comm_time = grad_sync_comm_time
            backward_times[stage_idx].cross_stage_comm_time = cross_stages_comm_time
        elif module_idx in apply_grad_module_idxs:
            apply_grad_times[stage_idx].intra_stage_comm_time = module_comm_time
            apply_grad_times[stage_idx].cross_stage_comm_time = cross_stages_comm_time
        else:
            raise RuntimeError("Mismatched module index.")
    
    # Verify whether OOM occurs
    avail_mem = local_devices[0].available_memory()
    for stage_idx in stage_mem_bytes:
        print(f"[TMP] Memory footprint of the stage {stage_idx} " + 
              f"is {stage_mem_bytes[stage_idx] / GB} GB, " + 
              f"while the available memory of the XLA device is " + 
              f"{avail_mem / GB} GB.")

        if stage_mem_bytes[stage_idx] > avail_mem:
            print(f"[WARN] As estimation, Out-Of-Memory (OOM) error may occur " + 
                  f"with the given parallel plan of the cell. Drop this plan " + 
                  f"and skip compilation and profile of computation operators.")
            return (None, None, None, None, None, None)


    @ray.remote(num_cpus=1, num_gpus=0.1)
    class SingleGPUShardingWorker:    
        """ A Ray actor for single-GPU HLO sharding. 
        
        This worker (for stage i+1) would be executed in parallel with the compile-profile 
        worker of stage i. 
        """

        @staticmethod
        def run_single_device_sharding(wrapped_hlo_module, num_micro_batches_):
            """ Run single-device sharding. """
            sharded_hlo_module, _ = shard_one_hlo_module(wrapped_hlo_module, logical_mesh_shape=(1, 1), 
                                                         force_batch_dim_to_mesh_dim=0, 
                                                         num_micro_batches=num_micro_batches_, 
                                                         num_devices=1, run_spmd_partition=True)

            # local_devices_ = [xb.local_devices(backend='gpu')[0]]     # Only one xla device
            # backend_ = xb.get_device_backend(local_devices_[0])
            # compiled = compile_one_sharded_hlo_module(sharded_hlo_module, num_devices=len(local_devices_), 
            #                                           backend=backend_, bypass_device=False)            

            return sharded_hlo_module

    # We use end-to-end hlo_constr + compile + profile timer 
    _e2e_timer = time.time()
    # Only-sharded hlos for compile and profile computation operators.
    if not use_ray_for_parallel_compile:
    
        hlo_constr_times, compile_times, profile_times = list(), list(), list()
        for _module_idx in hlo_text_table:
            module_idx = int(_module_idx)
            is_forward = (module_idx in forward_module_idxs)
            is_backward = (module_idx in backward_module_idxs)
            is_apply_grad = (module_idx in apply_grad_module_idxs)

            stage_idx = module_idx_to_stage_idx_mapping[module_idx]
            dp_degree, mp_degree = (submesh_logical_shapes[stage_idx][0], 
                                    submesh_logical_shapes[stage_idx][1])
            # Seperate grad_sync in the module if is the backward process 
            # with data parallelism existed.
            seperate_grad_sync = (dp_degree > 1 and 
                                module_idx in backward_module_idxs)
            # Environmental variables
            os.environ["DP_DEGREE"] = str(dp_degree)
            os.environ["MP_DEGREE"] = str(mp_degree)
            os.environ["MICRO_BATCH_SIZE"] = str(micro_batch_size // dp_degree)
            os.environ["NUM_DEVICES_CUR_STAGE"] = str(dp_degree * mp_degree)

            device_mapping = replica_to_device_mapping_table[stage_idx] \
                if stage_idx in replica_to_device_mapping_table else None
            (entry, _, _, _) = reconstr_hlo_entry_and_stat_comm(comm_time_tables, 
                                                                device_mapping,
                                                                None, 
                                                                hlo_text_table[_module_idx])
            
            # Compile and profile the entry computation to get the 
            # kernel execution time by constructing one XLA operator 
            # from the entry computation.
            print(f"[I] Only-sharded module {module_idx} for stage {stage_idx} " + 
                f"(forward: {is_forward}, backward: {is_backward}, apply_grad: {is_apply_grad})," +
                f"constructing entry XLA computation...")
            _time_marker = time.time()
            hlo_module, legacy_gemm_op_num = constr_one_xla_op(entry)
            hlo_constr_times.append(time.time() - _time_marker)
            
            # Compile and profile to get module computation time
            _niter = input_cfgs.trainer_cfgs["niter"]
            (module_kernel_time, per_kernel_infos, 
            _compile_time, _profile_time) = compile_and_profile_one_xla_op(hlo_module, 
                                                                            num_micro_batches, 
                                                                            backend, 
                                                                            local_devices, 
                                                                            _niter, 
                                                                            warmup_num, 
                                                                            disable_cupti)
            compile_times.append(_compile_time)
            profile_times.append(_profile_time)
            
            # Add overhead of legacy gemm operators
            module_kernel_time += legacy_gemm_op_num * AVG_GEMM_KERNEL_TIME
            if legacy_gemm_op_num > 0:
                print(f"[WARN] Due to legacy bugs in tensorflow, there " + 
                    f"are {legacy_gemm_op_num} GEMM operators that be " + 
                    f"estimated with {AVG_GEMM_KERNEL_TIME} s.")
            
            if module_idx in forward_module_idxs:
                forward_times[stage_idx].comp_time = module_kernel_time
            elif module_idx in backward_module_idxs:
                backward_times[stage_idx].comp_time = module_kernel_time
            elif module_idx in apply_grad_module_idxs:
                apply_grad_times[stage_idx].comp_time = module_kernel_time
            else:
                raise RuntimeError("Mismatched module index.")

        # Hlo construction and compilation can be executed concurrently
        hlo_constr_time = np.max(hlo_constr_times)
        compile_time = np.max(compile_times)
        # Profiling can only be executed sequentially since only one gpu
        profile_time = np.sum(profile_times)

    else:
        # Pipeline stage i's profiling with stage i + 1's compilation
        hlo_constr_time, compile_time, profile_time = -1.0, -1.0, -1.0

        # HLO construction for stage 0
        stage_idx = module_idx_to_stage_idx_mapping[0]
        is_forward = (0 in forward_module_idxs)
        is_backward = (0 in backward_module_idxs)
        is_apply_grad = (0 in apply_grad_module_idxs)
        dp_degree, mp_degree = (submesh_logical_shapes[stage_idx][0], 
                                submesh_logical_shapes[stage_idx][1])
        # Seperate grad_sync in the module if is the backward process 
        # with data parallelism existed.
        seperate_grad_sync = (dp_degree > 1 and 0 in backward_module_idxs)
        # Environmental variables
        os.environ["DP_DEGREE"] = str(dp_degree)
        os.environ["MP_DEGREE"] = str(mp_degree)
        os.environ["MICRO_BATCH_SIZE"] = str(micro_batch_size // dp_degree)
        os.environ["NUM_DEVICES_CUR_STAGE"] = str(dp_degree * mp_degree)

        device_mapping = replica_to_device_mapping_table[stage_idx] \
            if stage_idx in replica_to_device_mapping_table else None
        (entry, _, _, _) = reconstr_hlo_entry_and_stat_comm(comm_time_tables, 
                                                            device_mapping,
                                                            None, 
                                                            hlo_text_table['0'])

        # Compile and profile the entry computation to get the 
        # kernel execution time by constructing one XLA operator 
        # from the entry computation.
        print(f"[I] Only-sharded module 0 for stage {stage_idx}, " + 
            f"(forward: {is_forward}, backward: {is_backward}, apply_grad: {is_apply_grad})," +
            f"constructing entry XLA computation...")
        hlo_module, legacy_gemm_op_num = constr_one_xla_op(entry)

        # Sharding worker for stage 0
        shard_worker = SingleGPUShardingWorker.remote()
        shard_ref = shard_worker.run_single_device_sharding.remote(hlo_module, num_micro_batches)

        last_module_idx = 0
        last_shard_ref = shard_ref
        last_shard_worker = shard_worker
        last_legacy_gemm_op_num = legacy_gemm_op_num
        last_stage_idx = stage_idx
        skip_this_stage_profile, skip_last_stage_profile = False, False
        for module_idx in range(1, len(hlo_text_table) + 1):
            module_idx_str = str(module_idx)

            is_forward = (module_idx in forward_module_idxs)
            is_backward = (module_idx in backward_module_idxs)
            is_apply_grad = (module_idx in apply_grad_module_idxs)

            # Launch this stage's HLO construction/sharding/SPMD partitioning
            # HLO construction
            stage_idx = module_idx_to_stage_idx_mapping[module_idx] if module_idx < len(hlo_text_table) else -1
            
            skip_this_stage_profile = False
            if stage_idx in repeated_stages:           
                if is_forward:
                    if cached_module_kernel_time[0] is not None:
                        skip_this_stage_profile = True
                    else:
                        # This stage should be the first repeated forward stage
                        cached_module_kernel_time[0] = 0.0

                if is_backward: 
                    if cached_module_kernel_time[1] is not None:
                        skip_this_stage_profile = True
                    else:
                        # This stage should be the first repeated backward stage
                        cached_module_kernel_time[1] = 0.0

                if is_apply_grad:
                    if cached_module_kernel_time[2] is not None:
                        skip_this_stage_profile = True
                    else:
                        # This stage should be the first repeated apply_grad stage
                        cached_module_kernel_time[2] = 0.0
            
            if module_idx == len(hlo_text_table):
                assert not is_forward and not is_backward and not is_apply_grad
                # Skip HLO constr/sharding/SPMD partitioning, only compile/profile the last stage
                skip_this_stage_profile = True

            if not skip_this_stage_profile:
                dp_degree, mp_degree = (submesh_logical_shapes[stage_idx][0], 
                                        submesh_logical_shapes[stage_idx][1])
                # Seperate grad_sync in the module if is the backward process 
                # with data parallelism existed.
                seperate_grad_sync = (dp_degree > 1 and module_idx in backward_module_idxs)
                # Environmental variables
                os.environ["DP_DEGREE"] = str(dp_degree)
                os.environ["MP_DEGREE"] = str(mp_degree)
                os.environ["MICRO_BATCH_SIZE"] = str(micro_batch_size // dp_degree)
                os.environ["NUM_DEVICES_CUR_STAGE"] = str(dp_degree * mp_degree)

                backward_repeated_stages = [s + num_stages for s in repeated_stages]
                if is_backward and (module_idx in backward_repeated_stages):
                    # Since the backward modules are reversed, but only the first forward stage has been
                    # constructed, we prefetch the backward module of the first forward stage.
                    assert module_idx == backward_repeated_stages[0], \
                        f"The first backward stage should be {backward_repeated_stages[0]}, but got {module_idx}."
                    module_idx_str = str(backward_repeated_stages[-1])

                    # print("Modified backward module index:", module_idx_str)
                    # print(f"Prefetched hlo text is None?: {hlo_text_table[module_idx_str] is None}")
                    # exit(0)

                device_mapping = replica_to_device_mapping_table[stage_idx] \
                    if stage_idx in replica_to_device_mapping_table else None
                (entry, _, _, _) = reconstr_hlo_entry_and_stat_comm(comm_time_tables, 
                                                                    device_mapping,
                                                                    None, 
                                                                    hlo_text_table[module_idx_str])
                
                # Compile and profile the entry computation to get the 
                # kernel execution time by constructing one XLA operator 
                # from the entry computation.
                print(f"[I] Only-sharded module {module_idx} for stage {stage_idx}, " + 
                    f"(forward: {is_forward}, backward: {is_backward}, apply_grad: {is_apply_grad})," +
                    f"constructing entry XLA computation...")
                hlo_module, legacy_gemm_op_num = constr_one_xla_op(entry)

                # Compilation
                shard_worker = SingleGPUShardingWorker.remote()
                shard_ref = shard_worker.run_single_device_sharding.remote(hlo_module, num_micro_batches)
            
            else:
                print(f"[I] Skipped only-sharded module {module_idx} " + 
                      f"(forward: {is_forward}, backward: {is_backward}, apply_grad: {is_apply_grad}) " +
                      f"for stage {stage_idx}'s HLO constr/sharding/SPMD partition. ")

            # Profile the last stage
            is_last_forward = (last_module_idx in forward_module_idxs)
            is_last_backward = (last_module_idx in backward_module_idxs)
            is_last_apply_grad = (last_module_idx in apply_grad_module_idxs)

            if not skip_last_stage_profile:
                sharded_hlo_module = ray.get(last_shard_ref)
                compiled = compile_one_sharded_hlo_module(sharded_hlo_module, num_devices=len(local_devices), 
                                                    backend=backend, bypass_device=False)

                # Profile executable
                print(f"[I] Profiling HLO modules {last_module_idx} " + 
                      f"(forward: {is_last_forward}, backward: {is_last_backward}, apply_grad: {is_last_apply_grad}) " +
                      f"for stage {last_stage_idx}...")
                module_kernel_time, per_kernel_infos = profile_one_compiled_executable(
                    compiled, backend, local_devices, input_cfgs.trainer_cfgs["niter"], 
                    warmup_num, disable_cupti,
                )

                # Add overhead of legacy gemm operators
                module_kernel_time += last_legacy_gemm_op_num * AVG_GEMM_KERNEL_TIME
                if last_legacy_gemm_op_num > 0:
                    print(f"[WARN] Due to legacy bugs in tensorflow, there " + 
                        f"are {last_legacy_gemm_op_num} GEMM operators that be " + 
                        f"estimated with {AVG_GEMM_KERNEL_TIME} s.")
            
                if is_last_forward:
                    forward_times[last_stage_idx].comp_time = module_kernel_time
                    if last_stage_idx in repeated_stages:
                        # The first repeated forward stage
                        cached_module_kernel_time[0] = module_kernel_time

                elif is_last_backward:
                    backward_times[last_stage_idx].comp_time = module_kernel_time
                    if last_stage_idx in repeated_stages:
                        # The first repeated backward stage
                        cached_module_kernel_time[1] = module_kernel_time

                elif is_last_apply_grad:
                    apply_grad_times[last_stage_idx].comp_time = module_kernel_time
                    if last_stage_idx in repeated_stages:
                        # The first repeated apply_grad stage
                        cached_module_kernel_time[2] = module_kernel_time

                else:
                    raise RuntimeError("Mismatched module index.")
            
            else:
                print(f"[I] Skiped profiling HLO modules {last_module_idx} " + 
                      f"(forward: {is_last_forward}, backward: {is_last_backward}, apply_grad: {is_last_apply_grad}) " +
                      f"for stage {last_stage_idx}...")  
                if is_last_forward:
                    assert cached_module_kernel_time[0] is not None and cached_module_kernel_time[0] > 0.0
                    forward_times[last_stage_idx].comp_time = cached_module_kernel_time[0]
                
                elif is_last_backward:
                    assert cached_module_kernel_time[1] is not None and cached_module_kernel_time[1] > 0.0
                    backward_times[last_stage_idx].comp_time = cached_module_kernel_time[1]
                
                elif is_last_apply_grad:
                    assert cached_module_kernel_time[2] is not None and cached_module_kernel_time[2] > 0.0
                    apply_grad_times[last_stage_idx].comp_time = cached_module_kernel_time[2]
                
                else:
                    raise RuntimeError("Mismatched module index.")

            last_shard_ref = shard_ref
            last_shard_worker = shard_worker
            last_module_idx = module_idx
            last_legacy_gemm_op_num = legacy_gemm_op_num
            last_stage_idx = stage_idx
            skip_last_stage_profile = skip_this_stage_profile

    print(f"[I] HLO construction, compilation and profiling time: {time.time() - _e2e_timer} s")
    
    return (forward_times, backward_times, apply_grad_times, 
            hlo_constr_time, compile_time, profile_time)


def constr_e2e_pipeline_iter_time(
    forward_times: Sequence[IterTimeCollection], 
    backward_times: Sequence[IterTimeCollection], 
    apply_grad_times: Sequence[IterTimeCollection],
    num_micro_batches: int,
    pipeline_strategy: str = "gpipe",
    enable_overlap: bool = False,
) -> Tuple[float, float, float, float]:
    """ 
    Construct end-to-end iteration time of a batch with micro-batch pipeline. 
    Default to gpipe pipeline strategy, additionally consider seperated grad_sync.

    Consider the overlap relationships between computation (and intra-stage communication) 
    and cross-stages communication. When comp overhead > comm overhead, the cross-stages 
    comm between stage i and i + 1 should be overlapped by computation of stage i + 1 
    (reversed in backward pass, except for the first communication); otherwise, stage 
    computation should be overlapped by communication (except for the computation of the 
    first microbatch).

    Gradient apply stage is concurrently done after the last microbatch, while gradient sync 
    (if necessary) is sequentially done during the backward of the last microbatch.

    Args:
     - `pipeline_strategy`: Pipeline strategy used to estimate the end-to-end iteration time. 
                            Options: ["gpipe"].
     - `enable_overlap`: Enable the overlap between the inter-stages communication of stage i -> i + 1 
                         and the computation of stage i + 1. In some cases, the framework does
                         not support this feature by default. Even though the framwork supports 
                         overlappiing (e.g., use multi-streams for computation and communication), 
                         if the GPU memory has been heavily occupied by the computation (of 
                         micro-batch i), the p2p communication (of micro-batch i + 1) cannot be 
                         overlapped since no enough GPU memory to store.
    """

    if forward_times is None:
        # Error occurs during operator profiling. For example, the estimated
        # memory footprint of the specified parallelism within the cell is
        # larger than available gpu memory.
        return -1.0, -1.0, -1.0, -1.0

    if pipeline_strategy == "gpipe":
        return _gpipe_e2e_pipeline(
            forward_times, backward_times, apply_grad_times, num_micro_batches, enable_overlap,
        )
    else:
        raise ValueError(f"Invalid pipeline strategy: {pipeline_strategy}")


def _gpipe_e2e_pipeline(
    forward_times: Sequence[IterTimeCollection], 
    backward_times: Sequence[IterTimeCollection], 
    apply_grad_times: Sequence[IterTimeCollection],
    num_micro_batches: int,
    enable_overlap: bool = False,
) -> Tuple[float, float, float, float]:
    """ 
    Gpipe style e2e pipeline latency modeling construction. 

    The schedule will look like below:
        (i,j): tuple of each executable 
        st: index of stage
        i: index of micro-batch
        j: index of partition/device
        t: clock number

         t    0     1     2     3     4     5
        --  ----- ----- ----- ----- ----- -----
        s1  (0,0) (1,0)                   (reverse)
        s2        (0,1) (1,1)
        s3              (0,2) (1,2)
        s4                    (0,3) (1,3) 
    
    In this example, there are 4 stages and the number of micro batches B = 2. There are two scenarios
    of the construction of pipeline formulation (take forward pass as the example):

     - Scenario 1 (overlap is enabled):
    
       fw_stage_j (forward intra-stage time) = max(
                                                    fw_comp_t_j (forward computation time) + 
                                                    fw_ia_comm_t_j (forward intra-stage comm time),
                                                    fw_ir_comm_t_{j-1} (forward inter-stage comm time)
                                               )
    
       fw_e2e_t = (B - 1) * max(fw_stage_j) * (B - 1) + sum(fw_stage_j) + sum(fw_ir_comm_t_j)
    
     - Scenario 2 (overlap is disabled):

       fw_stage_j (forward intra-stage time) = fw_comp_t_j (forward computation time) + 
                                               fw_ia_comm_t_j (forward intra-stage comm time)
       
       fw_e2e_t = max(fw_stage_j) * (B - 1) + sum(fw_stage_j) + 
                  max(fw_ir_comm_t_j) * (B - 1) + sum(fw_ir_comm_t_j)                                 
    """

    e2e_t = 0
    num_stages = len(forward_times)

    # --------------- Forward pass ---------------
    # Forward intra-stage computation & communication
    fw_stage_times = []
    for i in range(num_stages):
        if enable_overlap:        
            fw_stage_times.append(
                max(
                    forward_times[i].comp_time + forward_times[i].intra_stage_comm_time,
                    forward_times[i - 1].cross_stage_comm_time,
                ) if i > 0 else (forward_times[i].comp_time + forward_times[i].intra_stage_comm_time)
            )
        else:
            fw_stage_times.append(
                (forward_times[i].comp_time + forward_times[i].intra_stage_comm_time)
            )
    # Forward inter-stages communication
    fw_inter_stages_comm_times = [_t.cross_stage_comm_time for _t in forward_times]
    # Update end-to-end time
    e2e_t += max(fw_stage_times) * (num_micro_batches - 1) + \
             sum(fw_stage_times) + \
             max(fw_inter_stages_comm_times) * (num_micro_batches - 1) + \
             sum(fw_inter_stages_comm_times)

    # --------------- Backward pass ---------------
    # Backward intra-stage computation & communication
    bw_stage_times = []
    for i in range(num_stages):
        if enable_overlap:        
            bw_stage_times.append(
                max(
                    backward_times[i].comp_time + backward_times[i].intra_stage_comm_time,
                    backward_times[i + 1].cross_stage_comm_time,
                ) if i < (len(backward_times) - 1) else (backward_times[i].comp_time + backward_times[i].intra_stage_comm_time)
            )
        else:
            bw_stage_times.append(
                (backward_times[i].comp_time + backward_times[i].intra_stage_comm_time)
            )
    # Backward inter-stages communication
    bw_inter_stages_comm_times = [_t.cross_stage_comm_time for _t in backward_times]
    # Update end-to-end time
    e2e_t += max(bw_stage_times) * (num_micro_batches - 1) + \
             sum(bw_stage_times) + \
             max(bw_inter_stages_comm_times) * (num_micro_batches - 1) + \
             sum(bw_inter_stages_comm_times)

    # --------------- Grad-sync and apply-grad pass ---------------
    # Gradient synchronization time
    e2e_t += sum([_t.grad_sync_comm_time for _t in backward_times])
    # Apply gradient time
    apply_grad_time_vals = [(_t.comp_time + _t.intra_stage_comm_time) for _t in apply_grad_times if (_t.comp_time + _t.intra_stage_comm_time) < 1e5]
    e2e_t += max(apply_grad_time_vals) if len(apply_grad_time_vals) > 0 else 0

    print_kernel_times = os.environ.get("PRINT_KERNEL_TIMES", "false") == "true"
    if print_kernel_times:
        # Kernel latencies
        comp_time, intra_stage_comm_time, inter_stage_comm_time = 0.0, 0.0, 0.0
        for time_collect in forward_times + backward_times + apply_grad_times:
            comp_time += time_collect.comp_time if time_collect.comp_time < 1e5 else 0.0
            intra_stage_comm_time += (time_collect.intra_stage_comm_time + time_collect.grad_sync_comm_time)
            inter_stage_comm_time += time_collect.cross_stage_comm_time
        return comp_time, intra_stage_comm_time, inter_stage_comm_time, e2e_t

    return -1.0, -1.0, -1.0, e2e_t


def measure_thr_with_alpa_enabled(input_cfgs: InputConfigs):
    """ Measure the throughput of the model with specified parallelism by profiling with alpa. """

    # Environmental variables
    os.environ["XLA_FLAGS"] = f"--xla_gpu_autotune_level={XLA_AUTO_TUNE_LEVEL}"
    os.environ["NCCL_USE_MULTISTREAM"] = NCCL_USE_MULTISTREAM
    
    # global_config.profile_with_whole_ray_cluster = False
    global_config.pipeline_distributed_compile = False
    # Disable load balance in inter-stages communication
    global_config.resharding_loadbalance_mode = "no_loadbalance"
    # global_config.resharding_loadbalance_mode = "normal"
    # global_config.enable_overlapping = True
    # global_config.nccl_mode = "xla_extension"
    # global_config.use_local_allgather = False

    print("[I] Setting sync timer in Alpa workers to perform accurate execution time.")
    # Sync all activities on the worker before begining to sync and calculate time cost
    global_config.shard_parallel_sync_for_timer = True
    global_config.pipeline_sync_for_timer = True

    # Init ray cluster and alpa backend. If using profiling tools (e.g., nsys), (re-)establish a new
    # ray cluster to ensure that the ray instance is a child process of this python script.
    init_backend()
    # Prepare model
    trainer, _, _, _, _ = prepare_flax_model(input_cfgs, enable_alpa=True, 
                                             disable_alpa_profiling_db=False)
    # Train
    _time_marker = time.time()
    e2e_iter_time = trainer.train()
    _num_devices = input_cfgs.hardware_configs.num_nodes * input_cfgs.hardware_configs.num_devices_per_node
    _measure_time = time.time() - _time_marker
    print(f"[I] Measuring one parallelzing configuration takes {_measure_time} (GPU time = {_measure_time * _num_devices}) s.")
    
    return -1.0, -1.0, -1.0, e2e_iter_time


def optimize_thr_with_alpa_enabled(input_cfgs: InputConfigs):
    """ Measure model's e2e iteration time with the optimized parallelism searched by alpa. """    
    
    # Global configs
    # global_config.profile_with_whole_ray_cluster = False
    # global_config.pipeline_distributed_compile = False
    
    # Environmental variables
    num_devices = input_cfgs.hardware_configs.num_nodes * \
        input_cfgs.hardware_configs.num_devices_per_node
    os.environ["CRIUS_GLOBAL_DEVICE_NUM"] = str(num_devices)    # Global device num
    # Whether to skip legacy error (in tensorflow) layer combination
    os.environ["CRIUS_SKIP_LEGACY_ERROR"] = "true"

    batch_size = input_cfgs.trainer_cfgs["batch_size"]
    num_micro_batches = input_cfgs.trainer_cfgs["num_micro_batches"]
    assert batch_size % num_micro_batches == 0, \
        f"Global batch size {batch_size} is not divisible by num micro batches {num_micro_batches}."
    
    # FIXME(chunyu): Remove like in estimate_one_cell(), maybe risky.
    # local_batch_size = batch_size // num_micro_batches
    # if local_batch_size < num_devices:
    #     print(f"[WARN] Local batch size {local_batch_size} is not divisible by num devices {num_devices}, skipping...")
    #     return None, None, None, None

    if input_cfgs.disable_alpa_profiling_db:
        # Disable alpa's profiling database
        print("[I] Alpa's built-in profiling database is disabled.")
    
    from alpa.util import disable_tqdm_globally
    disable_tqdm_globally()
    # Init ray cluster and alpa backend
    init_backend()
    
    # Prepare model
    trainer, _, _, _, _ = prepare_flax_model(input_cfgs, enable_alpa=True, 
                                             disable_alpa_profiling_db=input_cfgs.disable_alpa_profiling_db)
    # Train
    try:
        ret_value = list()

        def train_func():
            ret_value.append(trainer.train(job_id=input_cfgs.rt_job_id, 
                                           try_idx=input_cfgs.try_idx))

        # train_func()
        
        # Run with timeout
        print(f"[I] Compiling and executing model with timeout = {MAX_TRAIN_TIMEOUT} s...")
        t = threading.Thread(target=train_func, args=(), daemon=True)
        t.start()
        t.join(timeout=MAX_TRAIN_TIMEOUT)
        if t.is_alive():
            raise TimeoutError(f"Exceed timeout = {MAX_TRAIN_TIMEOUT} in compiling and executing model.")
        
        if len(ret_value) == 0 or ret_value[0] <= 0:
            raise RuntimeError("Error occurred, resubmitting target job...")
    
    except TimeoutError as e:
        print(f"[E] Meet timout error, killing training process: {e}")
        traceback.print_exc()
        # Kill current process and all subprocesses
        os.killpg(os.getpgid(os.getpid()), 9)
        time.sleep(5)
        
        return None, None, None, None
    except Exception as e:
        print(f"[E] Meet unexpected error in compiling and executing model, killing training process: {e}")
        traceback.print_exc()
        # Kill current process and all subprocesses
        os.killpg(os.getpgid(os.getpid()), 9)
        time.sleep(5)
        
        return None, None, None, None
    
    _iter_time = ret_value[0] if len(ret_value) > 0 else None

    return -1.0, -1.0, -1.0, _iter_time


def estimate_one_cell(
    input_cfgs: InputConfigs, 
    warmup_num: int,
    enable_cell_profile: bool = False,
    cell_prof_strategy: str = "auto",
    # enable_auto_pipeline: bool = False,
    max_plan_set_size: int = 32,
    only_symmetric_sharding: bool = False,
    only_universal_shape: bool = False,
    max_universal_shape_num: int = 0,
    universal_shape_stage_num: int = 0,
    min_layers_per_stage: int = 1,
    disable_cupti: bool = False,
    use_ray_for_parallel_compile: bool = False,
    use_one_microbatch_for_profile: bool = False,
) -> Optional[Cell]:
    """ 
    Estimate the end-to-end iteration time with given cell configuration. 
    
    Args:
     - `input_cfgs`: Input configurations of Crius profiler.
     - `warmup_num`" Number of warmup steps before profiling operator execution time.
     - `enable_cell_profile`: Description in `load_model_and_generate_sharded_hlo`.
     - `cell_prof_strategy`: Description in `load_model_and_generate_sharded_hlo`.
     - `max_plan_set_size`: Maximum size of the Pareto optimal plan set.
     - `only_symmetric_sharding`: Only allow plans that each stage is allocated with the
                                  same GPU num compared to each other.
     - `only_universal_shape`: Only allow all stages to share the same shape when 
                               determining intra-stage parallelism.
     - `max_universal_shape_num`: Maximum stage num that share the same stage logical shape.
     - `universal_shape_stage_num`: Only allow the first several stages to share the 
                                    same shape when determining intra-stage parallelism.
     - `min_layers_per_stage`: Minimal layer num per pipeline stage.
     - `disable_cupti`: Disable profiling operator (i.e., cuda kernel) performance 
                        with Nvidia cupti toolkit, simply execute operators instead.
     - `use_ray_for_parallel_compile`: Use ray actors for parallel HLO compilation.
     - `use_one_microbatch_for_profile`: Use one micro-batch for single-device profiling, then all for e2e modeling.
    """
    
    assert os.environ.get("ENABLE_CRIUS_PROFILER", "false") == "true", \
        "Please specify `export ENABLE_CRIUS_PROFILER=true` before executing the script."

    # Environmental variables
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                              f" --xla_gpu_autotune_level={XLA_AUTO_TUNE_LEVEL}")
    os.environ["NCCL_USE_MULTISTREAM"] = NCCL_USE_MULTISTREAM
    
    num_devices = input_cfgs.hardware_configs.num_nodes * \
        input_cfgs.hardware_configs.num_devices_per_node

    batch_size = input_cfgs.trainer_cfgs["batch_size"]
    num_micro_batches = input_cfgs.trainer_cfgs["num_micro_batches"]
    assert batch_size % num_micro_batches == 0, \
        f"Global batch size {batch_size} is not divisible by num " + \
        f"micro batches {num_micro_batches}."

    # NOTE(chunyu): Instead of naively check, we limit Cell's parallel plan enumeration: DP degree <= local batch size.    
    local_batch_size = batch_size // num_micro_batches
    os.environ["CRIUS_LOCAL_BATCH_SIZE"] = str(local_batch_size)
    # if local_batch_size < num_devices:
    #     print(f"[WARN] Local batch size {local_batch_size} is not divisible " + 
    #           f"by num devices {num_devices}, skipping...")
    #     return None
    
    if use_ray_for_parallel_compile:
        print(f"\n\n=================== Using Ray for parallel compilation ===================\n\n")
        os.environ["USE_RAY_FOR_PARALLEL_COMPILE"] = "true"
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        init_ray_cluster()

    if use_one_microbatch_for_profile:
        print(f"\n\n[I] Using one micro-batch for profiling (microbatch size: {local_batch_size}, " + 
              f"original global batch size: {batch_size}), then all for e2e modeling.")
        os.environ["USE_ONE_MICROBATCH_FOR_PROFILE"] = "true"
        os.environ["CRIUS_ORIGINAL_NUM_MICRO_BATCHES"] = str(input_cfgs.trainer_cfgs["num_micro_batches"])
        input_cfgs.trainer_cfgs["num_micro_batches"] = 1
        input_cfgs.trainer_cfgs["batch_size"] = local_batch_size

    # Cell for performance estimation
    cell = Cell(
        job_id=input_cfgs.job_id, 
        model_name=input_cfgs.trainer_cfgs["model_name"],
        hardware_cfgs=input_cfgs.hardware_configs, 
        cell_cfgs=CellConfigs(
            max_plan_set_size=max_plan_set_size,
            only_symmetric_sharding=only_symmetric_sharding,
            only_universal_shape=only_universal_shape,
            max_universal_shape_num=max_universal_shape_num,
            universal_shape_stage_num=universal_shape_stage_num,
            min_layers_per_stage=min_layers_per_stage,
        ),
        num_stages=input_cfgs.parallel_method.forward_stage_num,
    )
    
    # Pipeline planner
    pipeline_planner = PipelinePlanner(cell, input_cfgs.trainer_cfgs["num_micro_batches"])

    # Step 1. Load target model and generate hlo text
    load_model_and_generate_sharded_hlo(
        input_cfgs, 
        cell,
        pipeline_planner,
        enable_cell_profile=enable_cell_profile,
        cell_prof_strategy=cell_prof_strategy,
    )
    
    # Step 2. Parse hlo text, analyze communication, profile kernel-granularity 
    #         computation and model e2e pipeline iteration time
    
    # Inter-stages communicated variables
    hlo_pth = os.environ.get("HLO_LOG_PATH")
    pth = os.path.join(hlo_pth, f"{cell.job_id}_inter_stages_comm_vars.pkl")
    with open(pth, "rb") as f:
        cross_stages_comm_vars_table = pickle.load(f)
    # Only-sharded
    with open(
        os.path.join(
            hlo_pth, f"{cell.job_id}_sharded_stages.pkl"
        ), "rb",
    ) as f:
        hlo_text_table = pickle.load(f)
    # Optimized
    with open(
        os.path.join(
            hlo_pth, f"{cell.job_id}_optimized_stages.pkl"
        ), "rb"
    ) as f:
        opt_hlo_text_table = pickle.load(f)
    
    # Parse hlo, construct xla computation and profile
    for i, stage_shapes in enumerate(cell.parallel_plans):    
        # Trasform `StageShape` into `List[int]`
        submesh_logical_shapes = [_s.stage_shape for _s in stage_shapes]
        print("")
        print(f"[I] Parallel plan idx: {i} | Parallelism: {submesh_logical_shapes}")

        hashkey = Cell.gen_hashkey_with_parallelism(submesh_logical_shapes)
        (forward_times, backward_times, 
        apply_grad_times, hlo_constr_time, 
        compile_time, profile_time) = parse_hlo_and_profile_xla_ops(input_cfgs,
                                                                    cell, i, 
                                                                    disable_cupti, 
                                                                    warmup_num,
                                                                    cross_stages_comm_vars_table[hashkey],
                                                                    hlo_text_table[hashkey],
                                                                    opt_hlo_text_table[hashkey],
                                                                    use_ray_for_parallel_compile)
        
        if use_one_microbatch_for_profile:
            # Restore the original number of microbatches
            input_cfgs.trainer_cfgs["num_micro_batches"] = int(os.environ.get("CRIUS_ORIGINAL_NUM_MICRO_BATCHES", "1"))
            print(f"[I] Restored the original number of microbatches: {input_cfgs.trainer_cfgs['num_micro_batches']}.")

        (comp_e2e_iter_time, 
        comm_e2e_iter_time,
        cross_e2e_iter_time, 
        e2e_iter_time) = constr_e2e_pipeline_iter_time(forward_times, 
                                                            backward_times, 
                                                            apply_grad_times, 
                                                            input_cfgs.trainer_cfgs["num_micro_batches"])
        
        hashkey = Cell.gen_hashkey_with_parallelism(submesh_logical_shapes)
        cell.update_perf(hashkey, e2e_iter_time)

        print(f"[I] Constructing single-device HLO modules concurrently takes {hlo_constr_time} s.")
        print(f"[I] Compile single-device HLO modules concurrently takes {compile_time} s.")
        print(f"[I] Profile all compiled sequentially on one GPU takes {profile_time} s.")
        print(f"[I] Estimated e2e pipeline iteration time is: {e2e_iter_time} s.")
        print(f"    - Kernel computation time is: {comp_e2e_iter_time} s.")
        print(f"    - Communication time is: {comm_e2e_iter_time} s.")
        print(f"    - Cross-stages communication time is: {cross_e2e_iter_time} s.")

    exit(0)
    
    return cell


def profile_once(
    profile_cfgs: ProfileConfigs, 
    estimate_e2e: bool = True, 
    measure_with_alpa: bool = False, 
    optimize_with_alpa: bool = False, 
    enable_alpa_profiling_db: bool = False,
    prune_search_space: bool = False,
    enable_auto_tuning: bool = False,
):
    """ 
    Profile once with the given model and configuration. 
    
    Args:
     - `profile_cfgs`: Common profiling configurations include model, device and parallel plan.
     - `estimate_e2e`: Estimate e2e iteration time of the given cell configuration, in which a
                       cell is constructed based on `profile_cfgs` and profiled.
     - `measure_with_alpa`: Directly measure model's e2e iteration time with the specified 
                            configuration. It should be noted that the auto parallelizing
                            techniques of alpa is not exploited.
     - `optimize_with_alpa`: Measure the e2e iteration time with the optimal parallelism searched
                             by alpa's auto parallelizing techniques.
     - `enable_alpa_profiling_db`: Enable alpa's built-in profiling database for hlo cost model. 
                                   It should be WARNED that enabling this can affect the results 
                                   of parallelism search. Only works when measuring e2e iteration 
                                   time with optimal parallelism to be searched.
     - `prune_search_space`: Manually prune the search space of optimal parallelism. Only works when 
                             measuring e2e iteration time with optimal parallelism to be searched.
     - `enable_auto_tuning`: Automatically generate candidate tuning plan set (i.e., set of parallelism 
                             to be searched) based on the global tuning database.
    """

    if estimate_e2e or measure_with_alpa:
        # Timer of e2e profiling overhead
        time_marker = time.time()

    # Environmental variables
    # Devices
    os.environ["CRIUS_NUM_HOSTS"] = str(profile_cfgs.num_hosts)
    os.environ["CRIUS_NUM_DEVICES_PER_HOST"] = str(profile_cfgs.num_devices_per_host)
    os.environ["GPU_COMPUTE_MAJOR"] = str(profile_cfgs.compute_major)
    if estimate_e2e and int(profile_cfgs.real_gpu_rank) > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(profile_cfgs.real_gpu_rank)
    
    # # Cell profile settings
    # assert (not profile_cfgs.enable_cell_profile or 
    #         profile_cfgs.enable_auto_pipeline), \
    #     "When cell profile is enabled, auto pipeline must also " + \
    #     "be enabled due to unable-to-specified parallelism."
    
    if profile_cfgs.enable_cell_profile:
        os.environ["CRIUS_ENABLE_CELL_PROFILE"] = "true"
        os.environ["CRIUS_CELL_PROFILE_STRATEGY"] = profile_cfgs.cell_prof_strategy
    if profile_cfgs.force_plan_shape_hashkey != "none":
        os.environ["CELL_FORCE_PLAN_SHAPE_HASHKEY"] = profile_cfgs.force_plan_shape_hashkey
    if profile_cfgs.enum_all_parallelism:
        os.environ["ENUM_ALL_PARALLELISM"] = "true"
    if profile_cfgs.disable_plan_set:
        os.environ["DISABLE_PLAN_SET"] = "true"
    if profile_cfgs.load_with_cpu:
        os.environ["CPU_LOAD_MODEL"] = "true"
    if profile_cfgs.print_kernel_times:
        os.environ["PRINT_KERNEL_TIMES"] = "true"
    
    # # Pipeline partition settings
    # # Construct layer-to-stage mapping uniformly or automatically
    # if profile_cfgs.enable_auto_pipeline:
    #     os.environ["CRIUS_ENABLE_AUTO_PIPELINE"] = "true"
    
    # Xla related
    os.environ["XLA_AUTO_TUNE_LEVEL"] = XLA_AUTO_TUNE_LEVEL
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_gpu_enable_async_all_reduce=false" +
                           " --xla_gpu_force_compilation_parallelism=8")
    # Path
    os.environ["HLO_LOG_PATH"] = f"{CUR_PATH}/tmp"
    os.environ["COMM_LOG_PATH"] = f"{CUR_PATH}/comm_data"
    os.environ["KERNEL_METADATA_PATH"] = f"{CUR_PATH}/kernel_metadata"
    os.environ["ALPA_PROF_PATH"] = f"{CUR_PATH}/../crius_worker/jax"
    # Others
    os.environ["PROFILING_WARMUP_NUM"] = str(profile_cfgs.warmup_num)
    #  Compared to 1f1b, "gpipe" consumes more gpu memory
    os.environ["PIPELINE_SCHEDULE"] = "gpipe"
    os.environ["PRINT_LANUCHED_KERNEL"] = "false"
    # Comm
    if profile_cfgs.use_ib_comm_data:
        os.environ["USE_IB_COMM_DATA"] = "true"
    # os.environ["NCCL_DEBUG"] = "INFO"
    # For infiniband rdma
    # Ref: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-disable
    # Specifies which RDMA interfaces to use for communication. Use `rdma link` to check the 
    # interface name of your IB devices.
    # os.environ["NCCL_IB_HCA"] = "mlx5,ibp"
    
    # Virtually-applied mesh shapes
    if profile_cfgs.enable_cell_profile:
        # Multiple parallelisms are generated inside the cell based on 
        # the given gpu num and pipeline stage num.
        num_stages = profile_cfgs.num_pipeline_stages
        submesh_physical_shapes, submesh_logical_shapes = None, None
    # elif profile_cfgs.enable_auto_pipeline:
    #     # Overwrite user-specified physical shapes since might 
    #     # be modified by gpu sharding.
    #     (_, submesh_logical_shapes, num_stages) = _parse_parallel_degrees(profile_cfgs)
    #     submesh_physical_shapes = None
    #     assert submesh_logical_shapes is not None, \
    #         "Auto pipeline partition still requires user-specified parallelism " + \
    #         "to parallelize each stage."
    else:
        # The parallelism degrees are user-specified, physical mesh shapes are uniformly 
        # generated based on stage num and the locality of the allocated gpus.
        (submesh_physical_shapes, 
         submesh_logical_shapes, 
         num_stages) = _parse_parallel_degrees(profile_cfgs) if not optimize_with_alpa \
                                                            else (None, None, None)
    
    if prune_search_space:
        # Manual tuning with specified pruning prompt
        assert optimize_with_alpa, \
            "'prune_search_space' option is only supported with 'optimize_with_alpa' is enabled."

        os.environ["PRUNE_SEARCH_SPACE"] = "true"               
        # Overwrite the layer num in layer coarsening
        os.environ["CRIUS_OVERWRITE_LAYER_NUM"] = str(profile_cfgs.overwrite_coarsened_layer_num)
        # Pruning prompt of [l_p, h_p, l_d, h_d, l_m, h_m]
        os.environ["CRIUS_PRUNE_PROMPT"] = profile_cfgs.prune_prompt
    
    if enable_auto_tuning:
        # Auto tuning based on tuning database
        assert optimize_with_alpa, \
            "'enable_auto_tuning' option is only supported with 'optimize_with_alpa' is enabled."
        
        os.environ["ENABLE_AUTO_TUNING"] = "true"
    
    # Devices
    gpu_type = profile_cfgs.devices_name.split("_")[1]
    assert gpu_type in CAND_GPU_TYPES, f"Unsupported GPU type: {gpu_type}."
    num_hosts = int(os.environ.get("CRIUS_NUM_HOSTS"))
    num_devices_per_host = int(os.environ.get("CRIUS_NUM_DEVICES_PER_HOST"))
    num_gpus = num_hosts * num_devices_per_host
    
    # Model info
    model_name = profile_cfgs.model_name
    param_num = profile_cfgs.param_num
    batch_size = int(profile_cfgs.batch_size)
    num_micro_batches = int(profile_cfgs.num_micro_batches)
    niter = int(profile_cfgs.niter)
    num_pipeline_layers = int(profile_cfgs.num_pipeline_layers)
    if num_pipeline_layers < num_gpus:
        # Overwrite pipeline layer num to no less than gpu num
        num_pipeline_layers = num_gpus
    assert (num_pipeline_layers % num_gpus == 0), \
        f"Number of layers ({num_pipeline_layers}) must be divisible to " + \
        f"the number of GPUs ({num_gpus})."

    # Parallel method
    if estimate_e2e:
        # Jax model will be loaded without jax transformation (i.e., 
        # loaded without alpa parallelization). (1) `forward_stage_num`
        # will be used to intialize cell object. 
        # (2) `submesh_physical_shapes` will be used when auto pipeline 
        # (and cell profiling) is disabled. (3) `submesh_logical_shapes` 
        # will be used when only cell profile (i.e., enumerate parallel 
        # plans) is disabled (auto pipeline is enabled). 
        parallel_method = ParallelMethod(
            forward_stage_num=num_stages,
            forward_stage_layer_id=None,
            submesh_physical_shapes=submesh_physical_shapes,
            submesh_logical_shapes=submesh_logical_shapes,
            auto_sharding_option=None,
        )
    
    elif measure_with_alpa:
        # Directly measure the e2e performance with the specified parallel plan
        if profile_cfgs.force_plan_shape_hashkey != "none":
            # Parse plan_shape_hashkey to formulate parallel method object
            (plan_hashkey, _) = profile_cfgs.force_plan_shape_hashkey.split("::")
            # Forward stage layer ids
            forward_stage_layer_id = Cell.gen_hashkey_with_partition_plan(
                plan_hashkey=plan_hashkey, decode=True,
            )
        
        else:
            # Formulate the parallel plan specified either by `parallel_degrees` (for symmetric) 
            # or by `force_logical_shapes` (mostly for asymmetric, already parsed in 
            # `_parse_parallel_degrees()`).
            assert (submesh_physical_shapes is not None and 
                    submesh_logical_shapes is not None), \
                "Parallelism must be specified when measuring e2e performance, either " + \
                "by specifying `parallel_degrees` (for symmetric) or by " + \
                "`force_logical_shapes` (mostly for asymmetric)."

            # TODO(chunyu): Currently measure e2e performance with profiling 
            #               asymmetric parallel plan here is not supported (FIXME). 
            assert profile_cfgs.force_logical_shapes == "none", \
                "Currently measure e2e performance with profiling asymmetric " + \
                "parallel plan here is not supported."
            
            # Proportionally cluster layers based on the generated 
            # parallelism of each stage.
            num_layers_per_gpu = num_pipeline_layers // num_gpus
            num_layers_per_stage_list = [
                num_layers_per_gpu * np.prod(_submesh) 
                    for _submesh in submesh_logical_shapes
            ]
            forward_stage_layer_id = [
                [
                    _i + _j * num_layers_per_stage_list[_j]
                        for _i in range(num_layers_per_stage_list[_j])
                ] for _j in range(num_stages)
            ]
        
        auto_sharding_option=[{'force_batch_dim_to_mesh_dim': 0} for _ in range(num_stages)]
        parallel_method = ParallelMethod(
            forward_stage_num=num_stages, 
            forward_stage_layer_id=forward_stage_layer_id,
            submesh_physical_shapes=submesh_physical_shapes,
            submesh_logical_shapes=submesh_logical_shapes,
            auto_sharding_option=auto_sharding_option
        )
    
    elif optimize_with_alpa:
        # Load jax model without manually specified parallelism, 
        # auto searched by alpa.
        parallel_method = ParallelMethod(None, None, None, None, None)
    
    else:
        raise ValueError("Please specify one of the following options: (1) `estimate_e2e`; " + 
                         "(2) `measure_with_alpa`; (3) `optimize_with_alpa`.")

    # Hardware configs
    hardware_cfgs = HardwareConfigs(num_hosts, num_devices_per_host, gpu_type)
    
    # Job and trainer configs
    job_id = f"{model_name}_{param_num}_{batch_size}"
    trainer_cfgs = load_trainer_configs(model_name, param_num, 
                                         batch_size, num_micro_batches, 
                                         num_pipeline_layers, niter)
    # Input configs
    input_cfgs = InputConfigs(
        job_id=job_id,
        trainer_cfgs=trainer_cfgs, 
        tmp_pth="./tmp",
        is_dp_only=False, 
        is_pp_only=False, 
        is_mp_only=False,
        is_manual_config_test=(not optimize_with_alpa),
        optimize_with_alpa=optimize_with_alpa,
        disable_alpa_profiling_db=(not enable_alpa_profiling_db),
        parallel_method=parallel_method,
        hardware_configs=hardware_cfgs,
        devices_name=f"{hardware_cfgs.num_nodes}_{hardware_cfgs.gpu_type}",
        rt_job_id=profile_cfgs.job_id, 
        try_idx=profile_cfgs.try_idx,
    )
    
    if optimize_with_alpa:
        # Measure model's e2e iteration time with the optimized parallelism searched by alpa
        return optimize_thr_with_alpa_enabled(input_cfgs)

    if estimate_e2e:
        # Estimate e2e iteration time 
        if profile_cfgs.repeated_stages and profile_cfgs.repeated_stages != "none":
            # Repeated stages in the model
            os.environ["REPEATED_STAGES_IN_MODEL"] = str(profile_cfgs.repeated_stages)
        
        cell = estimate_one_cell(
            input_cfgs, profile_cfgs.warmup_num, profile_cfgs.enable_cell_profile, profile_cfgs.cell_prof_strategy, 
            profile_cfgs.max_plan_set_size, profile_cfgs.only_symmetric_sharding, profile_cfgs.only_universal_shape, 
            profile_cfgs.max_universal_shape_num, profile_cfgs.universal_shape_stage_num, 
            profile_cfgs.min_layers_per_stage, profile_cfgs.disable_cupti, profile_cfgs.use_ray_for_parallel_compile,
            profile_cfgs.use_one_microbatch_for_profile,
        )
        assert cell.is_profiled(), f"Cell is not profiled."
        
        _time_cost = time.time() - time_marker
        print("")
        print(f"[I] The e2e profiling overhead with estimation is {_time_cost} s " + 
              f"(GPU time = {_time_cost} s)")
        
        if not profile_cfgs.enable_cell_profile:
            # Only one user-specified parallelism is profiled
            assert cell.num_parallel_prof() == 1, \
                f"Only one parallelism should be profiled."
            return -1, -1, -1, cell.perf_lookup_table[cell.last_perf_hashkey()]
        else:
            # Multiple parallelisms profiled in cell
            return None, None, None, cell.perf_lookup_table
    
    if measure_with_alpa:
        # Measure e2e iteration time with direct execution
        (_, _, _, e2e_iter_time) =  measure_thr_with_alpa_enabled(input_cfgs)
        _time_cost = time.time() - time_marker
        _num_devices = num_hosts * num_devices_per_host
        print("")
        print(f"[I] The e2e profiling overhead with direct measuring is {_time_cost} s " + 
              f"(GPU time = {_time_cost * _num_devices} s)")
        
        return (-1.0, -1.0, -1.0, e2e_iter_time)


def main():
    """ 
    Entrypoint for measure/optimize with alpa enabled. 

    If the user needs to dummy-test the e2e estimation of cell, set `ENABLE_CRIUS_PROFILER` as "true" before executing 
    this script. However, it should be noted that the profiling results in this case will not be saved. Turn to 
    `./crius_cell_profile.py` for in-group profiling of multiple cells. 
    """

    # Profiling workdir
    if args.measure_with_alpa:
        prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "ground_truth")
    elif args.optimize_with_alpa:
        if not args.prune_search_space and not args.enable_alpa_profiling_db:
            prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "optimal")
        elif args.prune_search_space and not args.enable_alpa_profiling_db:
            prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "optimal_prune")
        elif not args.prune_search_space and args.enable_alpa_profiling_db:
            prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "optimal_with_prof")
        elif args.prune_search_space and args.enable_alpa_profiling_db:
            prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "optimal_prune_with_prof")
    else:
        # Dummy test for estimating e2e iteration time, not recorded
        prof_log_pth = None
    # File path
    if prof_log_pth and not os.path.exists(prof_log_pth):
        os.mkdir(prof_log_pth)
    _file_name = f"{args.model_name}_{args.param_num}_{args.batch_size}.pkl"
    pth = os.path.join(prof_log_pth, _file_name) if prof_log_pth else None
    if pth and os.path.exists(pth):
        print(f"[TMP] Existed profiling results in `{pth}`, updating/rewriting it...")
        try:
            if os.path.getsize(pth) > 0:
                with open(pth, "rb") as f:
                    profile_results = pickle.load(f)
            else:
                profile_results = dict()
        except EOFError:
            assert os.path.getsize(pth) == 0, \
                "EOF error should be triggered by empty pickle file."
            profile_results = dict()
    else:
        print(f"[TMP] Profiling results not found in `{pth}`, creating it...")
        # "{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d"
        #    -> [(parallel_degrees, comp_e2e_iter_time, comm_e2e_iter_time, cross_e2e_iter_time, e2e_iter_time)]
        profile_results = dict()
    
    # Device info
    device_info_table = load_device_info_table()

    record_key = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d"
    if args.optimize_with_alpa and not args.overwrite_data and record_key in profile_results:
        print(f"[TMP] Key `{record_key}` has been profiled in `{pth}` when optimizing with alpa, loading cache...")
        return 

    if args.measure_with_alpa and record_key not in profile_results:
        profile_results[record_key] = list()
    
    # Profile
    gpu_type = args.devices_name.split("_")[1]
    compute_major = device_info_table[gpu_type]["compute_major"] if gpu_type in device_info_table else None
    profile_cfgs = ProfileConfigs(
        # Device
        devices_name=args.devices_name, 
        num_hosts=args.num_hosts, 
        num_devices_per_host=args.num_devices_per_host,
        base_gpu_type=BASE_GPU_TYPE,
        real_gpu_rank=args.real_gpu_rank,                                   # Only for single gpu profiling
        compute_major=compute_major,                                        # (Deprecated)
        # Model
        model_name=args.model_name, 
        param_num=args.param_num,
        batch_size=args.batch_size, 
        num_micro_batches=args.num_micro_batches,
        num_pipeline_layers=args.num_pipeline_layers, 
        niter=args.niter, 
        warmup_num=args.warmup_num, 
        repeated_stages=args.repeated_stages,
        # Parallel plan
        parallel_degrees=args.parallel_degrees,
        # Search pruning
        prune_prompt=args.prune_prompt, 
        overwrite_coarsened_layer_num=args.overwrite_coarsened_layer_num,
        # Job 
        job_id=args.rt_job_id, 
        try_idx=args.try_idx,
        # Cell profile
        enable_cell_profile=args.enable_cell_profile,
        num_pipeline_stages=int(args.num_pipeline_stages),                  # Only for cell profiling
        cell_prof_strategy=args.cell_prof_strategy,
        use_ib_comm_data=args.use_ib_comm_data,
        force_logical_shapes=args.force_logical_shapes,
        enum_all_parallelism=args.enum_all_parallelism,
        disable_plan_set=(not args.enable_plan_set),
        max_plan_set_size=args.max_plan_set_size,
        only_symmetric_sharding=args.only_symmetric_sharding,
        only_universal_shape=args.only_universal_shape,
        max_universal_shape_num=args.max_universal_shape_num,
        universal_shape_stage_num=args.universal_shape_stage_num,
        min_layers_per_stage=args.min_layers_per_stage,
        load_with_cpu=args.load_with_cpu,
        enable_auto_pipeline=args.enable_auto_pipeline,                     # (Experimental)
        # Measure
        force_plan_shape_hashkey=args.force_plan_shape_hashkey,
        # Ablation
        print_kernel_times=args.print_kernel_times,
        # Others
        disable_cupti=args.disable_cupti, 
        skip_regen_hlo=args.skip_regen_hlo,                                 # (Deprecated)
        only_migration=args.only_migration,                                 # (Deprecated)
        use_ray_for_parallel_compile=args.use_ray_for_parallel_compile,
        use_one_microbatch_for_profile=args.use_one_microbatch_for_profile,
    )
    
    (_, _, _, e2e_iter_time) = profile_once(profile_cfgs, args.estimate_e2e, 
                                            args.measure_with_alpa, args.optimize_with_alpa,
                                            args.enable_alpa_profiling_db, args.prune_search_space)

    if isinstance(e2e_iter_time, dict) or not e2e_iter_time:
        # Multiple parallelisms in cell profile 
        # or other infeasible cases. 
        return
    
    # Update profiling results
    if e2e_iter_time > 0.0 and args.optimize_with_alpa:
        profile_results[record_key] = float(e2e_iter_time)
    elif args.measure_with_alpa and args.parallel_degrees != "none":
        _para_degrees = tuple([int(_c) for _c in args.parallel_degrees.split(",")])
        for _rec in profile_results[record_key]:
            if _rec[0] == _para_degrees:
                profile_results[record_key].remove(_rec)
        profile_results[record_key].append((_para_degrees, float(e2e_iter_time)))

    # Store as pickle
    if args.optimize_with_alpa or args.measure_with_alpa:
        _val = profile_results[record_key] if record_key in profile_results else "none"
        print(f"[TMP] Current profiling results of key `{record_key}`: {_val}")
        print(f"[TMP] Updated profiling results stored in `{pth}`...")
        with open(pth, "wb") as f:
            pickle.dump(profile_results, f)


if __name__ == "__main__":
    # Args 
    parser = argparse.ArgumentParser()
    # Hardware settings
    parser.add_argument("--devices_name", default="1_a40", type=str)
    parser.add_argument("--num_devices_per_host", default=2, type=int)
    parser.add_argument("--num_hosts", default=1, type=int)
    parser.add_argument("--ray_address", default="auto", type=str)
    parser.add_argument("--compute_major", default=8, type=int)
    parser.add_argument("--real_gpu_rank", default=-1, type=str, 
                        help="Rank of the GPU that used to compile, parse and profile hlo modules.")
    # Profile options
    parser.add_argument("--estimate_e2e", default=False, action='store_true', 
                        help="Whether to estimate e2e pipeline iteration time of cell.")
    parser.add_argument("--only_migration", default=False, action='store_true', 
                        help="Skip execution and only migrate GPU performance from the profiled base data.")
    parser.add_argument("--skip_regen_hlo", default=False, action='store_true', 
                        help="Skip the regeneration of hlo texts.")
    parser.add_argument("--measure_with_alpa", default=False, action='store_true', 
                        help="Measure model's e2e iteration time by enabling alpa.")
    parser.add_argument("--optimize_with_alpa", default=False, action='store_true', 
                        help="Measure model's e2e iteration time with the optimal parallelism searched by alpa.")
    parser.add_argument("--enable_alpa_profiling_db", default=False, action='store_true', 
                            help="Enable alpa's built-in profiling database for hlo cost model. " + 
                                 "It should be warned that enabling this can affect the results " + 
                                 "of parallelism search.")
    parser.add_argument("--disable_cupti", default=False, action='store_true', 
                        help="Disable Nvidia CUPTI profiling of kernel performance in cpp backend.")
    parser.add_argument("--overwrite_data", default=False, action='store_true', 
                            help="Whether to overwrite profiled performance data.")
    parser.add_argument("--use_ray_for_parallel_compile", default=False, action='store_true', 
                            help="Use multiple ray actors to compile XLA HLOs in parallel.")
    parser.add_argument("--use_one_microbatch_for_profile", default=False, action='store_true', 
                            help="Only use a single microbatch (size: global batch size // microbatch size) for " + 
                                 "single-device profiling, then use all microbatches in e2e latency modeling.")
    # Model settings
    parser.add_argument("--model_name", default="wide_resnet", type=str)
    parser.add_argument("--param_num", default="500M", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_micro_batches", default=16, type=int, 
                        help="The num of micro batches for pipeline. Local bs of each stage = bs / num_mb at each time slot.")
    parser.add_argument("--num_pipeline_layers", default=16, type=int, help="The num of layers for operators clustering.")
    parser.add_argument("--niter", default=1, type=int, 
                        help="Iteration num when profiling one compiled.")
    parser.add_argument("--warmup_num", default=1, type=int, 
                        help="Iteration num of warmup phase before profiling.")
    parser.add_argument("--parallel_degrees", default="none", type=str, 
                        help="Only for symmetric parallelism. Format: `#pp,#dp,#mp`.")
    parser.add_argument("--force_logical_shapes", default="none", type=str, 
                        help="User-specified logical shapes for asymmetric parallelism." + 
                             "Format: `#dp1_#mp1,#dp2_#mp2`, where `,` divides each stage.")
    parser.add_argument("--repeated_stages", default='none', type=str, 
                        help="Repeated stages in the model in the format of: `1,2,3` (0-base).")
    # (Experimental)
    parser.add_argument("--overwrite_parallelism", default=False, action='store_true', 
                        help="Whether to overwrite parallelism with user-specified " + 
                             "submesh shapes when profiling model.")
    parser.add_argument("--enable_auto_pipeline", default=False, action='store_true', 
                        help="Whether to enable auto pipeline partition from " + 
                             "layers to stages.")
    # Plan shape settings
    parser.add_argument("--force_plan_shape_hashkey", default="none", type=str, 
                        help="(for estimate cell parallelism or measuring with alpa) Forcibly specify plan_shape_hashkey with " + 
                             "the format of `f'{plan_hashkey}::{shape_hashkey_stage_1}__{shape_hashkey_stage_2}'`. The detailed " + 
                             "description of this can be found in `cell/cell/gen_pareto_plan_set_for_tuning()`.")
    # Ablation settings
    parser.add_argument("--print_kernel_times", default=False, action='store_true', 
                        help="Print compute and communication kernel times.")
    # Cell profile settings
    parser.add_argument("--enable_cell_profile", default=False, action='store_true', 
                        help="Whether to enable cell profile that profile with " + 
                             "multiple parallel configurations with the specified " + 
                             "resource quota and number of pipeline stages.")
    parser.add_argument("--num_pipeline_stages", default=1, type=int, 
                        help="Number of pipeline stages when cell profile is enabled.")
    parser.add_argument("--cell_prof_strategy", default="auto", type=str, 
                        help="The strategy to generate multiple parallel plans in one cell. " + 
                             "Options: ['minimal', 'uniform', 'auto']")
    parser.add_argument("--enum_all_parallelism", default=False, action='store_true', 
                        help="Enumerate and profile all candidate parallelism w.r.t. `./pipeline/planner/_enum_all_partition_plans_dp()`.")
    parser.add_argument("--enable_plan_set", default=False, action='store_true', 
                        help="Enable the generation of Pareto plan set.")
    parser.add_argument("--max_plan_set_size", default=32, type=int, 
                        help="Maximum size of the Pareto optimal plan set.")
    parser.add_argument("--only_symmetric_sharding", default=False, action='store_true', 
                        help="Only allow plans that each stage is allocated with the same GPU num compared to each other.")
    parser.add_argument("--only_universal_shape", default=False, action='store_true', 
                        help="Only allow all stages to share the same shape when determining intra-stage parallelism.")
    parser.add_argument("--max_universal_shape_num", default=0, type=int, 
                        help="Maximum stage num that share the same stage logical shape.")
    parser.add_argument("--universal_shape_stage_num", default=0, type=int, 
                        help="Only allow the first several stages to share the same shape when determining intra-stage parallelism.")
    parser.add_argument("--min_layers_per_stage", default=1, type=int, 
                        help="Minimal layer num per stage.")
    parser.add_argument("--use_ib_comm_data", default=False, action='store_true', 
                        help="Use profiled communication data with infiniband.")
    parser.add_argument("--load_with_cpu", default=False, action='store_true', 
                        help="Use CPU host memory to load model parameters, states, and optimizer.")
    # Auto tuning options
    # Automatically generate candidate plan set based on tuning database constructed by pipeline planner.
    parser.add_argument("--enable_auto_tuning", default=False, action='store_true', 
                        help="Enable auto tuning to automatically generate candidate tuning plan set (i.e., set of " + 
                             "parallelism to be searched) based on the global tuning database constructed by pipeline planner.")
    # Manual tuning options
    # Manually specify the pruning prompt to reduce search space of tuning.
    parser.add_argument("--prune_search_space", default=False, action='store_true', 
                        help="Whether to prune search space of optimal parallelism by " + 
                             "restricting #stages and max #GPUs-per-stage.")
    parser.add_argument("--overwrite_coarsened_layer_num", default="4", type=str, 
                        help="Force to coarsen pipeline layers into x layers.")
    parser.add_argument("--prune_prompt", default="2_4_1_2_1_2", type=str, 
                        help="Pruning prompts in the format of: [l_p, h_p, l_d, h_d, l_m, h_m].")
    # Runtime options
    parser.add_argument("--rt_job_id", default="none", type=str)
    parser.add_argument("--try_idx", default=1, type=int)
    
    args = parser.parse_args()
    
    # Environmental variables
    os.environ["PROF_LOG_PATH"] = f"{CUR_PATH}/prof_log"
    os.environ["DEVICE_INFO_PATH"] = f"{CUR_PATH}/device_info/device_infos.json"
    os.environ["TUNING_DB_PATH"] = f"{CUR_PATH}/tuning_database"
    os.environ["TUNING_DB_FILENAME"] = f"tuning_database.pkl"

    main()
