"""
Options of a benchmark case.
------------------------------
Copied from ./alpa/benchmark/alpa/benchmark_parallel_utils.py
"""


from collections import namedtuple
import json
import os
import time
from typing import Optional, Dict, Any, List

import numpy as np
import jax
from jax._src.tree_util import tree_flatten, tree_leaves, tree_unflatten

import ray
import alpa
from alpa import (AutoShardingOption, ShardParallel, PipeshardParallel,
                  ManualStageOption, AutoStageOption, AutoLayerOption,
                  global_config, PhysicalDeviceMesh)
from alpa.device_mesh import MeshHostWorker
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
from alpa.timer import timers
from alpa.util import (print_used_time, to_str_round,
                       count_communication_primitives, GB)

from macro.macro_def import LOG_INTERVAL

BenchmarkCase = namedtuple("BenchmarkCase", [
    "batch_size", "model_config", "num_micro_batches", "parallel_mode",
    "parallel_args"
])

ShardParallelArgs = namedtuple("ShardParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "logical_mesh_shape",
    "force_batch_dim_mapping"
])

UniformParallelArgs = namedtuple("UniformParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "dp", "op", "pp",
    "force_batch_dim_mapping"
])

SearchParallelArgs = namedtuple("SearchParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers", "auto_stage_option"
])

LoadSolutionParallelArgs = namedtuple("LoadSolutionParallelArgs", [
    "prefer_reduce_scatter", "use_remat", "num_auto_layers",
    "forward_stage_layer_ids", "submesh_physical_shapes",
    "submesh_logical_shapes", "submesh_autosharding_option_dicts"
])


def get_pipeshard_parallel_method(benchmark_case: BenchmarkCase,
                                  num_devices_per_host: Optional[int] = None,
                                  allow_mixed_mesh_shape: bool = False,
                                  use_fine_grained_remat: bool = False,
                                  pipeline_schedule: str = "1f1b"):
    """Create the parallel method of a benchmark case.

    Args:
        benchmark_case: The benchmark case.
        num_devices_per_host: The number of devices per host, used in uniform
          parallel mode.
        allow_mixed_mesh_shape: Whether to allow the mixed mesh shape in
          the autosharding pass.
    """

    num_micro_batches = benchmark_case.num_micro_batches
    parallel_mode = benchmark_case.parallel_mode
    parallel_args = benchmark_case.parallel_args

    if parallel_mode == "search":
        assert isinstance(parallel_args, SearchParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         auto_stage_option) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        remat_mode = "coarse_grained_remat" if use_remat else "none"
        auto_stage_option["cached_compute_cost"] = None
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(layer_num=num_auto_layers,
                                         remat_mode=remat_mode),
            stage_option=AutoStageOption(**auto_stage_option))
    elif parallel_mode == "load_solution":
        assert isinstance(parallel_args, LoadSolutionParallelArgs)
        (prefer_reduce_scatter, use_remat, num_auto_layers,
         forward_stage_layer_ids, submesh_physical_shapes,
         submesh_logical_shapes,
         submesh_autosharding_option_dicts) = parallel_args
        add_manual_layer_marker = None
        num_manual_pipeline_stages = None
        add_manual_remat = None
        if use_remat:
            remat_mode = ("fine_grained_remat"
                          if use_fine_grained_remat else "coarse_grained_remat")
        else:
            remat_mode = "none"
        model_num_layers = benchmark_case.model_config.num_layers
        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=AutoShardingOption(
                prefer_reduce_scatter=prefer_reduce_scatter,
                allow_mixed_mesh_shape=allow_mixed_mesh_shape,
            ),
            pipeline_schedule=pipeline_schedule,
            layer_option=AutoLayerOption(
                layer_num=num_auto_layers,
                remat_mode=remat_mode,
                fine_grained_remat_layer_num=model_num_layers),
            stage_option=ManualStageOption(forward_stage_layer_ids,
                                           submesh_physical_shapes,
                                           submesh_logical_shapes,
                                           submesh_autosharding_option_dicts))
    elif parallel_mode == "uniform":
        assert isinstance(parallel_args, UniformParallelArgs)
        (prefer_reduce_scatter, use_remat, dp, op, pp,
         force_batch_dim_mapping) = parallel_args
        as_option = AutoShardingOption(
            prefer_reduce_scatter=prefer_reduce_scatter,
            allow_mixed_mesh_shape=allow_mixed_mesh_shape,
        )
        if force_batch_dim_mapping:
            as_option.force_batch_dim_to_mesh_dim = 0
        add_manual_layer_marker = True
        add_manual_remat = use_remat

        logical_mesh_shape = (dp, op)
        num_manual_pipeline_stages = pp
        num_mesh_devices = np.prod(logical_mesh_shape)
        assert num_devices_per_host is not None
        if num_mesh_devices <= num_devices_per_host:
            physical_mesh_shape = (1, num_mesh_devices)
        else:
            assert num_mesh_devices % num_devices_per_host == 0
            physical_mesh_shape = (num_mesh_devices // num_devices_per_host,
                                   num_devices_per_host)

        method = PipeshardParallel(
            num_micro_batches=num_micro_batches,
            default_auto_sharding_option=as_option,
            pipeline_schedule=pipeline_schedule,
            layer_option="manual",
            stage_option=ManualStageOption(
                forward_stage_layer_ids=[[i] for i in range(pp)],
                submesh_physical_shapes=[physical_mesh_shape] * pp,
                submesh_logical_shapes=[logical_mesh_shape] * pp,
                submesh_autosharding_option_dicts=[{}] * pp))
    else:
        raise ValueError(f"Invalid parallel mode: {parallel_mode}")

    return (method, add_manual_remat, add_manual_layer_marker,
            num_manual_pipeline_stages)


def benchmark_mesh_workers(executable: PipeshardDriverExecutable):
    """ Benchmark mesh workers for each stage. """
    # _res = executable.profile_all_executable_with_dummy_inputs()

    # print(_res)

    for _, physical_mesh in enumerate(executable.mesh_group):
        for _, _worker in enumerate(physical_mesh.workers):
            _worker: MeshHostWorker
            _handle = _worker.profile_executable_with_dummy_inputs.remote(executable.executable_uuids[0], skip_grad_sync=False)
            _res = ray.get(_handle)
            print(_res)


def profile_stage_execution(executable: PipeshardDriverExecutable):
    """ Profile the execution trace of each stage, need to set 'global_config.collect_trace = True'. """
    _res = executable.get_stage_execution_info()

    # TODO: Overwrite get_stage_execution_info() function outside the executable to get the computation time only
    
    print(_res)


def benchmark_training_executable(niter,
                                  train_step,
                                  executable,
                                  state,
                                  other_train_step_inputs,
                                  batches,
                                  dump_debug_file_path,
                                  job_id, try_idx,
                                  profile_driver_time=False):
    print_used_time(None)
    
    # Print auto-sharding intermidiate results
    global_config.pipeline_distributed_compile = False
    # os.environ["ALPA_DEBUG_PRINT_AS_STRATEGY"] = "1"
    # Collect execution trace
    # global_config.collect_trace = True
    
    # # Benchmark stages
    # benchmark_mesh_workers(executable=executable)

    # exit(0)
    
    # Benchmark step time. 
    # NOTE: Add additional warmup rounds to avoid profile bias.
    # warmup = 10 if niter >= 5 else 5

    _warmup_num = os.environ.get("PROFILING_WARMUP_NUM")
    if _warmup_num != "" and _warmup_num is not None:
        warmup = int(_warmup_num)

    # Check
    assert profile_driver_time is True


    if profile_driver_time:
        # Benchmark latency with driver overhead
        global_config.use_dummy_value_for_benchmarking = False
        global_config.shard_parallel_sync_for_timer = False

        if warmup > 0:
            print("[I] Training process warmup ({} rounds) with dummy input batch...".format(warmup))
            for i in range(warmup):
                print(f"    - Warmup iteration: {i + 1}/{warmup}")
                # Use the dummy input batch to warmup
                state, metrics = train_step(state, *other_train_step_inputs)                
                if isinstance(state, tuple):
                    # NOTE: Still needed when profiling with driver overhead
                    # In case the train_step returns extra info (e.g. loss),
                    # Get the actual state out.
                    state = state[0]
            
                executable.sync()
        
        print("[I] Ready to perform training process.")
        print("[I] Benchmark the training process with entire dataset and profile with driver overhead...")

        total_e2e_iter_time = 0

        use_dataset = False
        evaluate_loss = os.environ.get("EVAL_LOSS", "false") == "true"
        losses = []

        cur_pth = os.path.dirname(os.path.abspath(__file__))

        # Check tmp path
        if not os.path.exists(f"{cur_pth}/tmp_res/"):
            os.mkdir(f"{cur_pth}/tmp_res/")
        # Iter cnt file
        iter_cnt_file = f"{job_id}_{str(try_idx)}.txt"
        file_path = os.path.join(f"{cur_pth}/tmp_res", iter_cnt_file)

        # Train 
        for i in range(niter):
            if (i % LOG_INTERVAL == 0):
                print(f"    - Iteration {i + 1} / {niter} is performed...")
            
            # NOTE: Place the timestamp here, record the time of every iteration
            tic = time.time()
            
            if use_dataset: 
                # Supported data shape
                supported_data_shape = batches[0]['images'].shape
                # Batch index
                batch_idx = i % len(batches)
                
                # NOTE: Skip the batch with different batch shape, since it will cause lead alpa
                #       to restart the auto-parallelization search and cause the following error:
                #       - 'NotImplementedError: automatically slicing layers with existing physical 
                #         meshes is notsupported yet'.
                #       Note that currently after alpa is initialized, we only allow a single run of 
                #       auto-parallelization search.
                if i > 0 and supported_data_shape is not None and batches[batch_idx]['images'].shape != supported_data_shape:
                    print("    - Warning: Data shape of batch {} mismatched (which will lead alpa to restart the auto-parallelization search and cause 'NotImplementedError',so we skip this batch). The current data shape is {}, while the proper data shape is: {}" \
                            .format(batch_idx, batches[batch_idx]['images'].shape, supported_data_shape))
                    continue

                state = train_step(state, batches[batch_idx])
            else:
                # Use dummy input
                state, metrics = train_step(state, *other_train_step_inputs)
                if evaluate_loss:
                    loss = metrics["loss"]._value
                    print(f"-------> Loss value of iter {i}: {loss}")
                    losses.append(loss)

            if isinstance(state, tuple):
                state = state[0]

            executable.sync()

            _iter_time = round(time.time() - tic, 3)
            total_e2e_iter_time += _iter_time

            print(f"Iteration time = {_iter_time} s, writing to {file_path}...")

            # Append the iteration time of this iter
            with open(file_path, 'a') as f:
                f.write(str(_iter_time) + '\n')

        # NOTE: Perform parallel inspection of alpa
        # Dump final HLO and other debug info
        # executable.dump_debug_info(dump_debug_file_path)
        # print("[I] The result of the last dynamic programming is as follows:")
        # Print auto-stage dynamic programming results if use auto stage partition
        # print(get_last_dp_result())

        # Total end-to-end training time
        # e2e_total_time = round(time.time() - tic, 3)
        # NOTE: turn to record the time of each iteration
        # End-to-end average latency
        e2e_latency = round(total_e2e_iter_time / niter, 3)
        latencies = [e2e_latency]
        # Total local training time
        # local_lats = executable.get_execution_time_costs()[warmup:]
        local_lats = executable.get_execution_time_costs()[warmup:]
    else:
        # NOTE: Won't be executed.
        # Benchmark latency without driver overhead
        for i in range(niter):
            print(f"Iteration {i} ...")
            state = train_step(state, *other_train_step_inputs)
            if isinstance(state, tuple):
                state = state[0]
            executable.sync()
        # latencies = executable.get_execution_time_costs()[warmup:]
        latencies = executable.get_execution_time_costs()[warmup:]

    print_used_time("Benchmark")

    print("")
    print(executable.get_execution_time_costs())
    print("")

    if evaluate_loss:
        print(f"\n\n[I] Loss values of {niter} iterations:")
        print(losses)

    # profile_stage_execution(executable=executable)

    # exit(0)

    return latencies, total_e2e_iter_time, niter, local_lats


def compile_pipeshard_executable(parallel_mode, train_step, state,
                                 other_train_step_inputs):
    print_used_time(None)

    executable = train_step.get_executable(state, *other_train_step_inputs)
    print_used_time("Compile (driver)")
    
    if parallel_mode == "search":
        compilation_times = {
            k: timers(k).elapsed(mode="sum") for k in [
                "stage-construction", "stage-construction-dp",
                "stage-construction-compilation", "stage-construction-profiling"
            ]
        }
        print(
            f"compilation time breakdown: {to_str_round(compilation_times, 2)}")
    else:
        compilation_times = None

    # executable.dump_debug_info("tmp")
    executable.sync()
    print_used_time("Compile (worker)")
    return executable, compilation_times


def compile_and_benchmark_pipeshard_training_executable(
        parallel_mode,
        niter,
        train_step,
        state,
        other_train_step_inputs,
        batches,
        dump_debug_file_path,
        job_id, try_idx,
        profile_driver_time=False):

    try:
        _time_marker = time.time()
        executable, compilation_times = compile_pipeshard_executable(
            parallel_mode, train_step, state, other_train_step_inputs)
        print(f"[I] Compiling training executable takes {time.time() - _time_marker} s.")
    except RuntimeError as e:
        print(f"[E] Meet unexpected error in compiling executables: {e}")
        return None, None, None, None, None, None, None

    cur_pth = os.path.dirname(os.path.abspath(__file__))
    
    # Check tmp path
    if not os.path.exists(f"{cur_pth}/tmp_res/"):
            os.mkdir(f"{cur_pth}/tmp_res/")
    # Compilation time file
    comp_time_file = f"compile_time_{job_id}_{str(try_idx)}.txt"
    file_path = os.path.join(f"{cur_pth}/tmp_res", comp_time_file)

    # Append the iteration time of this iter
    with open(file_path, 'a') as f:
        f.write(str(compilation_times["stage-construction"]) + '\n')
    
    try:
        _time_marker = time.time()
        latencies, e2e_total_time, niter, local_lats = benchmark_training_executable(
            niter,
            train_step,
            executable,
            state,
            other_train_step_inputs,
            batches,
            dump_debug_file_path,
            job_id, try_idx,
            profile_driver_time=profile_driver_time)
        print(f"[I] Profiling training executable takes {time.time() - _time_marker} s.")
    except:
        print("[E] Meet unexpected error in profiling executables...")
        return None, None, None, None, None, None, None

    max_mem_allocated = executable.mesh_group.get_max_memory_allocated()

    return latencies, e2e_total_time, niter, local_lats, max_mem_allocated, compilation_times, executable


def compute_avg_stage_latencies(timelines: List[tuple]):
    stage_latencies = []
    for request_timeline in timelines:
        sorted_timeline = sorted(request_timeline, key=lambda x: x[0])
        stage_borders = [sorted_timeline[0][0]]
        for _, e, _, _ in sorted_timeline:
            stage_borders.append(e)
        stage_latency = [
            stage_borders[i + 1] - stage_borders[i]
            for i in range(len(stage_borders) - 1)
        ]
        stage_latencies.append(stage_latency)
    return np.mean(stage_latencies, axis=0)
