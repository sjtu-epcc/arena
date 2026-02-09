#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to profiling all parallelism configs with crius profiling method. """

# FIXME(chunyu): In our alpa-profile.sif image, XLA_AUTO_TUNE_LEVEL of wide-resnet trainer
#                is set to 0, while in our crius-profiler.sif image, XLA_AUTO_TUNE_LEVEL is
#                set to 4 (runtime_profiler.py/measure_thr_with_alpa_enabled()). That's why
#                there exists small error between estimation and profiled data from crius-
#                profile.sif, but larger error between estimation and profiled data from 
#                alpa-profile.sif.

import argparse
import pickle
import numpy as np
import time
import threading
import traceback

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxpr.runtime_profiler import profile_once
from jaxpr.utils import (
    BASE_GPU_TYPE, MAX_TRAIN_TIMEOUT, ProfileConfigs, is_power_of, load_device_info_table)


def _enumerate_all_parallel_degrees(num_devices: int):
    """ Enumerate all candidate parallelism degrees. """
    assert is_power_of(base=2, target=num_devices), \
        f"Total device num ({num_devices}) should be the power of 2."
    # Enumerate each dimensions
    para_degrees = list()
    _log_nd = int(np.log2(num_devices))
    for _p_d in range(0, _log_nd + 1, 1):
        for _d_d in range(0, _log_nd - _p_d + 1, 1):
            _m_d = _log_nd - _p_d - _d_d
            para_degrees.append((pow(2, _p_d), pow(2, _d_d), pow(2, _m_d)))
    return para_degrees


def bisected_traversal():
    """ Traverse symmetric configurations in bisected manner. """
    return NotImplementedError()


def profile_with_all_configs():
    """ Profile the target model with all candidate configurations. """
    if args.measure_with_alpa:
        raise NotImplementedError(
            f"\n\n[BUG FIXED]. The internal error should be caused by: the returned values are DeviceArray, " + \
            f"which should be transformed into float before storing as pickle. Otherwise, pickle load " + \
            f"these values would consumes lots of gpu memory, which will lead to oom error and cause death " + \
            f"of ray actor. \nStill exploit `./measure_all_configs.sh` to isolate crius profiler with directly " + \
            f"measure or optimize with alpa. \n\n" + \
            f"----------------------------------------------------------------------------------------\n\n" + \
            f"Currently, this script is not supported to traverse all configurations with alpa enabled, " + \
            f"since there exists unhandled legacy bugs:\n\n" + \
            f"[BUG DESCRIPTION] When the alpa worker is activated, the Python scripts will not release " + \
            f"the occupied GPU memory, which almost exhausts the majority of the available GPU memory. " + \
            f"In this case, the Ray actor would die due to OOM. " + \
            f"However, if we directly execute `python runtime_profiler.py` with alpa enabled, this " + \
            f"bug won't occur (GPU memory occupied will be periodically released for alpa worker).\n" + \
            f"[SOLUTION] Please use `./measure_all_configs.sh` to traverse with alpa enabled."
        )

    # Profiling workdir
    if args.estimate_e2e:
        prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "estimate_all")
    elif args.measure_with_alpa:
        prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "ground_truth")
    else:
        raise RuntimeError("Unsupported profiling options.")
    if not os.path.exists(prof_log_pth):
        os.mkdir(prof_log_pth)
    # File path
    _file_name = f"{args.model_name}_{args.param_num}_{args.batch_size}.pkl"
    pth = os.path.join(prof_log_pth, _file_name)
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
    
    record_key = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d"    
    if record_key not in profile_results:
        profile_results[record_key] = list()
    
    if False:
        # TODO(chunyu): Not implemented yet.
        # Device info
        device_info_table = load_device_info_table()
        _gpu_type = args.devices_name.split("_")[1]
        compute_major = device_info_table[_gpu_type]["compute_major"]
    
    compute_major = 7
    
    # Enumerate all configs, traverse and profile
    _num_devices = args.num_hosts * args.num_devices_per_host
    all_para_degrees = _enumerate_all_parallel_degrees(_num_devices)

    if len(profile_results[record_key]) > len(all_para_degrees):
        print(f"[TMP] Key `{record_key}` has been duplicatedly profiled, refresh...")
        profile_results[record_key] = list()
    
    if ((len(profile_results[record_key]) == len(all_para_degrees) or 
         (len(profile_results[record_key]) == 1 and _num_devices == 1)) 
         and not args.overwrite_data):
        print(f"[TMP] Key `{record_key}` has been profiled in `{pth}`, loading cache...")
        return 
    
    for _i, _para_degree in enumerate(all_para_degrees):
        print("\n")
        print("------------------------------------------------------------------")
        print(f"- ({_i + 1}/{len(all_para_degrees)}) Profiling configurations (#PP, #DP, #MP): {_para_degree}...")
        print("------------------------------------------------------------------")
        _profile_cfgs = ProfileConfigs(devices_name=args.devices_name, num_hosts=args.num_hosts, 
                                       num_devices_per_host=args.num_devices_per_host, 
                                       base_gpu_type=BASE_GPU_TYPE,
                                       model_name=args.model_name, param_num=args.param_num,
                                       batch_size=args.batch_size, num_micro_batches=args.num_micro_batches,
                                       num_pipeline_layers=args.num_pipeline_layers, niter=args.niter, 
                                       warmup_num=args.warmup_num, parallel_degrees=_para_degree,
                                       disable_cupti=args.disable_cupti, skip_regen_hlo=args.skip_regen_hlo, 
                                       only_migration=args.only_migration,
                                       real_gpu_rank=args.real_gpu_rank, compute_major=compute_major,
                                       overwrite_coarsened_layer_num=None, prune_prompt=None)
        
        e2e_iter_time = None
        try:
            (comp_e2e_iter_time, comm_e2e_iter_time, 
            cross_e2e_iter_time, e2e_iter_time) = profile_once(_profile_cfgs, 
                                                               args.estimate_e2e, 
                                                               args.measure_with_alpa)
        except Exception as e:
            print(f"[E] Meet unexpected error in compiling and executing model: {e}")
            traceback.print_exc()
            # Kill current process and all subprocesses
            # os.killpg(os.getpgid(os.getpid()), 9)
            time.sleep(5)
            continue
        # Record
        # NOTE: The returned values are DeviceArray, which should be transformed into float before storing as pickle.
        #       Otherwise, pickle load these values would consumes lots of gpu memory, which will lead to oom error
        #       and cause death of ray actor.
        if e2e_iter_time is not None and e2e_iter_time > 0:
            profile_results[record_key].append((_para_degree,
                                                float(comp_e2e_iter_time), float(comm_e2e_iter_time), 
                                                float(cross_e2e_iter_time), float(e2e_iter_time)))
        else:
            profile_results[record_key].append(-1)
    
    # Store as pickle
    print(f"[TMP] Updated profiling results stored in `{pth}`...")
    with open(pth, "wb") as f:
        pickle.dump(profile_results, f)


def inspect_dp_oom():
    """ 
    Inspect OOM situations of varying data parallelism configurations on one GPU 
    (by dividing global batch size = global bs // target #gpu).
    """
    prof_log_pth = os.path.join(os.environ.get("PROF_LOG_PATH"), "inspect_dp")
    if not os.path.exists(prof_log_pth):
        os.mkdir(prof_log_pth)
    # File path
    _file_name = f"{args.model_name}_{args.param_num}_{args.batch_size}.pkl"
    pth = os.path.join(prof_log_pth, _file_name)
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
    
    record_key = f"{args.devices_name}_{args.num_hosts}_n_{args.num_devices_per_host}_d"    
    
    # Load cache
    _gpu_type = args.devices_name.split("_")[1]
    tmp_cache_pth = f"./tmp/inspect_dp_oom_cache_{_gpu_type}.pkl"
    if os.path.exists(tmp_cache_pth):
        try:
            if os.path.getsize(tmp_cache_pth) > 0:
                with open(tmp_cache_pth, "rb") as f:
                    cached_results = pickle.load(f)
            else:
                cached_results = dict()
        except EOFError:
            assert os.path.getsize(pth) == 0, \
                "EOF error should be triggered by empty pickle file."
            cached_results = dict()
    else:
        cached_results = dict()
    
    compute_major = 7
    
    # Global batch size for one worker
    _num_devices = args.num_hosts * args.num_devices_per_host
    _batch_size = args.batch_size // _num_devices
    
    print("\n")
    print("--------------------------------------------")
    print(f"Model info: {args.model_name} | Param num: {args.param_num} | " + 
          f"Per-worker batch size: {_batch_size} | Host num: {args.num_hosts} | " + 
          f"Device num per host: {args.num_devices_per_host}.")
    print("--------------------------------------------")
    
    assert args.measure_with_alpa, \
        f"Need to specify `--measure_with_alpa` when inspecting OOM of data parallelism."
    # Parallelism degrees for one worker of data parallelism
    _para_degrees = (1, 1, 1)
    # Profiling configurations
    _gpu_type = args.devices_name.split("_")[1]
    _profile_cfgs = ProfileConfigs(devices_name=f"1_{_gpu_type}", num_hosts=1, 
                                    num_devices_per_host=1, base_gpu_type=BASE_GPU_TYPE,
                                    model_name=args.model_name, param_num=args.param_num,
                                    batch_size=_batch_size, num_micro_batches=args.num_micro_batches,
                                    num_pipeline_layers=args.num_pipeline_layers, niter=args.niter, 
                                    warmup_num=args.warmup_num, parallel_degrees=_para_degrees,
                                    disable_cupti=args.disable_cupti, skip_regen_hlo=args.skip_regen_hlo, 
                                    only_migration=args.only_migration,
                                    real_gpu_rank=args.real_gpu_rank, compute_major=compute_major,
                                    overwrite_coarsened_layer_num=None, prune_prompt=None)
    
    _cached_key = f"{args.model_name}_{args.param_num}_{_batch_size}_{_gpu_type}"
    
    if record_key in profile_results and not args.overwrite_data:
        print(f"[TMP] Key `{record_key}` has been profiled in `{pth}`, loading cache...")
        # Update cache
        cached_results[_cached_key] = profile_results[record_key]
        # Store cache
        with open(tmp_cache_pth, "wb") as f:
            pickle.dump(cached_results, f)
        return
    
    if _cached_key in cached_results:
        # Cached, load and skip execution
        print(f"[TMP] Key `{record_key}` has been profiled in `{pth}` with cached key `{_cached_key}`, loading cache...")
        profile_results[record_key] = cached_results[_cached_key]
        return

    e2e_iter_time = None
    try:
        ret_value = list()

        def train_func():
            ret_value.append(
                profile_once(_profile_cfgs, False, measure_with_alpa=True)
            )
        
        t = threading.Thread(target=train_func, args=(), daemon=True)
        t.start()
        t.join(timeout=MAX_TRAIN_TIMEOUT)
        if t.is_alive():
            raise TimeoutError(f"Exceed timeout = {MAX_TRAIN_TIMEOUT} in inspecting OOM.")

    except TimeoutError as e:
        print(f"[E] Meet timout error: {e}")
        traceback.print_exc()
        # Kill current process and all subprocesses
        # os.killpg(os.getpgid(os.getpid()), 9)
        time.sleep(5)
    except RuntimeError as e:
        print(f"[E] Meet unexpected error in compiling and executing model: {e}")
        traceback.print_exc()
        # Kill current process and all subprocesses
        # os.killpg(os.getpgid(os.getpid()), 9)
        time.sleep(5)
    except Exception as e:
        print(f"[E] Meet unexpected error in compiling and executing model: {e}")
        traceback.print_exc()
        # Kill current process and all subprocesses
        # os.killpg(os.getpgid(os.getpid()), 9)
        time.sleep(5)
    
    e2e_iter_time = ret_value[0][3] if len(ret_value) > 0 and len(ret_value[0]) == 4 else None
    
    # Update profiling results
    if e2e_iter_time is not None and e2e_iter_time > 0:
        profile_results[record_key] = e2e_iter_time
    else:
        profile_results[record_key] = -1.0
    
    # Update cache
    cached_results[_cached_key] = profile_results[record_key]
    
    # Store as pickle
    print(f"[TMP] Updated profiling results stored in `{pth}`...")
    with open(pth, "wb") as f:
        pickle.dump(profile_results, f)

    # Store cache
    with open(tmp_cache_pth, "wb") as f:
        pickle.dump(cached_results, f)


def main():
    """ Entrypoint. """
    # Environmental variables
    os.environ["PROF_LOG_PATH"] = "./jaxpr/prof_log"
    os.environ["DEVICE_INFO_PATH"] = "./jaxpr/device_info/device_infos.json"
    # Whether to enable crius kernel-level profiler
    os.environ["ENABLE_CRIUS_PROFILER"] = "true"

    if args.profile_all_configs:
        profile_with_all_configs()
    
    if args.inspect_dp_oom:
        inspect_dp_oom()


if __name__ == "__main__":
    # Args 
    parser = argparse.ArgumentParser()
    # Configurations of model training
    parser.add_argument("--devices_name", default="1_a40", type=str)
    parser.add_argument("--num_devices_per_host", default=2, type=int)
    parser.add_argument("--num_hosts", default=1, type=int)
    parser.add_argument("--real_gpu_rank", default=-1, type=str, 
                        help="Rank of the GPU that used to compile, parse and profile hlo modules.")
    # Profile options
    parser.add_argument("--estimate_e2e", default=False, action='store_true', 
                        help="Whether to estimate e2e pipeline iteration time of model.")
    parser.add_argument("--profile_all_configs", default=False, action='store_true', 
                        help="Whether to profile all configs (varying parallelism degrees) when estimating e2e iteration time.")
    parser.add_argument("--only_migration", default=False, action='store_true', 
                        help="Whether to skip execution and only migrate GPU performance from the profiled base data.")
    parser.add_argument("--skip_regen_hlo", default=False, action='store_true', 
                        help="Whether to skip the regeneration of hlo texts.")
    parser.add_argument("--measure_with_alpa", default=False, action='store_true', 
                        help="Whether to measure model's e2e iteration time by enabling alpa.")
    parser.add_argument("--disable_cupti", default=False, action='store_true', 
                        help="Whether to disable CUPTI profiling of kernel performance in cpp backend.")
    parser.add_argument("--overwrite_data", default=False, action='store_true', 
                            help="Whether to overwrite profiled performance data.")
    parser.add_argument("--inspect_dp_oom", default=False, action='store_true', 
                            help="Whether to inspect OOM situations of data parallelism.")
    # Profile configs
    parser.add_argument("--model_name", default='wide_resnet', type=str)
    parser.add_argument("--param_num", default='500M', type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_micro_batches", default=16, type=int, help="The num of micro batches for pipeline. \
                                                                        Local bs of each stage = bs / num_mb at each time slot.")
    parser.add_argument("--num_pipeline_layers", default=16, type=int, help="The num of layers for operators clustering.")
    parser.add_argument("--niter", default=1, type=int, 
                        help="Iteration num when profiling one compiled.")
    parser.add_argument("--warmup_num", default=1, type=int, 
                        help="Iteration num of warmup phase before profiling.")
    parser.add_argument("--parallel_degrees", default="1,2,1", type=str, help="Degree format: (#pp, #dp, #mp).")
    args = parser.parse_args()

    main()
