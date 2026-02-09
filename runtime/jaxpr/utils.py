#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
import json
import pickle
from collections import namedtuple
from typing import Any, Sequence
from dataclasses import dataclass
import numpy as np
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from alpa.util import XlaPassContext

# Since kernel-level profile involves in XLA auto-tuning (e.g., operator fusion), we must enable
# XLA auto-tuning when measuring e2e iteration time with alpa. Otherwise, e2e iteration time of alpa
# (without XLA auto-tuning) will be much longer than kernel-level profile.

# 0:   Disable gemm and convolution autotuning.
# 1:   Enable autotuning, but disable correctness checking.
# 2:   Also set output buffers to random numbers during autotuning.
# 3:   Also reset output buffers to random numbers after autotuning each
#      algorithm.
# 4+:  Also check for correct outputs and for out-of-bounds reads/writes.
# Ref: https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune
XLA_AUTO_TUNE_LEVEL = "0"

# Whether to enable nccl multistream to accelerate communication
NCCL_USE_MULTISTREAM = "True"

# Byte to gigabyte
GB = 1024**3
# Nanosecond to second
NS_TO_S = 1e-9
# Average kernel time of gemm operator
AVG_GEMM_KERNEL_TIME = 1e-3
# Scaling factor for estimating comm time with non-profiled large comm amount
LARGE_COMM_SCALE_FACTOR = 1.1
# Base GPU type for real profiling
BASE_GPU_TYPE = "a40"
# Default settings ref to alpa/shard_parallel/auto_sharding.py
ALL_REDUCE_THRESHOLD = 1 << 60
ALL_GATHER_THRESHOLD = 1 << 60
# Timeout for multi-hosts communication profiling
COMM_PROF_TIMEOUT = 30
# Max device num per host
MAX_DEVICE_NUM_PER_HOST = 16
# Max training timeout 
MAX_TRAIN_TIMEOUT = 5000

# Communication settings
# Global repeated times for each comm size
REPEAT_TIMES_EACH_COMM_SIZE = 5
# Barrier interval
BARRIER_INTERVAL_INTRA_HOST = 5
BARRIER_INTERVAL_INTER_HOSTS = 1
# Total communication amount
TOTAL_COMM_SIZE = 1 << 32
# Minimum communication size
MIN_COMM_SIZE = 1
# Maximum Communication size
MAX_COMM_SIZE = 1 << 28
MAX_COMM_SIZE_P2P = 1 << 25
MAX_COMM_SIZE_LOW_BW = 1 << 28
# Threshold of communication size
THRE_COMM_SIZE = 1 << 20
THRE_COMM_SIZE_P2P = 1 << 17
THRE_COMM_SIZE_LOW_BW = 1 << 22
# Maximum comm size interval
MAX_COMM_SIZE_INTERVAL = 1 << 20
MAX_COMM_SIZE_INTERVAL_P2P = 1 << 17
MAX_COMM_SIZE_INTERVAL_LOW_BW = 1 << 22
# Low bandwidth gpu types
LOW_BW_GPU_TYPES = ["a40", "1080ti", "a10"]


# Candidate GPU type for profiling
CAND_GPU_TYPES = [
    "a10",
    "a40",
    "v100",
    "a100",
    "1080ti",
    "p100"
]

# Available memory of GPU type
AVAIL_MEM_MB_GPU_TYPES = {
    "a40": 45487,
}

# Dtype to xla primitive type
NUMPY_DTYPE_TABLE = {
    "f16": xc.PrimitiveType.F16, 
    "f32": xc.PrimitiveType.F32,
    "s32": xc.PrimitiveType.S32,
    "u32": xc.PrimitiveType.U32,
    "u8": xc.PrimitiveType.U8,
    "pred": xc.PrimitiveType.PRED,
}

# xla primitive type to #bytes
XLA_PRIMITIVE_TYPE_NUM_BYTES = {
    xc.PrimitiveType.F16: 2,
    xc.PrimitiveType.F32: 4,
    xc.PrimitiveType.S32: 4,
    xc.PrimitiveType.U32: 4,
    xc.PrimitiveType.U8: 1,
    xc.PrimitiveType.PRED: 1,
}

# Dtype to float32 ratio
DTYPE_TO_F32_RATIO_TABLE = {
    "float16": 0.5,
    "bool": 0.25,
    "int32": 1,
}

# Profiled metrics of kernels when the compute capability (compute major) of the GPU 
# is higher than 7.0
KERNEL_METRICS = [
    # Single precision flop efficiency
    "smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed",
    # Dram read bytes
    "dram__bytes_read.sum",
    # Dram write bytes
    "dram__bytes_write.sum",
]

# Profiled metrics of kernels when the compute capability (compute major) of the GPU 
# is lower than 7.0
LEGACY_KERNEL_METRICS = [
    # Single precision flop efficiency
    "flop_sp_efficiency",
    # Dram read bytes
    "dram_read_bytes",
    # Dram write bytes
    "dram_write_bytes",
]


"""
Collection of hardware configurations.
"""
HardwareConfigs = namedtuple("HardwareConfigs", [
    "num_nodes", "num_devices_per_node", "gpu_type",
])

"""
Collection of parallelizing method.
"""
ParallelMethod = namedtuple("ParallelMethod", [
    "forward_stage_num", 
    "forward_stage_layer_id", 
    "submesh_physical_shapes",
    "submesh_logical_shapes",
    "auto_sharding_option",
])

"""
Collection of pipeline partition plans.
"""
PipelinePlan = namedtuple("PipelinePlan", [
    # Cluster layers into stages
    "stage_layer_ids", "stage_to_mesh", 
    # Sharded GPU num of each pipeline stage 
    # (i.e., the number of GPUs that allocated 
    # to each pipline stage).
    "gpu_sharding",
])

"""
A collection to represent the logical shapes of all layers in one stage.

Args:
 - `type`: Type of the stage shape as described in docstrings of 
           `_infer_min_intra_stage_comm_fixed_plan()`. Options: ["single", "mixed"].
 - `stage_shape`: Shared logical shape of all layers in the stage.
 - `layer_shape`: Unique logical shape of each layer in the stage.
"""
StageShape = namedtuple("StageShape", [
    "type", "stage_shape", "layer_shape",
])

"""
Collection of multiple operators in a computation.
"""
Computation = namedtuple("Computation", [
    "bias", "op_group",
])

"""
Collection of profiling configurations.
"""
_ProfileConfigs = namedtuple("ProfileConfigs", [
    "devices_name", "num_hosts", "num_devices_per_host", "base_gpu_type", 
    "model_name", "param_num", "batch_size", 
    "num_micro_batches", "num_pipeline_layers", 
    "niter", "warmup_num", "parallel_degrees",
    "disable_cupti", "skip_regen_hlo", "only_migration",
    "real_gpu_rank", "compute_major",
    "overwrite_coarsened_layer_num", "prune_prompt",
    "job_id", "try_idx",
    # Whether to enable cell profile that profile multiple parallelisms 
    "enable_cell_profile",
    "num_pipeline_stages",
    "cell_prof_strategy",
    # (Deprecated) Whether to enable auto pipeline partition and gpu sharding
    "enable_auto_pipeline",
    # User-specified logical shapes for asymmetric parallelism
    "force_logical_shapes",
])

"""
Collection of configurations for collective communication operators.
"""
CollectiveCommConfigs = namedtuple("CollectiveCommConfigs", [
    "op_type", "replica_groups", "data_shape", "data_type",
])

"""
Collection of configurations for p2p send/recv communication operators.
"""
P2PCommConfigs = namedtuple("P2PCommConfigs", [
    "data_shape", "data_type", "src_rank", "dst_rank",
    "src_gpu_idx", "dst_gpu_idx"
])


@dataclass
class IterTimeCollection:
    """ Dataclass of varying parts in iteration time. """
    comp_time: float = 0.0
    intra_stage_comm_time: float = 0.0
    grad_sync_comm_time: float = 0.0
    cross_stage_comm_time: float = 0.0


@dataclass
class ProfileConfigs:
    """ Profling configurations. """
    
    # Device
    devices_name: str = None
    num_hosts: int = None
    num_devices_per_host: int = None
    base_gpu_type: str = None
    real_gpu_rank: int = None
    compute_major: int = None
    # Model
    model_name: str = None 
    param_num: str = None
    batch_size: int = None
    num_micro_batches: int = None
    num_pipeline_layers: int = None
    niter: int = None
    warmup_num: int = None
    repeated_stages: str = None
    # User-specified pipeline + parallel plan
    parallel_degrees: str = None
    # Cell profile
    enable_cell_profile: bool = False
    num_pipeline_stages: int = None
    cell_prof_strategy: str = None
    force_logical_shapes: str = None
    enable_auto_pipeline: bool = False
    enum_all_parallelism: bool = False
    disable_plan_set: bool = False
    max_plan_set_size: int = 32
    only_symmetric_sharding: bool = False
    only_universal_shape: bool = False
    max_universal_shape_num: int = 0
    universal_shape_stage_num: int = 0
    min_layers_per_stage: int = 1
    use_ib_comm_data: bool = False
    load_with_cpu: bool = False
    # Ablation
    print_kernel_times: bool = False
    # Measure
    force_plan_shape_hashkey: str = None
    # Search pruning in manual tuning
    prune_prompt: str = None
    overwrite_coarsened_layer_num: int = None
    # Runtime
    job_id: str = None
    try_idx: int = None
    # Others
    disable_cupti: bool = False
    skip_regen_hlo: bool = False
    only_migration: bool = False
    use_ray_for_parallel_compile: bool = False 
    use_one_microbatch_for_profile: bool = False   

    
@dataclass
class InputConfigs:
    """ Input configurations of the profiler. """

    job_id: str = None
    trainer_cfgs: dict = None
    tmp_pth: str = None
    is_dp_only: bool = False
    is_pp_only: bool = False
    is_mp_only: bool = False
    is_manual_config_test: bool = False
    optimize_with_alpa: bool = False
    disable_alpa_profiling_db: bool = False
    parallel_method: ParallelMethod = None
    hardware_configs: HardwareConfigs = None
    devices_name: str = None
    rt_job_id: str = None
    try_idx: int = None


@dataclass
class CellConfigs:
    """ Configuratiosn of the cell parallelism determination. """

    max_plan_set_size: int = 32
    only_symmetric_sharding: bool = False
    only_universal_shape: bool = False
    max_universal_shape_num: int = 0
    universal_shape_stage_num: int = 0
    min_layers_per_stage: int = 1


def hlo_module_cost_analysis(client,
                             hlo_module,
                             num_micro_batches=1,
                             grad_sync_channel_ids=""):
    """
    Compute and network analysis of an HLO module, added by daixu.
    Modified from https://github.com/silencelamb/alpa/blob/77c50af55df5b1dc22e7c51c92dc3b4c842eb246/alpa/mesh_profiling.py#L909
    """
    with XlaPassContext({
            "gpu_cost_model::num_micro_batches": num_micro_batches,
            "gpu_cost_model::grad_sync_channel_ids": grad_sync_channel_ids,
    }):
        return xe.hlo_module_cost_analysis(client, hlo_module)


def remove_all(arr: Sequence[Any], elms: Sequence[Any]):
    "Remove all elms from arr."
    for elm in elms:
        while elm in arr:
            arr.remove(elm)
    return arr


def is_power_of(base: int, target: int):
    """ Judge whether target is a power of base. """
    if target == 1:
        return True
    assert base > 1
    _v = base
    while _v <= target:
        if _v == target:
            return True
        _v = _v * base
    return False 
       

def save_as_json(json_path, json_list):
    """ Json list should be the format as: [{...}, ] """
    with open(json_path, "w") as f:
        json.dump(json_list, f)


def read_json_content(json_path):
    """ Json list should be the format as: [{...}, ] """
    with open(json_path, "r", encoding='utf-8') as f:
        json_content = json.load(fp=f)
    return json_content


def load_device_info_table():
    """ Load the device info table offline profiled by ./cpp/src/inspect_hardware. """
    # `peak_gflops_per_second` is measured as the computation capability of single-percision 
    # float-point (f32).
    device_info_table = dict()  # gpu_type -> device_info
    records = read_json_content(os.environ.get("DEVICE_INFO_PATH"))
    for _rec in records:
        device_info_table[_rec["gpu_type"]] = _rec
    return device_info_table


def translate_to_device_properties(device_info: dict):
    """ Translate device_info to the format of DeviceProperties defined in cpp backend. """
    return {
        "name": device_info["gpu_type"],
        "compute_major": device_info["compute_major"],
        "compute_minor": device_info["compute_minor"],
        "max_threads_per_block": device_info["max_threads_per_block"],
        "max_threads_per_multiprocessor": device_info["max_threads_per_multiprocessor"],
        "regs_per_block": device_info["regs_per_block"],
        "regs_per_multiprocessor": device_info["regs_per_multiprocessor"],
        "warp_size": device_info["warp_size"],
        "shared_mem_per_block": device_info["shared_mem_per_block"],
        "shared_mem_per_multiprocessor": device_info["shared_mem_per_multiprocessor"],
        "num_sms": device_info["num_sms"],
        "shared_mem_per_block_optin": device_info["shared_mem_per_block_optin"],
        "mem_bandwidth_gb": device_info["mem_bandwidth_gb"],
        "base_clock_mhz": device_info["base_clock_mhz"],
        "peak_gflops_per_second": device_info["peak_gflops_per_second"]
    }


def unique_append(queue: list, item: Any):
    """ Append item to the queue if previously not existed. """
    if item not in queue:
        queue.append(item)
    return queue


def find_dict_key_based_on_value(table: dict, val: Any):
    """ Find the key of the target value. """
    for _key in table.keys():
        if table[_key] == val:
            return _key
    return None

def gen_hashkey_with_model_configs(
    model_name: str,
    param_num: str,
    batch_size: int,
    num_micro_batches: int,
    gpu_type: str,
    num_hosts: int,
    num_devices_per_host: int,
    num_stages: int = None,
    ignore_num_stages: bool = False
) -> str:
    """
    Generate hash key with the given model configurations to store tuning database.

    The format is: 

        f"{model_name}__{param_num}::{batch_size}__{num_micro_batches}::{num_stages}::" + 
        f"{gpu_type}__{num_hosts}__{num_devices_per_host}" (if `ignore_num_stages` is `False`)

        f"{model_name}__{param_num}::{batch_size}__{num_micro_batches}::" + 
        f"{gpu_type}__{num_hosts}__{num_devices_per_host}" (if `ignore_num_stages` is `True`)
    """
    
    if ignore_num_stages:
        return f"{model_name}__{param_num}::{batch_size}__{num_micro_batches}::" + \
               f"{gpu_type}__{num_hosts}__{num_devices_per_host}"    

    return f"{model_name}__{param_num}::{batch_size}__{num_micro_batches}::{num_stages}::" + \
           f"{gpu_type}__{num_hosts}__{num_devices_per_host}"


def init_tuning_database():
    """ Initialize a tuning database object. """
    return {"plan_set": {}, "selected_cell_num_stages": {}}


def load_tuning_database():
    """ Load the global tuning database. """

    tuning_db_pth = os.environ.get("TUNING_DB_PATH", None)
    tuning_db_filename = os.environ.get("TUNING_DB_FILENAME", None)
    assert (tuning_db_pth is not None and tuning_db_filename is not None), \
        f"Path of tuning database '{tuning_db_pth}' or filename '{tuning_db_filename}' is not set."
    if not os.path.exists(tuning_db_pth):
        os.mkdir(tuning_db_pth)

    pth = os.path.join(tuning_db_pth, tuning_db_filename)
    if pth and os.path.exists(pth):
        print(f"[TMP] Existed tuning database in `{pth}`, updating/rewriting it...")
        try:
            if os.path.getsize(pth) > 0:
                with open(pth, "rb") as f:
                    tuning_database = pickle.load(f)
            else:
                tuning_database = init_tuning_database()
        except EOFError:
            assert os.path.getsize(pth) == 0, \
                "EOF error should be triggered by empty pickle file."
            tuning_database = init_tuning_database()
    else:
        print(f"[TMP] Tuning database is not found in `{pth}`, creating it...")
        tuning_database = init_tuning_database()

    return tuning_database


def store_tuning_database(tuning_database):
    """ Store the global tuning database. """

    tuning_db_pth = os.environ.get("TUNING_DB_PATH", None)
    tuning_db_filename = os.environ.get("TUNING_DB_FILENAME", None)
    assert (tuning_db_pth is not None and tuning_db_filename is not None), \
        f"Path of tuning database '{tuning_db_pth}' or filename '{tuning_db_filename}' is not set."
    pth = os.path.join(tuning_db_pth, tuning_db_filename)

    print(f"[TMP] Storing tuning database to '{pth}'...")
    with open(pth, "wb") as f:
        pickle.dump(tuning_database, f)
