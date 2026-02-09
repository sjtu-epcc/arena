#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Hardware specs & mapping of multi-hierarchy heterogeneous GPU cluster.
-----------------------------------------------------------------------------------------------
"""

import os
from collections import namedtuple


"""
Supported GPU type.
"""
supported_gpu_type = [ "a100", "v100", "3090", "p100", "a40", "a10", "1080ti", 
                       "a40-r", "a10-r" ]
if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
    supported_gpu_type = ["h100", "l20"]


"""
GPU capacity rank.
"""
gpu_capacity_rank = [ "1080ti", "p100", "a10", "a40", "3090", "v100", "a100", 
                     "1080ti-2r", "1080ti-r", "p100-r", "a40-r", "3090-r", "a100-r" ]


"""
Node capacity indexed by GPU type.
"""
NODE_CAPACITY = {
    # For simulation
    "a100": 4,
    "v100": 16,
    "p100": 8,
    "a40": 2,
    "a10": 2,
    "1080ti": 8,
    "3090": 8,
    "h100": 8,
    "l20": 16,
    # For real-world orchestration
    "1080ti-r": 4,
    "1080ti-2r": 2,
    "p100-r": 4,
    "a40-r": 2,
    "3090-r": 4,
    "a100-r": 1,
}

"""
GPU type that forbidden cross-nodes placement due to the network legacy.
"""
forbid_cross_nodes_gpu_type = ["3090-r"]


"""
Support model name and related param & batch size.
--------------------------------------------------------
Suggested by Alpa doc: https://github.com/alpa-projects/alpa/tree/main/benchmark/alpa
"""
supported_model_name = ["wide_resnet", "bert", "moe"]
supported_model_cfgs = {
    "wide_resnet": {
        "param_num": ["500M", "1B", "2B", "4B", "6.8B", "13B"],
        "batch_size": [256, 512, 1024],
        "layer_num": 16,
    },
    "bert": {
        "param_num": ["760M", "1.3B", "2.6B", "6.7B", "15B", "39B"],
        "batch_size": [128, 256, 512],
        "layer_num": 6,
    },
    "moe": {
        "param_num": ["690M", "1.3B", "2.4B", "10B", "27B", "70B"],
        "batch_size": [256, 512, 1024],
        "layer_num": 8,
    },
}


"""
Ports option of Ray cluster.
"""
ray_port_option_list = [
                            {
                                "port": 6379, "object_manager_port": 6380, "node_manager_port": 6381, 
                                "ray_client_server_port": 10001, "min_worker_port": 10002, "max_worker_port": 11001,
                            },
                            {
                                "port": 6389, "object_manager_port": 6390, "node_manager_port": 6391, 
                                "ray_client_server_port": 11002, "min_worker_port": 11003, "max_worker_port": 12002,
                            },
                            {
                                "port": 6399, "object_manager_port": 6400, "node_manager_port": 6401, 
                                "ray_client_server_port": 12003, "min_worker_port": 12004, "max_worker_port": 13003,
                            },
                            {
                                "port": 6409, "object_manager_port": 6410, "node_manager_port": 6411, 
                                "ray_client_server_port": 13004, "min_worker_port": 13005, "max_worker_port": 14004,
                            },
                            {
                                "port": 6419, "object_manager_port": 6420, "node_manager_port": 6421, 
                                "ray_client_server_port": 14005, "min_worker_port": 14006, "max_worker_port": 15005,
                            },
                            {
                                "port": 6429, "object_manager_port": 6430, "node_manager_port": 6431, 
                                "ray_client_server_port": 15006, "min_worker_port": 15007, "max_worker_port": 16006,
                            },
                            {
                                "port": 6439, "object_manager_port": 6440, "node_manager_port": 6441, 
                                "ray_client_server_port": 16007, "min_worker_port": 16008, "max_worker_port": 17007,
                            },
                            {
                                "port": 6449, "object_manager_port": 6450, "node_manager_port": 6451, 
                                "ray_client_server_port": 17008, "min_worker_port": 17009, "max_worker_port": 18008,
                            },
                       ]


NodeSpec = namedtuple("NodeSpec", [
    "node_id", "node_alias", "node_type",
])

GPUSpec = namedtuple("GPUSpec", [
    "gpu_id", "gpu_alias", "gpu_type", 
    "max_mem", "max_bw", "sm_sum", 
])

"""
Pre-alloc memory fraction for XLA.
"""
pre_alloc_memory_fraction_for_xla = 0.8

"""
Hardware specs.
"""
# GPU specs (sm <- # of cuda cores & # of tensor cores)
gpu_specs_suite = {
    # A40
    "a40": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="a40", max_mem=48.0, max_bw=696.0, sm_sum="10752_c_336_t"),
    # A10
    "a10": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="a10", max_mem=24.0, max_bw=696.0, sm_sum="10752_c_336_t"),
    # A100 PCIe version
    "a100": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="a100", max_mem=40.0, max_bw=1935.0, sm_sum="6912_c_432_t"),
    # V100 32GB PCIe version (also have NVLink version)
    "v100": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="v100", max_mem=32.0, max_bw=1134.0, sm_sum="5120_c_640_t"),
    # P100 12GB PCIe version (also have NVLink version)
    "p100": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="p100", max_mem=12.0, max_bw=549.0, sm_sum="3584_c_0_t"),
    # 1008ti 
    "1080ti": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="1080ti", max_mem=11.0, max_bw=484.0, sm_sum="3584_c_0_t"),
    # 3090
    "3090": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="3090-r", max_mem=24.0, max_bw=484.0, sm_sum="none"),
    # H100
    "h100": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="h100", max_mem=80.0, max_bw=3000.0, sm_sum="none"),
    # L20
    "l20": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="l20", max_mem=48.0, max_bw=2000.0, sm_sum="none"),
    
    # A100 PCIe version real-world
    "a100-r": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="a100", max_mem=40.0, max_bw=1935.0, sm_sum="6912_c_432_t"),
    # A40 real-world
    "a40-r": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="a40-r", max_mem=48.0, max_bw=696.0, sm_sum="10752_c_336_t"),
    # P100 12GB PCIe version (also have NVLink version) real-world
    "p100-r": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="p100-r", max_mem=12.0, max_bw=549.0, sm_sum="3584_c_0_t"),
    # 1008ti real-world
    "1080ti-r": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="1080ti-r", max_mem=11.0, max_bw=484.0, sm_sum="3584_c_0_t"),
    # 1008ti real-world
    "1080ti-2r": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="1080ti-2r", max_mem=11.0, max_bw=484.0, sm_sum="3584_c_0_t"),
    # 3090 real-world
    "3090-r": GPUSpec(gpu_id="none", gpu_alias="none", gpu_type="3090-r", max_mem=24.0, max_bw=484.0, sm_sum="none"),
}
