#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to the runtime trace re-generator to randomly generate GPU type on the 
basis of elasticflow trace and filter unnecessary items.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
import random
from typing import Sequence, Any

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_querier import (
    DatabaseQuerier, QueryConfigs)
from resources.hardware_specs import (
    supported_gpu_type, supported_model_name, supported_model_cfgs, NODE_CAPACITY)
from utils import read_csv_file, write_csv_file
from macro.macro_def import (
    ITER_NUM_SCALE_FACTOR, RUNTIME_MAX_ITER_NUM, ITER_NUM_SCALE_FACTOR_SIM)

# Args 
parser = argparse.ArgumentParser()
parser.add_argument("--dummy_test", default=False, action="store_true", 
                    help="Generate runtime trace for test.")
parser.add_argument("--analyze_raw_trace", default=False, action="store_true", 
                    help="Analyze raw trace.")
parser.add_argument("--raw_trace_path", default="none", type=str)
args = parser.parse_args()

"""
Supported GPU num.
"""
supported_gpu_num = [1, 2, 4, 8, 16]
# Probability proportion for model name
prob_list_model_name = [20, 50, 100]
# Probability proportion for param num
# prob_list_param_num = [25, 45, 60, 70, 75, 80]
prob_list_param_num = [20, 40, 60, 75, 85, 90]
# Probability proportion for gpu type
prob_list_gpu_type = [10, 25]
# Probability proportion for gpu num
prob_list_gpu_num = [10, 25, 45, 70, 80]
# Probability proportion for batch size
prob_list_batch_size = [50, 55, 56]


def parse_trace_entry(rec: Sequence[Any]):
    """ Parse the entry in elasticflow philly trace. """
    return (rec[0], int(rec[1]), int(rec[2]), rec[3], 
            int(rec[4]), int(rec[5]), int(rec[6]))


def get_best_locality(gpu_num: int, gpu_type: str, prev_locality: Sequence[int] = None, 
                      local_mgrt_num: int = None):
    """ 
    Get the best locality with the constraint of maximal migration num from 
    the previous locality. 
    """
    if (prev_locality is None) or (prev_locality[0] * (local_mgrt_num + 1) 
                                   >= min(gpu_num, NODE_CAPACITY[gpu_type])):
        return [gpu_num] if gpu_num <= NODE_CAPACITY[gpu_type] \
                else [NODE_CAPACITY[gpu_type] for _ in range(gpu_num // NODE_CAPACITY[gpu_type])]
    else:
        pn_quota = prev_locality[0] * (local_mgrt_num + 1)
        return [pn_quota for _ in range(int(gpu_num / pn_quota))]


def get_supported_gpu_type():
    """ Filter supported GPU type. """
    _list = list()
    for _type in supported_gpu_type:
        if  "-r" not in _type:
            continue
        _list.append(_type.split("-r")[0])
    return _list


def load_and_regen_trace(file_path: str = "./traces/runtime_trace_raw.csv", 
                         target_file_path: str = "./traces/dummy_runtime_trace.csv"): 
    """ 
    Read orginal csv file of elasticflow philly trace, randomly generate GPU 
    type and export trace file. 
    """
    # Querier
    querier = DatabaseQuerier()

    # Filter supported GPU type
    _supported_gpu_type = get_supported_gpu_type()

    # Since the workload with large models is much heavier than the original worklaod
    # in our selected traces, we sample the job arrival events while retaining the 
    # job arrival pattern in the production traces.
    sample_size = 1
    
    data = list()
    data_list = read_csv_file(file_path, style="iterate_row_to_list")
    for _i, _rec in enumerate(data_list):
        if _i % sample_size != 0:
            # Not sampled
            continue
        
        # Parse entry
        (_job_id, _sub_time, _iter_num, _model_name, 
         _deadline, _int_batch_size, _gpu_num) = parse_trace_entry(_rec)
        
        # Overwrite model name and batch size
        _gpu_type = None
        is_feasible = False
        while not is_feasible:
            # Model name
            _model_name = None
            _model_idx = random.randint(0, prob_list_model_name[-1] - 1)
            for _i, _threshold in enumerate(prob_list_model_name):
                if _model_idx < _threshold:
                    _model_name = supported_model_name[_i]
                    break
            assert _model_name is not None
            # Batch size
            _batch_size = None
            _bs_idx = random.randint(0, prob_list_batch_size[-1] - 1)
            for _i, _threshold in enumerate(prob_list_batch_size):
                if _bs_idx < _threshold:
                    _batch_size = supported_model_cfgs[_model_name]["batch_size"][_i]
                    break
            assert _batch_size is not None
            # Param num
            _param_num = None
            _param_idx = random.randint(0, prob_list_param_num[-1] - 1)
            for _i, _threshold in enumerate(prob_list_param_num):
                if _param_idx < _threshold:
                    _param_num = supported_model_cfgs[_model_name]["param_num"][_i]
                    break
            assert _param_num is not None
            # GPU num
            _gpu_num = None
            _gpu_idx = random.randint(0, prob_list_gpu_num[-1] - 1)
            for _i, _threshold in enumerate(prob_list_gpu_num):
                if _gpu_idx < _threshold:
                    _gpu_num = supported_gpu_num[_i]
                    break
            assert _gpu_num is not None
            # GPU type
            _gpu_type = None
            _type_idx = random.randint(0, prob_list_gpu_type[-1] - 1)
            for _i, _threshold in enumerate(prob_list_gpu_type):
                if _type_idx < _threshold:
                    _gpu_type = _supported_gpu_type[_i]
                    break
            assert _gpu_type is not None
            _locality = get_best_locality(_gpu_num, _gpu_type)
            # Feasible if the optimal paralllelism can be deployed
            is_feasible, _, _ = querier.query_db(
                QueryConfigs(len(_locality), _locality[0], _gpu_type, _model_name, _param_num, 
                            _batch_size, only_opt=True)
            )

        # Scale iter num = _init_iter_num * _init_batch_size / (_new_batch_size * scale_factor)
        _iter_num = min(
            int((_iter_num * _int_batch_size) / (_batch_size * ITER_NUM_SCALE_FACTOR)), 
            RUNTIME_MAX_ITER_NUM
        )
        _model_name = _model_name + "__" + _param_num
        # Entry
        _entry = {
            "job_id": _job_id, 
            "submission_time": _sub_time,
            "num_iteration": _iter_num,
            "model_name": _model_name,
            "deadline": _deadline,
            "batch_size": _batch_size,
            "num_gpu": _gpu_num,
            "gpu_type": _gpu_type,
        }
        data.append(_entry)
    # Regenerate trace file
    write_csv_file(target_file_path, data)


def analyze_raw_trace(trace_pth: str):
    """ Analyze job distribution of the target raw trace. """
    data_list = read_csv_file(trace_pth, style="iterate_row_to_list")
    time_interval = 3600
    sub_times = [_rec[1] for _rec in data_list]
    hist_groups = list()
    
    idx, cur_cnt, cur_timestamp = 0, 0, sub_times[0]
    while idx < len(sub_times):
        while idx < len(sub_times) and sub_times[idx] <= cur_timestamp + time_interval:
            cur_cnt += 1
            idx += 1
        hist_groups.append(cur_cnt)
        cur_cnt = 0
        cur_timestamp += time_interval

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(hist_groups)), hist_groups)
    plt.title("Distribution of Job Submission Time")
    plt.xlabel("Hour")
    plt.ylabel("Job amount")

    _file_name = trace_pth.split("/")[-1]
    file_path = f"./figures/job_distribution_runtime.pdf" 
    plt.savefig(file_path, bbox_inches='tight')


if __name__ == "__main__":
    if args.analyze_raw_trace:
        analyze_raw_trace(args.raw_trace_path)
    else:
        _target_pth = "./traces/runtime_trace.csv" if not args.dummy_test \
                        else "./traces/dummy_runtime_trace.csv"
        load_and_regen_trace(args.raw_trace_path, _target_pth)
