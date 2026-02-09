#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to common util functions.
"""

import os
import orjson
import pickle
import json
from dataclasses import (dataclass, field)
import pandas as pd
import numpy as np
from typing import (
    Sequence, Tuple, Optional, Any)
from collections import namedtuple

from resources.hardware_specs import gpu_specs_suite, NODE_CAPACITY, gpu_capacity_rank
from macro.macro_def import MAX_SHRINK_TIMES, INFEASIBLE_THR, PREC


"""
Colloection of the entries in job arrival event.
"""
ArrivalEvent = namedtuple('ArrivalEvent', [
    'type', 'job_id', 'job_alias', 'sub_time', 'iter_num', 'model_name', 'deadline', 'batch_size', 'job_gpu_type', 'job_gpu_num',
])

"""
Colloection of the entries in job departure event.
"""
DepartureEvent = namedtuple('DepartureEvent', [
    'type', 'job_alias',
])


def is_power_of(base: int, target: int):
    """ Judge whether target is a power of base. """
    assert base == 2
    if target <= 4096:
        return target in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        _v = base
        while _v <= target:
            if _v == target:
                return True
            _v = _v * base
        return False 


def get_int_upper_bound(target: int):
    """ 
    Given an integer, return an integer that satisfies: 
    (1) All zero except for the highest digit;
    (2) The minimal that satisfies (1).
    """
    highest_digit = int(str(target)[0]) + 1
    return int(str(highest_digit) + "0" * (len(str(target)) - 1))


def path_join_and_check(base_dir: str, dir: str):
    """ Join path, check whether exists and mkdir if not. """
    pth = os.path.join(base_dir, dir)
    mkdir_if_not_exist(pth)
    return pth
    

def search_list(target_list: Sequence[Any], entry_target: str, entry: str):
    """ Search for the target item with specified entry in the list and return. """
    for _item in target_list:
        if _item[entry] == entry_target:
            return _item
    return None


def search_list_with_uuid(target_list: Sequence[Any], uuid: str):
    """ Search for the target item with specified uuid in the list and return. """
    for _item in target_list:
        assert hasattr(_item, 'uuid')
        if _item.uuid == uuid:
            return _item
    return None


def dict_counter_add(target_dict: dict, entry: str, add_amt: int):
    """ Add the counter of the entry in target dict. """
    if entry not in target_dict.keys():
        target_dict[entry] = add_amt
    else:
        target_dict[entry] += add_amt
    return target_dict


def unique_append(queue: list, item: Any):
    """ Append item to the queue if previously not existed. """
    if item not in queue:
        queue.append(item)
    return queue

def create_entry_and_append_queue(target_dict: dict, entry: str, item: Any):
    """ Create entry in the dict if not existed and append the item into the queue under this entry. """
    if entry not in target_dict.keys():
        target_dict[entry] = [item]
    else:
        target_dict[entry].append(item)
    return target_dict


def replace_all(target: str, item_list: Sequence[Any], target_item: Any):
    """ Place all items into target_item in the target string. """
    for _item in item_list:
        target = target.replace(_item, target_item)
    return target


def remove_all(arr: Sequence[Any], elms: Sequence[Any]):
    "Remove all elms from arr."
    for elm in elms:
        while elm in arr:
            arr.remove(elm)
    return arr


def remove_if_exist(queue: list, item_list: Sequence[Any]):
    """ Remove the item from the queue if exists. """
    for _item in item_list:
        if _item in queue:
            queue.remove(_item)
    return queue


def read_csv_file(file_path: str, style: str = 'iterate_row_to_list', remove_nan: bool = False):
    """ Common tool to read and parse csv file. """
    # Read csv
    data = pd.read_csv(filepath_or_buffer=file_path)
    if remove_nan:
        data.fillna(0, inplace=True)

    if style == 'iterate_row_to_list':
        _list = list()
        for _row_idx in data.index:
            _rec = [data.iloc[_row_idx][i] for i in range(len(data.iloc[_row_idx]))]
            _list.append(_rec)
        return _list


def write_csv_file(file_path: str, data: Sequence[dict]):
    """ 
    Common tool to write into a csv file. 
    --------------------------------------------
    Data format: [ {'entry_1': xxx, 'entry_2': xxx, ...}, ... ]
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=True)


"""
Customized deepcopy function.
Ref: https://stackoverflow.com/questions/45858084/what-is-a-fast-pythonic-way-to-deepcopy-just-data-from-a-python-dict-or-list
"""

_dispatcher = {}

def _copy_list(l, dispatch):
    ret = l.copy()
    for idx, item in enumerate(ret):
        cp = dispatch.get(type(item))
        if cp is not None:
            ret[idx] = cp(item, dispatch)
    return ret

def _copy_dict(d, dispatch):
    ret = d.copy()
    for key, value in ret.items():
        cp = dispatch.get(type(value))
        if cp is not None:
            ret[key] = cp(value, dispatch)

    return ret

_dispatcher[list] = _copy_list
_dispatcher[dict] = _copy_dict

def deepcopy_v1(sth):
    cp = _dispatcher.get(type(sth))
    if cp is None:
        return sth
    else:
        return cp(sth, _dispatcher)
    

def deepcopy(obj):
    """ Implement deepcopy with serialization and deserialization. """
    return orjson.loads(orjson.dumps(obj))


def deepcopy_pickle(obj):
    """ Implement deepcopy with pickle for unserializable objects. """
    return pickle.loads(pickle.dumps(obj))


# def deepcopy(obj):
#     return pickle.loads(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def save_as_json(json_path, json_list):
    """ Json list should be the format as: [{...}, ] """
    with open(json_path, "w") as f:
        json.dump(json_list, f)


def read_json_content(json_path):
    """ Json list should be the format as: [{...}, ] """
    with open(json_path, "r", encoding='utf-8') as f:
        json_content = json.load(fp=f)
    return json_content

def mkdir_if_not_exist(path: str):
    """ Mkdir if the path is not existed. """
    if not os.path.exists(path):
        os.mkdir(path)

"""
Read profiling data.
"""

def _get_parallel_method(forward_stage_layer_ids=None, mesh_shapes=None, logical_mesh_shapes=None, autosharding_option_dicts=None):
    """ Get the parallel method in a string manner. """
    method = []
    # If no sol, return
    if logical_mesh_shapes == 'no_sol':
        return 'na'

    # Transfer logical_mesh_shapes (str) into list
    _tmp_list = logical_mesh_shapes.replace('[', '').replace(']', '').split(', (')
    _tmp_list_2 = autosharding_option_dicts.replace('[', '').replace(']', '').split(',')
    if len(_tmp_list) > 1:
        method.append('pipe')
    for _idx in range(len(_tmp_list)):
        _item = _tmp_list[_idx]
        _flag = _tmp_list_2[_idx]
        _pair = _item.replace('(', '').replace(')', '').split(', ')
        if int(_pair[0]) > 1 and _flag.replace(' ', '') != '{}' and 'dp' not in method:
            method.append('dp')
        if int(_pair[1]) > 1 and 'mp' not in method:
            method.append('mp')
        # When the _flag is {} and _pair[0] > 1, the ILP solver will determine to perform DP or MP.
        if int(_pair[0]) > 1 and _flag.replace(' ', '') == '{}' and 'hp' not in method:
            method.append('hp')
        
    if len(method) == 0:
        # Single GPU case
        return 's'

    # Concate
    ret = method[0][0]
    for i in range(1, len(method), 1):
        ret = ret + method[i][0]

    return ret


def get_throughput(avg_e2e_iter_time, global_batch_size):
    """ Legacy: Get the training throughput. """
    return round(global_batch_size / float(avg_e2e_iter_time), PREC) if avg_e2e_iter_time != 'no_sol' else 0.0


def get_thr(avg_e2e_iter_time: float, batch_size: int):
    """ Get training throughput. """
    return round(batch_size / float(avg_e2e_iter_time), PREC) if avg_e2e_iter_time else 0.0 


def get_profile_data(file_path):
    """ 
    Read csv file and formulate data input based on profiling data.
    ------------------------------------------------------------------------
    Return:
        - prof_data = {
            'model_name': {
                'device_name': {
                    'param_num': {
                        'batch_size': { 
                            'parallel_method': '', 
                            'throughput': x.x,
                        },
                    }, 
                }, 
            }, 
        } 
    """
    prof_data = {}

    # Read csv
    data = pd.read_csv(filepath_or_buffer=file_path, skip_blank_lines=True)
    for row_idx in data.index:
        # First, check GPU type
        _gpu_type =str(data.iloc[row_idx]['gpu_type'])
        if _gpu_type != 'nan':
            # Then, check other entries
            _node_num = str(int(data.iloc[row_idx]['node_num']))
            _gpu_num_per_node = str(int(data.iloc[row_idx]['gpu_num_per_node']))
            _device_name = _node_num + '_' + _gpu_type + '_' + _gpu_num_per_node + '_d'
            _model_name = str(data.iloc[row_idx]['model_name'])
            _param_num = str(data.iloc[row_idx]['param_num'])
            # _batch_size = str(int(data.iloc[row_idx]['batch_size'] / data.iloc[row_idx]['micro_batch_num']))
            _batch_size = str(int(data.iloc[row_idx]['batch_size']))
            # At last, check performance metrics
            if str(data.iloc[row_idx]['forward_stage_layer_ids']) != 'nan':
                _parallel_method = _get_parallel_method(logical_mesh_shapes=data.iloc[row_idx]['logical_mesh_shapes'], 
                                                        autosharding_option_dicts=data.iloc[row_idx]['autosharding_option_dicts'])
                _thr = get_throughput(avg_e2e_iter_time=data.iloc[row_idx]['average_e2e_iteration_time'], global_batch_size=int(_batch_size))
                _alloc_mem = round(float(data.iloc[row_idx]['max_allocated_memory_among_devices']), PREC) if data.iloc[row_idx]['max_allocated_memory_among_devices'] != 'no_sol' else 0.0
                # Insert
                if _model_name not in prof_data.keys():
                    prof_data[_model_name] = {}
                if _device_name not in prof_data[_model_name].keys():
                    prof_data[_model_name][_device_name] = {}
                if _param_num not in prof_data[_model_name][_device_name].keys():
                    prof_data[_model_name][_device_name][_param_num] = {}
                if _batch_size not in prof_data[_model_name][_device_name][_param_num].keys():
                        prof_data[_model_name][_device_name][_param_num][_batch_size] = {}
                prof_data[_model_name][_device_name][_param_num][_batch_size]['parallel_method'] = _parallel_method
                prof_data[_model_name][_device_name][_param_num][_batch_size]['throughput'] = _thr
                prof_data[_model_name][_device_name][_param_num][_batch_size]['memory'] = _alloc_mem
    
    return prof_data


def get_dummy_delta_thr(target_job: Any, new_gpu_type: str, new_gpu_num: int, new_locality: list, \
                                            prev_gpu_type: str = None, prev_gpu_num: int = None, prev_locality: Sequence[int] = None, 
                                            job_init_gpu_num_table: dict = None):
    """ Get the dummy structured marginal gain based on intuitive construction. """
    # Limit the shrink times
    if new_gpu_num * np.exp(MAX_SHRINK_TIMES) < job_init_gpu_num_table[target_job.uuid]:
        return INFEASIBLE_THR
    
    # Check whether is new job
    if prev_gpu_type is None:
        # New job
        _type_rank = gpu_capacity_rank.index(new_gpu_type) + 1
        return _type_rank * 100 + new_gpu_num * 10 + len(new_locality) * (-5)

    # Fake gain
    _type_rank_1 = gpu_capacity_rank.index(prev_gpu_type)
    _type_rank_2 = gpu_capacity_rank.index(new_gpu_type)
    _fake_gain = ((_type_rank_2 - _type_rank_1) * 100 + round(100 / (new_gpu_num - prev_gpu_num), PREC) + (len(prev_locality) - len(new_locality)) * 5) if new_gpu_num != prev_gpu_num \
                 else (((_type_rank_2 - _type_rank_1) * 100 + (len(prev_locality) - len(new_locality)) * 5) / prev_gpu_num)
    return _fake_gain


def translate_namedtuple_to_dict(obj: namedtuple):
    """ Translate namedtuple into dict format. """
    _dict = dict()
    for _entry in obj._fields:
        _dict[_entry] = str(getattr(obj, _entry))
    return _dict
