#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
A script related to parse and profile (estimate) the HLO computation performance 
without actual cross-devices communication. 
"""

import time
import pickle
from typing import (
    Sequence, Optional, Any, List,
)
from deprecated import deprecated
import numpy as np
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from alpa.wrapped_hlo import (WrappedHlo, HloStatus)

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cpp backend package
import jaxpr.crius_cupti as cc
# Python scripts
from jaxpr.hlo_ops import *
from jaxpr.transform_hlo import (
    shard_one_hlo_module, compile_one_sharded_hlo_module)
from jaxpr.communication import (
    get_crossed_host_num_of_replica_groups, 
    generate_normalized_replica_groups,
    enumerate_all_replica_groups
)
from jaxpr.utils import (
    remove_all, translate_to_device_properties, load_device_info_table, 
    NUMPY_DTYPE_TABLE, MAX_COMM_SIZE, CAND_GPU_TYPES, NS_TO_S, 
    LARGE_COMM_SCALE_FACTOR, DTYPE_TO_F32_RATIO_TABLE, Computation)

# Global operator database
global_op_db = dict()
# Global parameter number
global_param_number = 0
# Global fused XLA operator database
global_fused_xla_op_db = dict()
# Global device info table
device_info_table = None


def _slice_src_op_bias_list(_str: str):
    """ Slice source ops. """
    if "(" in _str and ")" in _str:
        # Single src op
        return _str.split("(")[1].split(")")[0]
    elif "(" in _str:
        return _str.split("(")[1].split(",")[0]
    elif ")" in _str:
        return _str.split(")")[0]
    else:
        return _str.split(",")[0]
    

def _slice_dst_data_types(_str: str):
    """ Slice destinated data types. """
    if "/*index=" in _str:
        _str = _str.split("*/")[1]
    if "(" in _str:
        return NUMPY_DTYPE_TABLE[_str.replace("((", "(").split("(")[1].split("[")[0]]
    elif ")" in _str:
        return NUMPY_DTYPE_TABLE[_str.split("[")[0]]
    else:
        return NUMPY_DTYPE_TABLE[_str.split("[")[0]]


def _slice_dst_shapes(_str: str):
    """ Slice destinated data shapes. """
    _s = remove_all(_str.split("[")[1].split("]")[0].split(","), [""])
    return [int(_c) for _c in _s] if len(_s) > 0 else []


def _estimate_comm_op_time(op_type: str, data_shape: Sequence[int], dtype: Any, 
                           replica_groups: Sequence[Sequence[int]], comm_time_table: dict, 
                           is_recursive: bool = False):
    """ Estimate the time of the communication operator based on offline profiling results. """
    # Querying key
    key = str((op_type, replica_groups)) if op_type != "send" else "send"
    # Estimate comm time
    comm_size = np.prod(data_shape)
    last_comm_size, last_comm_time = -1, -1
    dtype = str(dtype)

    if dtype in DTYPE_TO_F32_RATIO_TABLE:
        # Resize comm_size based on dtype
        comm_size *= DTYPE_TO_F32_RATIO_TABLE[dtype]
        dtype = "float32"

    if comm_size <= np.prod(comm_time_table[key][0][0]):
        return comm_time_table[key][0][2]
    
    for _i, (_shape, _dtype, _comm_time, _replica_groups) in enumerate(comm_time_table[key]):
        assert dtype == "float32", f"Currently only support data type with float32, got {dtype}."
        assert _replica_groups == replica_groups, f"Mismatched replica groups in record ({_replica_groups}) " + \
                                                  f"and in key ({replica_groups})."
        _comm_size = np.prod(_shape)
        if _comm_size >= comm_size:
            # Linear interpolate
            _ratio = (comm_size - last_comm_size) / (_comm_size - last_comm_size)
            return last_comm_time + (_comm_time - last_comm_time) * _ratio
        last_comm_size, last_comm_time = _comm_size, _comm_time

    # Exceed max profiled comm size
    max_profiled_comm_size = np.prod(comm_time_table[key][-1][0])
    # Decompose communication size into profiled scope
    max_profiled_cnt = comm_size // max_profiled_comm_size
    rest_comm_time = _estimate_comm_op_time(op_type, (comm_size % max_profiled_comm_size,), 
                                             dtype, replica_groups, comm_time_table, is_recursive=True)
    max_comm_time = comm_time_table[key][-1][2]

    # if True:
    if max_profiled_comm_size < MAX_COMM_SIZE // 2:
        # Only for low bandwidth cases, which linearly scale the comm time
        est_comm_time = max_profiled_cnt * max_comm_time + rest_comm_time
    else:    
        # When comm amount is extremely large, comm time scales a little faster than linearly scaling 
        # probably due to network interference.
        est_comm_time = (max_profiled_cnt * max_comm_time + rest_comm_time) \
                         * pow(LARGE_COMM_SCALE_FACTOR, max_profiled_cnt)
    
    print(f"[WARN] Communication size of '{op_type}' excceeds the maximum profiled size. Linear scaling is applied.")
    print(f"       - Communication size of the operator: {comm_size} | Estimated comm time: {est_comm_time}")
    print(f"       - Max profiled communication size: {max_profiled_comm_size} | Comm time: {max_comm_time}")
    
    return est_comm_time


def parse_one_hlo_op(line: str):
    """ Parse one HLO operator from the HLO text. """
    global global_op_db, global_param_number
    strs = remove_all(line.split(" "), ["", "\n", "ROOT"])
    if len(strs) == 0:
        return None

    # Get envion
    num_hosts = int(os.environ.get("CRIUS_NUM_HOSTS"))
    num_devices_per_host = int(os.environ.get("CRIUS_NUM_DEVICES_PER_HOST"))
    # Common entries
    bias = strs[0]
    op_type = bias.split("_")[0].split(".")[0]
    if "done" in op_type:
        # `all-reduce-done` appears when setting xla autotunning level to 4.
        return None
    if op_type in operator_bias_table.keys():
        op_type = operator_bias_table[op_type]
    assert op_type in cls_table.keys(), f"Unsupported opeartor type: {op_type}."
    data_type = NUMPY_DTYPE_TABLE[strs[2].split("[")[0].replace("(", "")]
    # Shape (first one if multiple existed)
    _s = strs[2].split("[")[1].split("]")[0]
    shape = [int(_i) for _i in _s.split(",")] if _s != "" else []
    if "{" in strs[2]:
        _s = strs[2].split("{")[1].split("}")[0]
        shape_dim_idx = [int(_i) for _i in _s.split(",")] if _s != "" else []
    else:
        shape_dim_idx = []
    # Entry placeholders
    dst_data_type = None
    src_op_bias, lhs_bias, rhs_bias, pred_bias, direction = None, None, None, None, None
    value = None
    dynamic_slice_sizes, padding_configs, concate_dim, \
        start_indices, limit_indices, slice_strides, reverse_dims = None, None, None, None, None, None, None
    window_sizes, window_dims, window_strides, padding, \
        lhs_dilate, rhs_dilate = None, None, None, None, None, None
    lhs_batch_dims, rhs_batch_dims, lhs_contracting_dims, rhs_contracting_dims = None, None, None, None

    # Operands for compare & binary arithmetic & bitwise
    if op_type in ("compare", 
                   "minimum", "maximum", "add", "subtract", "multiply", "divide", "power",
                   "and", "or"):
        _s = strs[2].split("[")[1].split("]")[0]
        shape = [int(_i) for _i in _s.split(",")] if _s != "" else []
        lhs_bias = strs[3].split("(")[1].split(",")[0]
        rhs_bias = strs[4].split(")")[0]
    # Direction for compare operation
    if op_type == "compare":
        direction = strs[5].split("=")[1].replace("\n", "")
    # Operands for select operation
    if op_type == "select":
        pred_bias = strs[3].split("(")[1].split(",")[0]
        lhs_bias = strs[4].split(",")[0]
        rhs_bias = strs[5].split(")")[0]
    
    # Sharding
    if op_type in ("param", "copy"):
        if "sharding" in line:
            if "replicated" in strs[4]:
                sharding = ShardingSpec(type="replicated", 
                                        device_list=[_i for _i in range(num_hosts * num_devices_per_host)])
            else:
                _s = strs[4].split("]")[1].split("}")[0]
                device_list = [int(_i) for _i in _s.split(",")] if _s != "" else []
                sharding = ShardingSpec(type="sharding", device_list=device_list)
        else:
            sharding = None
    
    # Src bias
    if op_type in ("dynamic-slice", "pad", "slice", "reshape", "reverse",
                   "convert", "transpose", "cosine", "exponential", "log", "negate", "rsqrt", "sqrt",
                   "opt-barrier", "bitcast", "broadcast", "select-and-scatter",
                   "get-tuple-element", "all-reduce"):
        for _i, _str in enumerate(strs):
            if _i > 0 and str(op_type + "(") in _str:
                src_op_bias = _str.split("(")[1].split(")")[0]
                if "," in src_op_bias: 
                    # Multi srcs but use only the first src
                    src_op_bias = src_op_bias.split(",")[0]

    # Src and dst data type
    if op_type in ("convert"):
        _s = strs[2].split("[")[1].split("]")[0]
        shape = [int(_i) for _i in _s.split(",")] if _s != "" else []
        dst_data_type = data_type
    
    # Src and dst shape
    if op_type in ("bitcast", "broadcast"):
        dst_shape, dst_shape_dim_idx = shape, shape_dim_idx
        if src_op_bias in global_op_db.keys() and hasattr(global_op_db[src_op_bias], "shape_dim_idx"):
            src_shape = global_op_db[src_op_bias].shape
            src_shape_dim_idx = global_op_db[src_op_bias].shape_dim_idx
        else:
            src_shape, src_shape_dim_idx = [], []
    
    # Dynamic slice sizes
    if op_type == "dynamic-slice":
        for _str in strs:
            if "dynamic_slice_sizes" in _str:
                _s = _str.split("{")[1].split("}")[0]
                _s = _s.split(",")
                dynamic_slice_sizes = [int(_v) if _v != "" else -1 for _v in _s]
                break
        assert dynamic_slice_sizes is not None, "Dynamic slice sizes is not properly set."
    
    # Padding configs
    if op_type == "pad":
        padding_configs = list()
        for _str in strs:
            if "padding" in _str:
                _s = _str.split("=")[1].split("x")
                for _c in _s:
                    _one_dim = [int(_e) for _e in _c.split("_")]
                    padding_configs.append(tuple(_one_dim)) if len(_one_dim) == 3 \
                        else padding_configs.append(tuple(_one_dim + [0]))

    # Concat src operands and dimensions
    if op_type == "concatenate":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "concatenate" in _str:
                idx_1 = _i
            if "dimensions" in _str:
                idx_2 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Concate dimension
        concate_dim = int(strs[idx_2].split("{")[1].split("}")[0])
    
    # Slice sizes
    if op_type == "slice":
        idx = -1
        for _i, _str in enumerate(strs):
            if "slice={" in _str:
                idx = _i
                break
        start_indices, limit_indices, slice_strides = list(), list(), list()
        for _i in range(idx, len(strs), 1):
            _s = strs[_i].split("[")[1].split("]")[0].split(":")
            start_indices.append(int(_s[0]))
            limit_indices.append(int(_s[1]))
            _stride = int(_s[2]) if len(_s) == 3 else 1
            slice_strides.append(_stride) 
        assert len(start_indices) > 0, "Slice indices is not properly set."
    
    # Reverse dimensions
    if op_type == "reverse":
        for _i, _str in enumerate(strs):
            if _i > 0 and "dimensions" in _str:
                reverse_dims = [int(_v) for _v in _str.split("{")[1].split("}")[0].split(",")]
    
    # Transpose dimensions
    if op_type == "transpose":
        for _i, _str in enumerate(strs):
            if _i > 0 and "dimensions" in _str:
                transpose_dims = [int(_v) for _v in _str.split("{")[1].split("}")[0].split(",")]

    # Tuple elements
    if op_type == "tuple":
        idx_1 = -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "tuple(" in _str:
                idx = _i
        # Src op bias 
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx:len(strs)]]
        # Dst shapes
        dst_shapes = [_slice_dst_shapes(_str) for _str in strs[2:idx]]
    
    # Reduced operands and dimension
    if op_type == "reduce":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "reduce" in _str:
                idx_1 = _i
            elif "dimensions" in _str:
                idx_2 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Data types
        dst_data_types = [_slice_dst_data_types(_str) for _str in strs[2:idx_1]]
        # Src shape
        src_shapes = list()
        for _bias in src_op_bias_list:
            if "constant" not in _bias:
                src_shapes.append(global_op_db[_bias].shape)
        # Reduced dimension
        reduced_dims = [int(_c) for _c in strs[idx_2].split("{")[1].split("}")[0].split(",")]
    
    # Window-reduced operands and window configs
    if op_type == "reduce-window":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "reduce-window(" in _str:
                idx_1 = _i
            elif "window={" in _str:
                idx_2 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Data types
        dst_data_types = [_slice_dst_data_types(_str) for _str in strs[2:idx_1]]
        # Window configs
        for _str in strs[idx_2:]:
            if "size" in _str:
                window_dims = [int(_c) for _c in _str.split("size=")[1].split("}")[0].split("x")]
            elif "stride" in _str:
                window_strides = [int(_c) for _c in _str.split("stride=")[1].split("}")[0].split("x")]
            elif "pad" in _str:
                padding = list()
                _s = _str.split("pad=")[1].split("}")[0].split("x")
                for _c in _s:
                    padding.append(tuple([int(_e) for _e in _c.split("_")]))
            elif "to_apply" in _str:
                break
        assert all(_elm is not None for _elm in (window_dims, window_strides, padding)), \
                "Window configs is not properly set."
    
    # Select-and-scatter operands and window configs
    if op_type == "select-and-scatter":
        idx_1, idx_2, idx_3 = -1, -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "select-and-scatter(" in _str:
                idx_1 = _i
            elif "window={" in _str:
                idx_2 = _i
            elif "select=" in _str:
                idx_3 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Window configs
        for _str in strs[idx_2:]:
            if "size" in _str:
                window_dims = [int(_c) for _c in _str.split("size=")[1].split("}")[0].split("x")]
            elif "stride" in _str:
                window_strides = [int(_c) for _c in _str.split("stride=")[1].split("}")[0].split("x")]
            elif "pad" in _str:
                padding = list()
                _s = _str.split("pad=")[1].split("}")[0].split("}")[0].split("x")
                for _c in _s:
                    padding.append(tuple([int(_e) for _e in _c.split("_")]))
    
    # Broadcast dimension
    if op_type == "broadcast":
        for _str in strs:   
            if "dimensions" in _str:
                _s = remove_all(_str.split("{")[1].split("}")[0].split(","), [""])
                broadcast_dims = [int(_v) for _v in _s] if len(_s) > 0 else []
                break
        assert broadcast_dims is not None, "Broadcast dimension is not properly set."

    # Iota dimension
    if op_type == "iota":
        for _str in strs:   
            if "dimension" in _str:
                _s = _str.split("=")[1]
                iota_dim = int(_s) if _s != "" else -1
                break
        assert iota_dim is not None, "Iota dimension is not properly set."
    
    # Fusion kind
    if op_type == "fusion":
        idx = -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "fusion(" in _str:
                idx = _i
            if "kind" in _str:
                fusion_kind = _str.split("kind=")[1].split(",")[0]
            if "calls=" in _str:
                called_fusion_bias = _str.split("calls=")[1].replace("\n", "")
        # Dst shapes
        dst_shapes = [_slice_dst_shapes(_str) for _str in strs[2:idx]]
        # Data types
        dst_data_types = [_slice_dst_data_types(_str) for _str in strs[2:idx]]

    # GTE index
    if op_type == "get-tuple-element":
        for _str in strs:
            if "index" in _str:
                gte_idx = int(_str.split("index=")[1])
                break
    
    # Clamp
    if op_type == "clamp":
        idx = -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "clamp(" in _str:
                idx = _i
                break
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx:]]

    # Cudnn-conv configs
    if op_type == "cudnn-conv":
        idx_1, idx_2, idx_3 = -1, -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and (("convolution" not in bias and "custom-call(" in _str) 
                           or ("convolution" in bias and "convolution(" in _str)):
                idx_1 = _i
            elif "window={" in _str:
                idx_2 = _i
            elif "dim_labels=" in _str:
                idx_3 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        src_shapes = [global_op_db[_bias].shape for _bias in src_op_bias_list]
        # Window configs
        for _i in range(idx_2, idx_3, 1):
            if "size" in strs[_i]:
                window_sizes = [int(_c) for _c in strs[_i].split("size=")[1].split("}")[0].split("x")]
            elif "pad" in strs[_i]:
                padding = list()
                _s = strs[_i].split("pad=")[1].split("}")[0].split("x")
                for _c in _s:
                    padding.append(tuple([int(_e) for _e in _c.split("_")]))
            elif "stride" in strs[_i]:
                window_strides = [int(_c) for _c in strs[_i].split("stride=")[1].split("}")[0].split("x")]
            elif "lhs_dilate" in strs[_i]:
                lhs_dilate = [int(_c) for _c in strs[_i].split("lhs_dilate=")[1].split("}")[0].split("x")]
            elif "rhs_dilate" in strs[_i]:
                rhs_dilate = [int(_c) for _c in strs[_i].split("rhs_dilate=")[1].split("}")[0].split("x")]
    
    # Cublas-gemm src ops
    if op_type == "cublas-gemm":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and (("dot" not in bias and "custom-call(" in _str) 
                           or ("dot" in bias and "dot(" in _str)):
                idx_1 = _i
            elif ("dot" not in bias and "custom_call_target=" in _str) \
                    or ("dot" in bias and "lhs_contracting_dims=" in _str) \
                    or ("dot" in bias and "lhs_batch_dims=" in _str):
                idx_2 = _i
            if idx_1 > -1 and idx_2 > -1:
                break
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        src_shapes = [global_op_db[_bias].shape for _bias in src_op_bias_list]
        # Contracting dims
        for _str in strs:
            if "lhs_batch_dims" in _str:
                lhs_batch_dims = [int(_v) for _v in _str.split("lhs_batch_dims={")[1].split("}")[0].split(",")]
            if "rhs_batch_dims" in _str:
                rhs_batch_dims = [int(_v) for _v in _str.split("rhs_batch_dims={")[1].split("}")[0].split(",")]
            if "lhs_contracting_dims" in _str:
                lhs_contracting_dims = [int(_v) for _v in remove_all(_str.split("lhs_contracting_dims={")[1].split("}")[0].split(","), [""])]
            if "rhs_contracting_dims" in _str:
                rhs_contracting_dims = [int(_v) for _v in remove_all(_str.split("rhs_contracting_dims={")[1].split("}")[0].split(","), [""])]
    
    # Channel ids and replica groups
    if op_type in ("all-reduce", "all-gather", "all-to-all", "reduce-scatter"):
        for _str in strs:
            if "channel_id" in _str:
                channel_id = int(_str.split("channel_id=")[1].split(",")[0])
            if "replica_groups" in _str:
                _s = _str.split("replica_groups={")[1].replace("},", "").replace("}", "").split("{")
                _tmp = remove_all(_s, ["", "\n"])
                replica_groups = list()
                for _c in _tmp:
                    replica_groups.append([int(_t) for _t in _c.split(",")])

    # All-reduce src ops
    if op_type == "all-reduce":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and ("all-reduce(" in _str or "all-reduce-start(" in _str):
                idx_1 = _i
            elif "channel_id=" in _str:
                idx_2 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        assert len(src_op_bias_list) == 1, \
            f"Allreduce operator can only have one source operator, got {src_op_bias_list}."
    
    # All-gather src ops, dst shapes and dimensions
    if op_type == "all-gather":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "all-gather" in _str:
                idx_1 = _i
            elif "channel_id=" in _str:
                idx_2 = _i
            elif "dimensions" in _str:
                all_gather_dimension = int(_str.split("dimensions={")[1].split("}")[0])
                break
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Dst shapes
        dst_shapes = [_slice_dst_shapes(_str) for _str in strs[2:idx_1]]
        # Src data types
        src_data_types = [global_op_db[_bias].data_type for _bias in src_op_bias_list]
    
    # Reduce-scatter src ops, dst shapes and dimensions
    if op_type == "reduce-scatter":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "reduce-scatter" in _str:
                idx_1 = _i
            elif "channel_id=" in _str:
                idx_2 = _i
            elif "dimensions" in _str:
                scatter_dimension = int(_str.split("dimensions={")[1].split("}")[0])
                break
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Dst shapes
        dst_shapes = [_slice_dst_shapes(_str) for _str in strs[2:idx_1]]
        # Src data types
        src_data_types = [global_op_db[_bias].data_type for _bias in src_op_bias_list]

    # All-to-all 
    if op_type == "all-to-all":
        idx_1, idx_2 = -1, -1
        for _i, _str in enumerate(strs):
            if _i > 0 and "all-to-all" in _str:
                idx_1 = _i
            elif "channel_id=" in _str:
                idx_2 = _i
        # Src op bias
        src_op_bias_list = [_slice_src_op_bias_list(_str) for _str in strs[idx_1:idx_2]]
        # Dst shapes
        dst_shapes = [_slice_dst_shapes(_str) for _str in strs[2:idx_1]]
        # Split and cancate dims are default to be 1
        split_dim, concate_dim = 0, 0
    
    # Useless Partition ID
    if op_type == "partition-id":
        return None
    
    try:
        operator_cls = cls_table[op_type]
    except KeyError as e:
        raise KeyError(
            f"Undefined HLO operator: Bias: {bias} | Operator type: {op_type}"
        ) from e
    
    # Instantiate HLO operator
    if op_type in ("compare", 
                   "minimum", "maximum", "add", "subtract", "multiply", "divide", "power",
                   "cosine", "exponential", "log", "negate", "rsqrt", "sqrt", "opt-barrier",
                   "and", "or", "select"):
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias, 
                              lhs_bias, rhs_bias, pred_bias, direction)
    elif op_type in ("dynamic-slice", "pad", "concatenate", "slice", "reshape", "reverse"):
        _src_op_bias_list = [src_op_bias] if op_type in ("dynamic-slice", "pad", "slice", "reshape", "reverse") \
                                            else src_op_bias_list
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, _src_op_bias_list, 
                              dynamic_slice_sizes, padding_configs, concate_dim, start_indices, limit_indices, 
                              slice_strides, reverse_dims)
    elif op_type in ("param", "copy"):
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, global_param_number, sharding)
        global_param_number += 1
    elif op_type == "constant":
        hlo_op = operator_cls(bias, op_type, data_type, value, shape)
    elif op_type == "convert":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias, dst_data_type, shape)
    elif op_type == "transpose":
        hlo_op = operator_cls(bias, op_type, data_type, shape, src_op_bias, transpose_dims)
    elif op_type == "bitcast":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias, 
                              src_shape, src_shape_dim_idx, 
                              dst_shape, dst_shape_dim_idx)
    elif op_type == "broadcast":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias,
                              src_shape, src_shape_dim_idx, 
                              dst_shape, dst_shape_dim_idx, broadcast_dims)
    elif op_type == "reduce":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias_list, 
                              shape, shape_dim_idx, src_shapes, dst_data_types, reduced_dims)
    elif op_type == "reduce-window":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias_list, shape, shape_dim_idx, 
                              dst_data_types, window_dims, window_strides, padding)
    elif op_type == "select-and-scatter":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias_list, shape, shape_dim_idx, window_dims, 
                              window_strides, padding)
    elif op_type == "iota":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, iota_dim)
    elif op_type == "tuple":
        hlo_op = operator_cls(bias, op_type, data_type, src_op_bias_list, shape, shape_dim_idx, dst_shapes)
    elif op_type == "fusion":
        hlo_op = operator_cls(bias, op_type, data_type, shape, dst_data_types, dst_shapes, 
                              called_fusion_bias, fusion_kind)
    elif op_type == "get-tuple-element":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias, 
                              gte_idx)
    elif op_type == "clamp":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias_list)
    elif op_type == "cudnn-conv":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias_list, 
                              src_shapes, window_sizes, window_strides, padding, lhs_dilate, rhs_dilate)
    elif op_type == "cublas-gemm":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias_list, src_shapes, 
                              lhs_batch_dims, rhs_batch_dims, lhs_contracting_dims, rhs_contracting_dims)
    elif op_type == "all-reduce":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias,
                              channel_id, replica_groups)
    elif op_type == "reduce-scatter":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias_list, 
                              src_data_types, dst_shapes, channel_id, replica_groups, 
                              scatter_dimension)
    elif op_type == "all-gather":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias_list, 
                              src_data_types, dst_shapes, channel_id, replica_groups, 
                              all_gather_dimension)
    elif op_type == "all-to-all":
        hlo_op = operator_cls(bias, op_type, data_type, shape, shape_dim_idx, src_op_bias_list, 
                              dst_shapes, channel_id, replica_groups, split_dim, concate_dim)
    
    # Update db
    global_op_db[hlo_op.bias] = hlo_op

    return hlo_op


def _get_comm_group_and_normalize_replica_groups(replica_groups: Sequence[Sequence[int]], 
                                                 replica_to_device_mapping: Sequence[Sequence[int]], 
                                                 num_devices: int):
    """ 
    Get the communication group with the given device cluster for current stage (mesh), get the 
    normalized replica groups for further querying the offline-profiled communication data.
    Format of communication group: (num_hosts, num_devices_per_host)
    Single-host normalizing example: [[0, 2], [1, 3]] (not profiled) -> [[0, 1], [2, 3]] (profiled)
    Multi-hosts normalizing example (Replica -> device: [[0, 1, 2, 3], [4, 5, 6, 7]]): 
                                     [[0, 2, 4, 6], [1, 3, 5, 7]] (not profiled, cross-nodes) 
                                        -> [[0, 1, 4, 5], [2, 3, 6, 7]] (profiled, cross-nodes)
    """
    norm_replica_groups = None
    comm_group = (len(replica_to_device_mapping), len(replica_to_device_mapping[0]))

    if len(replica_groups) == 1 and len(replica_groups[0]) == 1:
        # Since communication cannot be performed on one gpu, reset communication group to all gpus
        norm_replica_groups = [[_i for _i in range(num_devices)]]
        return comm_group, norm_replica_groups

    if len(replica_to_device_mapping) == 1:
        # Only single host
        _group_num = len(replica_groups)
        _group_size = len(replica_groups[0])
        assert _group_num * _group_size == num_devices, \
            f"Mismatched replica groups ({replica_groups}) with the devices num ({num_devices})"
        norm_replica_groups = [
            [_i * _group_size + _j for _j in range(_group_size)] for _i in range(_group_num)
        ]
    else:
        # Multiple hosts, need to further consider whether to normalized to replica groups with 
        # cross-nodes communication.
        num_hosts_crossed = get_crossed_host_num_of_replica_groups(replica_groups, replica_to_device_mapping)
        _intra_host_size = len(replica_groups[0]) // num_hosts_crossed
        norm_replica_groups = generate_normalized_replica_groups(replica_to_device_mapping, 
                                                                 num_hosts_crossed, _intra_host_size)
    
    return comm_group, norm_replica_groups
    
    # if len(replica_groups) == 1 and len(replica_groups[0]) == 1:
    #     # Since communication cannot be performed on one gpu, reset communication group to all gpus
    #     return (len(replica_to_device_mapping), len(replica_to_device_mapping[0]))

    # # Format of communication group: (num_hosts, num_devices_per_host)
    # comm_group = [0 for _ in range(len(replica_to_device_mapping))]
    # for _replica_id in replica_groups[0]:
    #     # Each replica group should share the same communication group, so just use the first one
    #     for _i, _node_rec in enumerate(replica_to_device_mapping):
    #         if _replica_id in _node_rec:
    #             comm_group[_i] += 1
    # comm_group = remove_all(comm_group, [0])

    # return (len(comm_group), comm_group[0])


def reconstr_hlo_entry_and_stat_comm(comm_time_tables: dict, 
                                     replica_to_device_mapping: Sequence[Sequence[int]],
                                     cross_stages_comm_vars: Optional[Sequence[Tuple[Any, str]]], 
                                     hlo_text: str, 
                                     send_stage_shape: Tuple[int] = None,
                                     recv_stage_shape: Tuple[int] = None,
                                     backward_pass: bool = False,
                                     analyze_comm: bool = False, 
                                     seperate_grad_sync: bool = False, 
                                     analyze_cross_comm: bool = False,
                                     cross_nodes_module: bool = False):
    """
    Parse HLO text and extract metadata of the main entrypoint of the HLO module. 
    If the given HLO text is optimized, statically analyze the intra-module and 
    inter-module communication time. 
    """
    
    global global_param_number
    num_devices = int(os.environ.get("NUM_DEVICES_CUR_STAGE"))
    # Intra-stage communication
    module_comm_time = 0
    # In multiple micro-batches case, we should skip gradient sync in backward propagation 
    # except for the last micro-batch.
    largest_comm_size = -1
    grad_sync_comm_time = 0

    # Parse hlo text
    entry, bias = None, None
    hlo_text = hlo_text.split("\n")
    for line in hlo_text:
        if line.split(" ")[0] == "ENTRY":
            # Begin of the entry
            bias = line.split(" ")[1]
            entry = Computation(bias=bias, op_group=list())
            continue
        if bias:
            if len(line) > 0 and line[0] != "}":
                assert entry is not None, "Entry is not properly init."
                if "ROOT" in line:
                    line = line.split("ROOT")[1]
                
                # Construct one HLO operator
                _hlo_op = parse_one_hlo_op(line)
                if _hlo_op is None:
                    continue
                entry.op_group.append(_hlo_op)

                # Intra-stage communication analysis
                if analyze_comm and _hlo_op.op_type in ("all-reduce", "all-gather", "all-to-all", "reduce-scatter"):
                    # Map replica groups to submeshes, get the replica groups of the operator
                    _comm_group, norm_replica_groups = _get_comm_group_and_normalize_replica_groups(_hlo_op.replica_groups, 
                                                                                                    replica_to_device_mapping, 
                                                                                                    num_devices)
                    assert len(norm_replica_groups) * len(norm_replica_groups[0]) == num_devices, \
                        f"Mismatched replica groups ({norm_replica_groups}) with the devices num ({num_devices})"
                    key = f"{_comm_group[0]}_n_{_comm_group[1]}_d"
                    assert key in comm_time_tables, f"Key `{key}` not found in comm_time_tables."
                    # Estimate
                    dtype = str(xc.XLA_ELEMENT_TYPE_TO_DTYPE[_hlo_op.data_type])
                    if _hlo_op.op_type == "all-reduce":
                        # All-reduce contains only one source operator
                        comm_size = np.prod(_hlo_op.shape)
                        comm_time = _estimate_comm_op_time(_hlo_op.op_type, _hlo_op.shape, dtype, 
                                                           norm_replica_groups, comm_time_tables[key])
                        if seperate_grad_sync and comm_size > largest_comm_size:
                            # Grad_sync must be an all-reduce communication. 
                            # We set grad_sync operator to the communication operator with the largest comm size
                            grad_sync_comm_time = comm_time
                            largest_comm_size = comm_size
                    else:
                        # Others can contain multiple source operators
                        comm_time = 0
                        for _shape in _hlo_op.dst_shapes:
                            comm_time += _estimate_comm_op_time(_hlo_op.op_type, _shape, dtype, 
                                                                norm_replica_groups, comm_time_tables[key])
                    module_comm_time += comm_time
            else:
                # End of the entry
                global_param_number = 0

    # Inter-stages communication analysis
    cross_stages_comm_time = 0
    if analyze_cross_comm:
        assert cross_stages_comm_vars is not None, \
            f"Cross-stages comm vars should not be none when analyzing cross-stages communication."
        
        # Estimate this by only performing p2p send/recv operation between two gpus
        key = "2_n_1_d" if cross_nodes_module and "2_n_1_d" in comm_time_tables else "1_n_2_d"
        assert key in comm_time_tables, f"Key `{key}` not found in comm_time_tables."
        for (shape, dtype) in cross_stages_comm_vars:
            unsharded_comm_size = np.prod(shape)
            all_tile_comm_size = _inspect_cross_stages_comm_size(
                send_stage_shape=send_stage_shape,
                recv_stage_shape=recv_stage_shape,
                unsharded_comm_size=unsharded_comm_size,
                backward_pass=backward_pass,
            )

            # Although the gpu-to-gpu send-recv communication can be performed concurrently, 
            # in most cases they are performed sequentially due to bandwidth contention.
            for tile_comm_size in all_tile_comm_size:
                # P2p send-recv 
                cross_stages_comm_time += _estimate_comm_op_time(
                    op_type="send",
                    data_shape=(tile_comm_size, ),
                    dtype=dtype,
                    replica_groups=None,
                    comm_time_table=comm_time_tables[key],
                )

            # _dp_degree_cur_stage = int(os.environ.get("DP_DEGREE"))
            # _comm_size = _comm_size // _dp_degree_cur_stage

            # # per_tile_comm_size = comm_size // cross_stages_comm_rank
            # per_tile_comm_size = comm_size

            # for _ in range(cross_stages_comm_rank):
            #     # Although the gpu-to-gpu send-recv communication can be performed concurrently, 
            #     # in most cases they are performed sequentially due to bandwidth contention.
            # # for _ in range(1):
            #     # Concurrently perform gpu-to-gpu send-recv communication
            #     cross_stages_comm_time += _estimate_comm_op_time(
            #         op_type="send",
            #         data_shape=(per_tile_comm_size, ),
            #         dtype=dtype,
            #         replica_groups=None,
            #         comm_time_table=comm_time_tables[key],
            #     )

            #     print(per_tile_comm_size)
            #     print(cross_stages_comm_time)
            #     print("")


            # cross_stages_comm_time += _estimate_comm_op_time("send", (comm_size, ), dtype, 
            #                                                  None, comm_time_tables[key])

        # root_hlo_op = entry.op_group[-1]
        # assert root_hlo_op.op_type == "tuple", "The root operator of the entry should be a tuple."
        # dtype = str(xc.XLA_ELEMENT_TYPE_TO_DTYPE[root_hlo_op.data_type])

        # for _shape in root_hlo_op.dst_shapes:
        #     _comm_size = np.prod(_shape)

        #     # TODO(chunyu): This simplification is only reasonable for stages with the same #devices per stage.
        #     # Since (most) params have been partitioned into multiple devices (also observed in data parallelism 
        #     # case, probably due to enabling zero), it should be many-to-many send/recv in real scenarios 
        #     # (local-allgather in alpa). Thus:
        #     # - Case 1: For dp case, since each replica can produce the entire (need further all-reduce in the end of 
        #     #           the backward flow) activation, device i in stage 1 can perform p2p send/recv communication to 
        #     #           device i in stage 2. Thus, we only consider comm_size / #gpu_for_dp data size in p2p send/recv 
        #     #           communication.
        #     # - Case 2: For mp case, since each partition cannot produce the entire activation, all-gather and other 
        #     #           communicating operators are called among devices in stage 1 before sending activation to stage 
        #     #           2. Thus, we consider comm_size data size in p2p send/recv communication.
        #     # - Case 3: When dp and mp are both applied in stages, we only conider #dp in the same way to case 1.
        #     #           TODO(chunyu): Evaluate this. 

        #     _dp_degree_cur_stage = int(os.environ.get("DP_DEGREE"))
        #     _comm_size = _comm_size // _dp_degree_cur_stage
        #     cross_stages_comm_time += _estimate_comm_op_time("send", (_comm_size, ), dtype, 
        #                                                      None, comm_time_tables[key])
    
    if analyze_comm and seperate_grad_sync:
        # assert grad_sync_comm_time > 0, "The time cost of gradient sync is not properly set."
        pass
    else:
        assert grad_sync_comm_time == 0, "Gradient sync should not be performed."
    
    return entry, (module_comm_time - grad_sync_comm_time), grad_sync_comm_time, cross_stages_comm_time


def _inspect_cross_stages_comm_size(
    send_stage_shape: Tuple[int],
    recv_stage_shape: Tuple[int],
    unsharded_comm_size: int,
    backward_pass: bool = False,
) -> List[int]:
    """
    Inspect the cross-tages communication size w.r.t. alpa's resharding strategies.

    The core idea is to track the data dependencies between sender and receiver stages. There are several cases 
    for cross-stages communication (assume that there are 2 GPUs for each stage by default, i.e., sender stage has
    two tiles (0, 1) and receiver stage has two tiles (2, 3)):

     - Case 1. The sender and receiver stage is parallelized with only data parallelism. 

         - Forward pass: Tile 2 only has data dependency from tile 0, and tile 3 only has that from tile 1, since 
                         all tiles are sliced only on the batch dimension and performed individually.
            
         - Backward pass: Tile 0 only has data dependency from tile 2, and tile 3 only has that from tile 1.

     - Case 2. The sender and receiver stage is parallelized with only tensor parallelism. 

         - Forward pass: Both tile 2 and tile 3 have data dependency from tile 0 and 1, since all tiles are sliced on 
                         the hidden dimension and need to be further gathered to serve as the input to each sharding 
                         (tile 2, 3) of the receiver stage.
            
         - Backward pass: Tile 0 only has data dependency from tile 2, and tile 3 only has that from tile 1, since
                          the backward proporgation ... [TODO]

     - Case 3. The sender stage is parallelized with only data parallelism, the receiver stage is parallelized 
               with only tensor parallelism.

         - Forward pass: Tile 2 only has data dependency from tile 0, and tile 3 only has that from tile 1, since 
                         all ... [TODO]
        
         - Backward pass: Both tile 0 and tile 1 have data dependency from tile 2 and 3, since tile 2 and 3 are sliced
                          on hidden dimension and must be gathered to align with the shape of tile 0 and 1.
    
     - Case 4. The sender stage is parallelized with only tensor parallelism, the receiver stage is parallelized 
               with only data parallelism.
        
         - Forward pass: Both tile 2 and tile 3 have data dependency from tile 0 and 1, since all tiles are sliced on 
                         the hidden dimension and need to be further gathered to serve as the input to each sharding 
                         (tile 2, 3) of the receiver stage.
        
         - Backward pass: Both tile 0 and tile 1 have data dependency from tile 2 and 3, since tile 2 and 3 are sliced
                          on batch dimension and must be gathered to align with the shape of tile 0 and 1.
     
     - Case 5. The sender/receiver stage is parallelized with hybrid parallelism.
         FIXME(chunyu): This is based on theretical analysis and infer.
    
         - Forward pass: A tile in the receiver stage has data dependency from all tiles with the same DP rank in the sender
                         stage.
         - Backward pass: Each receiver tile has data dependency from one sender tile.

     - Case 6. TODO(chunyu) The sender and receiver stages have different tile num.
    """

    if np.prod(send_stage_shape) == 1 and np.prod(recv_stage_shape) == 1:
        # One-to-one case
        return [unsharded_comm_size]

    if recv_stage_shape is None:
        # Last stage
        return [0]

    if (send_stage_shape[0] > 1 and send_stage_shape[1] == 1 and
        recv_stage_shape[0] > 1 and recv_stage_shape[1] == 1 and
        send_stage_shape[0] == recv_stage_shape[0]):
        # Case 1
        return [unsharded_comm_size // send_stage_shape[0] for _ in range(send_stage_shape[0])]
    
    elif (send_stage_shape[0] == 1 and send_stage_shape[1] > 1 and
          recv_stage_shape[0] == 1 and recv_stage_shape[1] > 1 and
          send_stage_shape[1] == recv_stage_shape[1]):
        # Case 2
        return [unsharded_comm_size for _ in range(send_stage_shape[1])] if not backward_pass \
            else [unsharded_comm_size // send_stage_shape[1] for _ in range(send_stage_shape[1])]
    
    elif (send_stage_shape[0] > 1 and send_stage_shape[1] == 1 and
          recv_stage_shape[0] == 1 and recv_stage_shape[1] > 1 and
          send_stage_shape[0] == recv_stage_shape[1]):
        # Case 3
        return [unsharded_comm_size // send_stage_shape[0] for _ in range(send_stage_shape[0])] if not backward_pass \
            else [unsharded_comm_size for _ in range(send_stage_shape[0])]

    elif (send_stage_shape[0] == 1 and send_stage_shape[1] > 1 and
          recv_stage_shape[0] > 1 and recv_stage_shape[1] == 1 and
          send_stage_shape[1] == recv_stage_shape[0]):
        # Case 4
        return [unsharded_comm_size for _ in range(send_stage_shape[1])] if not backward_pass \
            else [unsharded_comm_size for _ in range(send_stage_shape[1])]

    elif (send_stage_shape[0] > 1 and send_stage_shape[1] > 1 and
          recv_stage_shape[0] > 1 and recv_stage_shape[1] > 1 and
          send_stage_shape[0] == recv_stage_shape[0] and
          send_stage_shape[1] == recv_stage_shape[1]):
        # Case 5
        return [unsharded_comm_size // send_stage_shape[0] 
                    for _ in range(send_stage_shape[0] * send_stage_shape[1])] if not backward_pass \
            else [unsharded_comm_size // (send_stage_shape[0] * send_stage_shape[1]) for _ in range(send_stage_shape[0] * send_stage_shape[1])]

    else:
        raise ValueError(f"Unsupported cross-stages communication shape: Send stage shape: {send_stage_shape}, " + 
                         f"recv stage shape: {recv_stage_shape}.")


def constr_one_xla_op(computation: Computation):
    """ Construct one HLO computation as XLA operator. """
    xla_ops = dict()                            # OP alias -> xc.XlaOp
    _name = "XlaBuilder_" + computation.bias
    builder = xc.XlaBuilder(_name)              # Builder for XLA computations
    legacy_gemm_op_num = 0

    for hlo_op in computation.op_group:
        if hlo_op.op_type in ("param", "constant", "copy", "iota"):
            _xla_op = hlo_op.xla_op_expr(builder)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type in ("dynamic-slice", "pad", "concatenate", "slice", "reshape", "reverse"):
            _operands = [xla_ops[_bias] for _bias in hlo_op.src_op_bias_list]
            _op_data_types = [global_op_db[_bias].data_type for _bias in hlo_op.src_op_bias_list]
            _src_shape = global_op_db[hlo_op.src_op_bias_list[0]].shape
            _xla_op = hlo_op.xla_op_expr(builder, _operands, _src_shape, _op_data_types)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type in ("convert", "transpose", "bitcast", "broadcast", 
                                "all-reduce"):
            _operand = xla_ops[hlo_op.src_op_bias]
            _xla_op = hlo_op.xla_op_expr(_operand)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type in ("reduce", "reduce-window", "select-and-scatter", "reduce-scatter", 
                                "all-gather", "all-to-all"):
            operands = list()
            for _bias in hlo_op.src_op_bias_list:
                if ("constant" not in _bias) or (hlo_op.op_type not in ("reduce", "reduce-window")):
                    # Remove constant op for reduce/reduce-window
                    operands.append(xla_ops[_bias])
            _xla_op = hlo_op.xla_op_expr(builder, operands)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type == "tuple":
            _elements = [xla_ops[_bias] for _bias in hlo_op.src_op_bias_list]
            _xla_op = hlo_op.xla_op_expr(builder, _elements)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type in ("cosine", "exponential", "log", "negate", "rsqrt", "sqrt", "opt-barrier"):
            _operand = xla_ops[hlo_op.src_op_bias]
            _xla_op = hlo_op.xla_op_expr(_operand, None, None, None)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type in ("compare", "minimum", "maximum", "add", 
                                "subtract", "multiply", "divide", "power", "and", "or"): 
            _lhs = xla_ops[hlo_op.lhs_bias]
            _rhs = xla_ops[hlo_op.rhs_bias]
            _xla_op = hlo_op.xla_op_expr(None, _lhs, _rhs, None)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type == "select":
            # To get the shape of a XlaOp: builder.get_shape(xla_op)
            _lhs = xla_ops[hlo_op.lhs_bias]
            _rhs = xla_ops[hlo_op.rhs_bias]
            _pred = xla_ops[hlo_op.pred_bias]
            _xla_op = hlo_op.xla_op_expr(None, _lhs, _rhs, _pred)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type == "fusion":
            # Only reached when statically analyzing optimized module
            _key = hlo_op.called_fusion_bias
            _fused_xla_op = global_fused_xla_op_db[_key] \
                if _key in global_fused_xla_op_db.keys() else None
            _xla_op = hlo_op.xla_op_expr(builder, _fused_xla_op)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type == "get-tuple-element":
            _operand = xla_ops[hlo_op.src_op_bias]
            _xla_op = hlo_op.xla_op_expr(_operand)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type == "clamp":
            (_min, _operand, _max) = [xla_ops[_bias] for _bias in hlo_op.src_op_bias_list]
            _xla_op = hlo_op.xla_op_expr(_min, _operand, _max)
            xla_ops[hlo_op.bias] = _xla_op
        elif hlo_op.op_type in ("cudnn-conv", "cublas-gemm"):
            assert len(hlo_op.src_op_bias_list) == 2, \
                f"Mismatch operand num ({len(hlo_op.src_op_bias_list)}) of cudnn-conv operator."
            _lhs = xla_ops[hlo_op.src_op_bias_list[0]]
            _rhs = xla_ops[hlo_op.src_op_bias_list[1]]
            _xla_op = hlo_op.xla_op_expr(builder, _lhs, _rhs)
            xla_ops[hlo_op.bias] = _xla_op
            # Legacy case
            if hlo_op.op_type == "cublas-gemm" and hlo_op.is_legacy_bug:
                legacy_gemm_op_num += 1

    # Build the computation into one unified module
    root_xla_op = xla_ops[computation.op_group[-1].bias]
    hlo_module = builder.build(root_xla_op).as_hlo_module()

    # Record fused XLA computation
    if "fused_computation" in computation.bias or "horizontally" in computation.bias:
        global_fused_xla_op_db[computation.bias] = root_xla_op

    return WrappedHlo(module=hlo_module, status=HloStatus.UNOPTIMIZED), legacy_gemm_op_num


def profile_one_compiled_executable(compiled: xe.LoadedExecutable, backend: Any, 
                                    local_devices: Any, iter_num: int = 1, 
                                    warmup_num: int = 1, disable_cupti: bool = False):
    """ 
    Traverse one forward or backward (and apply_grad) stage for one time, profile 
    the performance of each fused kernel (hlo computation) except for communication 
    operations (offline profiled). 
    -----------------------------------------------------------------------
    Ref: alpa/util/profile_xla_executable()
    """
    global device_info_table
    is_sharded_execution = len(local_devices) > 1
    # Allow metric cache with same kernel spec
    cc.set_cache_metrics(True)
    # Prepare input
    hlo_module = compiled.hlo_modules()[0]
    input_shapes = hlo_module.parameter_shapes()            # Allocate dummy buffers
    free_mem = local_devices[0].available_memory()          # In case that OOM error occurs
    if free_mem < compiled.total_allocation_size() or free_mem == -1:
        return np.inf, None
    
    # Print option
    print_kernel = (os.environ.get("PRINT_LANUCHED_KERNEL", "false") == "true")
    # Input
    device_inputs = list()
    try:
        for shape in input_shapes:
            if not is_sharded_execution:
                device_inputs.append(
                    backend.buffer_from_pyval(
                        np.empty(shape.dimensions(), shape.numpy_dtype()), local_devices[0])
                )
            else:
                device_inputs.append([
                    backend.buffer_from_pyval(
                        np.empty(shape.dimensions(), shape.numpy_dtype()), device)
                    for device in local_devices
                ])
        local_devices[0].synchronize_all_activity()
    except RuntimeError:
        return np.inf, None

    # Run each executable with specified input
    def _runnable_func():
        # Execute one compiled module and trace GPU kernel events
        device_outputs = compiled.execute(device_inputs) if not is_sharded_execution \
                            else compiled.execute_sharded_on_local_devices(device_inputs)
        # Synchronize all activities, which will can `cuStreamSynchronize` in CUDA API.
        # Execution time of a kernel (profiled by CUPTI in device-side) might be overlapped in 
        # `cuStreamSynchronize` (host-side).
        local_devices[0].synchronize_all_activity()
    
    # # Profiling metrics
    # compute_major = int(os.environ.get("GPU_COMPUTE_MAJOR"))
    # prof_metrics = KERNEL_METRICS if compute_major >= 7 else LEGACY_KERNEL_METRICS

    # Per-kernel name and exec time
    per_kernel_infos = list()
    # Init global device info table
    if not device_info_table:
        device_info_table = load_device_info_table()
    
    # Device properties
    if False:
        # TODO(chunyu): Not implemented yet.
        device_props_list = list()
        for _gpu_type in CAND_GPU_TYPES:
            _device_info = device_info_table[_gpu_type]
            _properties = translate_to_device_properties(_device_info)
            device_props_list.append(cc.DeviceProperties(**_properties))
    
    # Warmup
    for _ in range(warmup_num):
        _runnable_func()

    # Run
    if disable_cupti:
        _runnable_func()
    else:
        total_kernel_time = 0
        for _i in range(iter_num):
            _kernel_time = 0
            # Profile execution time of the lanuched kernels with cpp backend
            kernels = cc.profile_runnable(_runnable_func)
            # Total execution time of the runnable
            if print_kernel:
                print("[I] Results of kernel profiling:")
            for _j, _kernel in enumerate(kernels):
                occupancies = dict()
                for _gpu_type in CAND_GPU_TYPES:
                    _device_info = device_info_table[_gpu_type] if _gpu_type in device_info_table else None
                    if _device_info is not None:
                        _properties = translate_to_device_properties(_device_info)
                        _device_props = cc.DeviceProperties(**_properties)
                        occupancies[_gpu_type] = _kernel.thread_block_occupancy(_device_props)
                per_kernel_infos.append((_kernel.name, _kernel.num_blocks, 
                                         _kernel.run_time_ns, occupancies))
                if print_kernel:
                    print(f"    - Kernel {_j} | Name: {_kernel.name} | Block num: {_kernel.num_blocks} | Time (ms): {_kernel.run_time_ns / 1e6}")
                _kernel_time += (_kernel.run_time_ns * NS_TO_S)
            print(f"    - Iteration {_i}/{iter_num}: Kernel execution time: {_kernel_time} s.")
            total_kernel_time += _kernel_time
    
    return (total_kernel_time / iter_num), per_kernel_infos


def compile_and_profile_one_xla_op(hlo_module: WrappedHlo, num_micro_batches: int, 
                                   backend: Any, local_devices: Any, 
                                   iter_num: int = 1, warmup_num: int = 1, 
                                   disable_cupti: bool = False):
    """ 
    Compile one XLA operator (unoptimized hlo module) and profile the performance of the lanuched kernel(s). 
    """
    # Annotate sharding on single device and compile the sharded hlo module
    print("[I] Compiling HLO module...")
    _time_marker = time.time()
    _sharded_hlo_module, _ = shard_one_hlo_module(hlo_module, logical_mesh_shape=(1, 1), 
                                                    force_batch_dim_to_mesh_dim=0, 
                                                    num_micro_batches=num_micro_batches,
                                                    num_devices=1)
    compiled = compile_one_sharded_hlo_module(_sharded_hlo_module, num_devices=len(local_devices), 
                                                backend=backend, bypass_device=False)
    compile_time = time.time() - _time_marker

    # Profile executable
    print("[I] Profiling HLO modules...")
    _time_marker = time.time()
    module_kernel_time, per_kernel_infos = profile_one_compiled_executable(compiled, backend, local_devices, iter_num,
                                                                           warmup_num, disable_cupti)
    profile_time = time.time() - _time_marker

    return module_kernel_time, per_kernel_infos, compile_time, profile_time


def _dummy_hlo_text():
    """ Dummy HLO text to be tested. """
    return [
        "param.4 = f32[112]{0} parameter(1), sharding={devices=[2]0,1}",
        "param.5 = f32[112]{0} parameter(120), sharding={devices=[2]0,1}",
        "param.2 = s32[16,224,224,3]{3,2,1,0} parameter(118), sharding={replicated}",
        "fusion.415 = f32[16,224,224,3]{2,1,3,0} fusion(param.2), kind=kLoop, calls=fused_computation.415",
        "param.3 = f32[7,7,3,112]{3,2,1,0} parameter(109), sharding={devices=[1,1,1,2]0,1}",
        "copy.1 = f32[7,7,3,112]{1,0,2,3} copy(param.3), sharding={devices=[1,1,1,2]0,1}",
        "cudnn-conv = (f32[16,112,112,112]{2,1,3,0}, u8[141312]{0}) custom-call(fusion.415, copy.1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f",
        "ROOT get-tuple-element.12 = f32[16,112,112,112]{2,1,3,0} get-tuple-element(cudnn-conv), index=0",
    ]


def test_translate_dummy_hlo_text_to_kernel_lanuch():
    """ Construct dummy HLO text to test translation from XLA operators to lanuched GPU kernels. """
    os.environ["CRIUS_NUM_HOSTS"] = "1"
    os.environ["CRIUS_NUM_DEVICES_PER_HOST"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    local_devices = [xb.local_devices()[0]]
    backend = xb.get_device_backend(local_devices[0])

    hlo_text = _dummy_hlo_text()
    computation = Computation(bias="test_comp", op_group=list())
    for _line in hlo_text:
        computation.op_group.append(parse_one_hlo_op(_line))
    hlo_module, _ = constr_one_xla_op(computation)
    hlo_print_option = xe.HloPrintOptions.short_parsable()
    print(hlo_module.get_module().to_string(hlo_print_option))

    _num_micro_batches = 16
    _iter_num, _warmup_num = 1, 1
    _disable_cupti = False
    compile_and_profile_one_xla_op(hlo_module, _num_micro_batches, backend, local_devices, _iter_num, 
                                   _warmup_num, _disable_cupti)
