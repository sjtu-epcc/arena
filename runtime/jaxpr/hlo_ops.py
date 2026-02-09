#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
A Python-style definition as the collection of all available HLO operators. 
---------------------------------------------------------------------
XLA Operation Sematics Doc: https://www.tensorflow.org/xla/operation_semantics?hl=en
"""

# TODO(chunyu): Define a base HLO operator class and generate multiple op types as Python dataclass.

from dataclasses import dataclass
from typing import Sequence, Union, Any, Tuple
from collections import namedtuple
from copy import deepcopy
import numpy as np
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxpr.utils import is_power_of

xops = xc.ops

"""
Collection of sharding specs for the operator.
"""
ShardingSpec = namedtuple("ShardingSpec", [
    "type", "device_list",
])

# Global type converted operators
global_type_converted_ops = list()
# Only consider sharded hlo rather than optimized by the compiler
only_sharded = True


class HloOperator:
    """ The base class of a HLO operator. """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 shape: Sequence[int], shape_dim_idx: Sequence[int] = None):
        self.bias = bias
        self.op_type = op_type
        self.data_type = data_type
        self.shape = shape
        self.shape_dim_idx = shape_dim_idx                          # Dimension index of the shape
        self.xla_op = None


#######################################
#          General Operators          #
#######################################

class ElementWiseOperator(HloOperator):
    """ The class of the general element-wise operators. """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], shape_dim_idx: Sequence[int], 
                 src_op_bias: str, lhs_bias: str, rhs_bias: str, pred_bias: str, direction: str = None):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias = src_op_bias
        self.lhs_bias = lhs_bias
        self.rhs_bias = rhs_bias
        self.pred_bias = pred_bias
        self.direction = direction
        # Available operator types
        self.avail_op_types = [
            "compare",
            "minimum", "maximum", "add", "subtract", "multiply", "divide", "power",
            "cosine", "exponential", "log", "negate", "rsqrt", "sqrt", "opt-barrier",
            "and", "or", "select",
        ]
    
    def xla_op_expr(self, operand: xc.XlaOp, lhs: xc.XlaOp, rhs: xc.XlaOp, pred: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        if self.op_type in ("compare"):
            return self._compare_op_expr(lhs, rhs)
        if self.op_type in ("minimum", "maximum", "add", "subtract", "multiply", "divide", "power"):
            return self._binary_arithmetic_op_expr(lhs, rhs)
        if self.op_type in ("cosine", "exponential", "log", "negate", "rsqrt", "sqrt", "opt-barrier"):
            return self._unary_op_expr(operand)
        if self.op_type in ("and", "or"):
            return self._bitwise_op_expr(lhs, rhs)
        if self.op_type in ("select"):
            return self._select_op_expr(pred, lhs, rhs)

    def _compare_op_expr(self, lhs: xc.XlaOp, rhs: xc.XlaOp):
        """ For compare operations. """
        assert self.direction is not None, \
            "Direction is not properly set for comparison operation."
        # Forcibly convert the data type of all operands
        if self.lhs_bias in global_type_converted_ops \
            or self.rhs_bias in global_type_converted_ops:
            lhs = xops.ConvertElementType(lhs, self.data_type)
            rhs = xops.ConvertElementType(rhs, self.data_type)
        if self.direction == "EQ":                  # Equal-to
            return xops.Eq(lhs, rhs)
        elif self.direction == "NE":                # Not equal-to
            return xops.Ne(lhs, rhs)
        elif self.direction == "GE":                # Greater-or-equal-than
            return xops.Ge(lhs, rhs)
        elif self.direction == "GT":                # Greater-than
            return xops.Gt(lhs, rhs)
        elif self.direction == "LE":                # Less-or-equal-than
            return xops.Le(lhs, rhs)
        elif self.direction == "LT":                # Less-than
            return xops.Lt(lhs, rhs)
    
    def _binary_arithmetic_op_expr(self, lhs: xc.XlaOp, rhs: xc.XlaOp):
        """ For binary arithmetic operations. """
        global global_type_converted_ops
        # Forcibly convert the data type of all operands
        if self.lhs_bias in global_type_converted_ops \
            or self.rhs_bias in global_type_converted_ops:
            lhs = xops.ConvertElementType(lhs, self.data_type)
            rhs = xops.ConvertElementType(rhs, self.data_type)
        if self.op_type == "minimum": 
            return xops.Min(lhs, rhs)
        elif self.op_type == "maximum":
            return xops.Max(lhs, rhs)
        elif self.op_type == "add":
            return xops.Add(lhs, rhs)
        elif self.op_type == "subtract":
            return xops.Sub(lhs, rhs)
        elif self.op_type == "multiply":
            return xops.Mul(lhs, rhs)
        elif self.op_type == "divide":
            return xops.Div(lhs, rhs)
        elif self.op_type == "power":
            return xops.Pow(lhs, rhs)

    def _unary_op_expr(self, operand: xc.XlaOp):
        """ For unary operations. """
        global global_type_converted_ops
        # Forcibly convert the data type of all operands
        if self.src_op_bias in global_type_converted_ops:
            operand = xops.ConvertElementType(operand, self.data_type)
        if self.op_type == "cosine":
            return xops.Cos(operand)
        elif self.op_type == "exponential":
            return xops.Exp(operand)
        elif self.op_type == "log":
            return xops.Log(operand)
        elif self.op_type == "negate":
            return xops.Neg(operand)
        elif self.op_type == "rsqrt":
            return xops.Rsqrt(operand)
        elif self.op_type == "sqrt":
            return xops.Sqrt(operand)
        elif self.op_type == "opt-barrier":
            return xops.OptimizationBarrier(operand)
    
    def _bitwise_op_expr(self, lhs: xc.XlaOp, rhs: xc.XlaOp):
        """ For bitwise operations. """
        if self.op_type == "and":
            return xops.And(lhs, rhs)
        elif self.op_type == "or":
            return xops.Or(lhs, rhs)
    
    def _select_op_expr(self, pred: xc.XlaOp, on_true: xc.XlaOp, on_false: xc.XlaOp):
        """ For select operation. """
        return xops.Select(pred, on_true, on_false)


class ReshapingOperator(HloOperator):
    """ 
    The class of the general reshaping operators. 
    -------------------------------------------------
    Types: reshape, concatenate, slice
    """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], shape_dim_idx: Sequence[int], 
                 src_op_bias_list: Sequence[str],
                 dynamic_slice_sizes: Sequence[int] = None, 
                 padding_configs: Sequence[Any] = None, 
                 concate_dim: int = None,
                 start_indices: Sequence[int] = None, limit_indices: Sequence[int] = None, 
                 slice_strides: Sequence[int] = None, reverse_dims: Sequence[int] = None):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list
        self.dynamic_slice_sizes = dynamic_slice_sizes
        self.padding_configs = padding_configs
        self.concate_dim = concate_dim
        self.start_indices = start_indices
        self.limit_indices = limit_indices
        self.slice_strides = slice_strides
        self.reverse_dims = reverse_dims
        # Available operator types
        self.avail_op_types = [
            "reshape", "concatenate", "slice", "dynamic-slice", "pad", "reverse"
        ]
        
    def _init_start_indices(self, builder: xc.XlaBuilder):
        """ Get the start indices (Seq of XlaOps) for dynamic slice. """
        # https://github.com/shawwn/jax/blob/602c783655fa6c0cdc9fe8d7d533d493e7de0ffe/jax/interpreters/pxla.py#L987
        zero = xops.Constant(builder, np.zeros((), dtype=np.uint32))
        indices = [zero] * len(self.dynamic_slice_sizes)
        return indices
    
    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp], src_shape: Sequence[int], 
                    op_data_types: Sequence[Any]):
        """ Get the expression in format of XLA operator. """
        if self.op_type == "dynamic-slice":
            _start_indices = self._init_start_indices(builder)
            self.xla_op = xops.DynamicSlice(operands[0], _start_indices, self.dynamic_slice_sizes)
        elif self.op_type == "pad":
            _padding_value = xops.Constant(builder, np.zeros((), dtype=xc.XLA_ELEMENT_TYPE_TO_DTYPE[op_data_types[0]]))
            _padding_configs = xc.make_padding_config(self.padding_configs)
            self.xla_op = xops.Pad(operands[0], _padding_value, _padding_configs)
        elif self.op_type == "concatenate":
            self.xla_op = xops.ConcatInDim(builder, operands, self.concate_dim)
        elif self.op_type == "slice":
            self.xla_op = xops.Slice(operands[0], self.start_indices, self.limit_indices, self.slice_strides)
        elif self.op_type == "reshape":
            _dims = [_i for _i in range(len(src_shape))]                    # Default: [0, 1, ..., N]
            self.xla_op = xops.Reshape(operands[0], _dims, self.shape)
        elif self.op_type == "reverse":
            self.xla_op = xops.Rev(operands[0], self.reverse_dims)
        return self.xla_op


#######################################
#          Fusable Operators          #
#######################################

class ParamOperator(HloOperator):
    """ The class of the parameter operator. """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], param_number: int, sharding: ShardingSpec):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.param_number = param_number                        # Number (order) of the parameter
        self.sharding = sharding
    
    def xla_op_expr(self, builder: xc.XlaBuilder):
        """ Get the expression in format of XLA operator. """
        shape = xc.Shape.array_shape(self.data_type, self.shape)
        replicated = list()
        self.xla_op = xops.Parameter(builder, self.param_number, 
                                     shape.with_major_to_minor_layout_if_absent(), 
                                     self.bias, replicated)
        return self.xla_op


class ConstantOperator(HloOperator):
    """ The class of the constant operator. """
    def __init__(self, bias: str, op_type: str, data_type: str, value: Any, shape: Sequence[int]):
        super().__init__(bias, op_type, data_type, shape)
        self.value = value
    
    def xla_op_expr(self, builder: xc.XlaBuilder):
        """ Get the expression in format of XLA operator. """
        self.xla_op = xops.Constant(builder, np.zeros(self.shape, 
                                                      dtype=xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.data_type]))
        return self.xla_op


class ConvertOperator(HloOperator):
    """ 
    The class of the convert operator, performing an single-element 
    data type conversion. 
    NOTE: The data type of the operand would also be converted.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, src_op_bias: str, 
                 dst_data_type: str, shape: Sequence[int]):
        super().__init__(bias, op_type, data_type, shape)
        self.src_op_bias = src_op_bias
        self.dst_data_type = dst_data_type 
        # Record type converted ops and converting ops
        global global_type_converted_ops
        # global_type_converted_ops.extend([self.src_op_bias, self.bias])
        global_type_converted_ops.extend([self.src_op_bias])

    def xla_op_expr(self, operand: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        self.xla_op = xops.ConvertElementType(operand, self.dst_data_type)
        return self.xla_op


class TransposeOperator(HloOperator):
    """ 
    The class of the transpose operator, permuting the operand dimensions with the 
    given permutation.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 src_op_bias: str, transpose_dims: Sequence[int]):
        super().__init__(bias, op_type, data_type, shape)
        self.src_op_bias = src_op_bias
        self.transpose_dims = transpose_dims
    
    def xla_op_expr(self, operand: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        self.xla_op = xops.Transpose(operand, permutation=self.transpose_dims)
        return self.xla_op


class BitCastOperator(HloOperator):
    """ 
    The class of the bitcast operator, performing an element-wise bitcast operation from a 
    data shape to a target shape. 
    """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 src_op_bias: str,
                 src_shape: Sequence[int], src_shape_dim_idx: Sequence[int], 
                 dst_shape: Sequence[int], dst_shape_dim_idx: Sequence[int]):
        super().__init__(bias, op_type, data_type, dst_shape, dst_shape_dim_idx)
        self.src_op_bias = src_op_bias
        self.src_shape = src_shape                              # Shape of the source data     
        self.src_shape_dim_idx = src_shape_dim_idx              # Dimension index of the source shape
        self.dst_shape = dst_shape                              # Shape of the destination data
        self.dst_shape_dim_idx = dst_shape_dim_idx              # Dimension index of the destionation shape

    def xla_op_expr(self, operand: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        # Implement as a Reshape operation
        _dims = [_i for _i in range(len(self.src_shape))]       # Default: [0, 1, ..., N]
        self.xla_op = xops.Reshape(operand, _dims, self.dst_shape)
        return self.xla_op


class BroadcastOperator(HloOperator):
    """ 
    The class of the broadcast operator, adding dimensions to an array by duplicating 
    the data in the array.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 src_op_bias: str,
                 src_shape: Sequence[int], src_shape_dim_idx: Sequence[int], 
                 dst_shape: Sequence[int], dst_shape_dim_idx: Sequence[int], 
                 broadcast_dims: Sequence[int]):
        super().__init__(bias, op_type, data_type, dst_shape, dst_shape_dim_idx)
        self.src_op_bias = src_op_bias
        self.src_shape = src_shape                              
        self.src_shape_dim_idx = src_shape_dim_idx              
        self.dst_shape = dst_shape                            
        self.dst_shape_dim_idx = dst_shape_dim_idx
        # Which dimension in the dst shape that dimension of the src shape corresponds to
        self.broadcast_dims = broadcast_dims
    
    def xla_op_expr(self, operand: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        # https://github.com/openxla/xla/blob/c145a9b8c8a3a7d57c54f8957c4d76c8e3b9e668/xla/service/shape_inference.cc#L867
        if len(self.broadcast_dims) > 0:
            self.xla_op = xops.BroadcastInDim(operand, self.dst_shape, self.broadcast_dims)
        else:
            self.xla_op = xops.Broadcast(operand, self.dst_shape)
        return self.xla_op


class ReduceOperator(HloOperator):
    """ The class of the reduce operator. """
    def __init__(self, bias: str, op_type: str, data_type: str, src_op_bias_list: Sequence[str],
                 shape: Sequence[int], shape_dim_idx: Sequence[int], src_shapes: Sequence[Sequence[int]], 
                 dst_data_types: Sequence[str], reduced_dims: Sequence[int]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list 
        self.src_shapes = src_shapes    
        self.dst_data_types = dst_data_types            
        self.reduced_dims = reduced_dims                                # Data dimension to be reduced

    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        # XLA computation is set to 'add' by default
        xla_ops = list()
        xla_computation = xc.XlaBuilder("add")
        for i in range(len(self.dst_data_types)):
            _dtype = xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.dst_data_types[i]]
            init_vals = xops.ConstantLiteral(builder, np.array(0, _dtype))
            shape = xc.Shape.array_shape(self.dst_data_types[i], ())    # Scalar
            xops.Add(xops.Parameter(xla_computation, 0, shape), 
                    xops.Parameter(xla_computation, 1, shape))
            _xla_op = xops.Reduce(builder, [operands[i]], [init_vals], 
                                    xla_computation.build(), self.reduced_dims)
            xla_ops.append(_xla_op)
        self.xla_op = xla_ops[0] if len(xla_ops) == 1 else xops.Tuple(builder, xla_ops)
        
        # self.xla_op = xops.Constant(builder, np.zeros(self.shape, 
        #                 dtype=xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.dst_data_types[0]]))
        # return self.xla_op
        
        return self.xla_op


class ReduceWindowOperator(HloOperator):
    """ 
    The class of the reduce-window operator, applying a reduction function to all elements in each window of 
    a sequence of N multi-dimensional arrays, producing a single or a tuple of N multi-dimensional arrays as 
    output.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, src_op_bias_list: Sequence[str],
                 shape: Sequence[int], shape_dim_idx: Sequence[int], dst_data_types: Sequence[str],
                 window_dims: Sequence[int], window_strides: Sequence[int], 
                 padding: Sequence[Tuple[int, int]]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list 
        self.dst_data_types = dst_data_types
        self.window_dims = window_dims                                  # Window size for reduction            
        self.window_strides = window_strides                            # Window stride
        self.padding = padding                                          # Padding

    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        # XLA computation is set to 'add' by default
        xla_ops = list()
        xla_computation = xc.XlaBuilder("add")
        for i in range(len(self.dst_data_types)):
            _dtype = xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.dst_data_types[i]]
            init_vals = xops.ConstantLiteral(builder, np.array(0, _dtype))
            shape = xc.Shape.array_shape(self.dst_data_types[i], ())    # Scalar
            xops.Add(xops.Parameter(xla_computation, 0, shape), 
                    xops.Parameter(xla_computation, 1, shape))
            _base_dilations = [1 for _ in range(len(self.padding))]
            _window_dilations = [1 for _ in range(len(self.padding))]
            _xla_op = xops.ReduceWindowWithGeneralPadding(operands[i], init_vals, xla_computation.build(), 
                                                          self.window_dims, self.window_strides, _base_dilations, 
                                                          _window_dilations, padding=self.padding)
            xla_ops.append(_xla_op)
        self.xla_op = xla_ops[0] if len(xla_ops) == 1 else xops.Tuple(builder, xla_ops)
        return self.xla_op


class SelectAndScatterOperator(HloOperator):
    """ 
    The class of the select-and-scatter operator, can be considered as a composite operation that first 
    computes ReduceWindow on the operand array to select an element from each window, and then scatters 
    the source array to the indices of the selected elements to construct an output array with the same 
    shape as the operand array.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, src_op_bias_list: Sequence[str],
                 shape: Sequence[int], shape_dim_idx: Sequence[int], window_dims: Sequence[int], 
                 window_strides: Sequence[int], padding: Sequence[Tuple[int, int]]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list
        self.window_dims = window_dims
        self.window_strides = window_strides
        self.padding = padding

    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        assert len(operands) == 3, "Mismatched src op num in select-and-scatter operator."
        (_operand, _source, _init_value) = operands
        _padding = self.padding if self.padding is not None \
                                else [(0, 0) for _ in range(len(self.window_strides))]

        select_computation = xc.XlaBuilder("select")
        _shape = xc.Shape.array_shape(xc.PrimitiveType.F32, ())
        xops.Ge(xops.Parameter(select_computation, 0, _shape),      # Default to direction=GE
                xops.Parameter(select_computation, 1, _shape))
        
        scatter_computation = xc.XlaBuilder("scatter")
        _shape = xc.Shape.array_shape(xc.PrimitiveType.F32, ())
        xops.Add(xops.Parameter(scatter_computation, 0, _shape), 
                 xops.Parameter(scatter_computation, 1, _shape))
        
        self.xla_op = xops.SelectAndScatterWithGeneralPadding(_operand, select_computation.build(), 
                                                              self.window_dims, self.window_strides,
                                                              _padding, _source, _init_value, 
                                                              scatter_computation.build())
        # # Fake implementation
        # self.xla_op = xops.Constant(builder, np.zeros(self.shape, 
        #                             dtype=xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.data_type]))
        return self.xla_op


class CopyOperator(HloOperator):
    """ The class of the copy operator, copying the target data. """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], param_number: int, sharding: ShardingSpec):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)                   
        self.param_number = param_number                        
        self.sharding = sharding
    
    def xla_op_expr(self, builder: xc.XlaBuilder):
        """ Get the expression in format of XLA operator. """
        shape = xc.Shape.array_shape(self.data_type, self.shape)
        replicated = list()
        self.xla_op = xops.Parameter(builder, self.param_number, 
                                     shape.with_major_to_minor_layout_if_absent(), 
                                     self.bias, replicated)
        return self.xla_op
    

class IotaOperator(HloOperator):
    """ 
    The class of the iota operator, which builds a constant literal on device rather than 
    a potentially large host transfer.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 shape: Sequence[int], shape_dim_idx: Sequence[int], iota_dimension: int):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.iota_dimension = iota_dimension
    
    def xla_op_expr(self, builder: xc.XlaBuilder):
        """ Get the expression in format of XLA operator. """
        shape = xc.Shape.array_shape(self.data_type, self.shape)
        self.xla_op = xops.Iota(builder, shape, self.iota_dimension)
        return self.xla_op


class TupleOperator(HloOperator):
    """ 
    The class of the tuple operator, a tuple containing a variable number of 
    data handles, each of which has its own shape.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 src_op_bias_list: Sequence[str], shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], dst_shapes: Sequence[Sequence[int]]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list
        self.dst_shapes = dst_shapes
    
    def xla_op_expr(self, builder: xc.XlaBuilder, elements: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        self.xla_op = xops.Tuple(builder, elements)
        return self.xla_op


#######################################
#         Unfusable Operators         #
#######################################

class FusionHookOperator(HloOperator):
    """ The hook operator for calling one fused computation. """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int],
                 dst_data_types: Sequence[str], dst_shapes: Sequence[Sequence[int]], 
                 called_fusion_bias: str, fusion_kind: str):
        super().__init__(bias, op_type, data_type, shape)
        self.dst_data_types = dst_data_types
        self.dst_shapes = dst_shapes
        self.called_fusion_bias = called_fusion_bias
        self.fusion_kind = fusion_kind
    
    def xla_op_expr(self, builder: xc.XlaBuilder, fused_xla_op: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        # Intro to kLoop/kInput XLA fusion startegy: 
        # https://bladedisc.oss-cn-hangzhou.aliyuncs.com/docs/performance-optimization-practice.pdf
        fake_fused_xla_op = True
        if not fake_fused_xla_op:
            self.xla_op = fused_xla_op
        else:
            xla_ops = list()
            for _i, _dst_shape in enumerate(self.dst_shapes):
                xla_ops.append(xops.Constant(builder, np.zeros(_dst_shape, 
                                                dtype=xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.dst_data_types[_i]])))
            self.xla_op = xla_ops[0] if len(xla_ops) == 1 else xops.Tuple(builder, xla_ops)
        return self.xla_op


class GTEOperator(HloOperator):
    """ 
    The class of the GetTupleElement operator, indexing into a tuple with a 
    compile-time-constant value. The value must be a compile-time-constant 
    so that shape inference can determine the type of the resulting value.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], src_op_bias: str, index: int):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias = src_op_bias
        self.index = index                                      # Index of the tuple
    
    def xla_op_expr(self, operand: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        self.xla_op = xops.GetTupleElement(operand, self.index)
        return self.xla_op


class ClampOperator(HloOperator):
    """ 
    The class of the Clamp operator, which clamps an operand to within the range 
    between a minimum and maximum value. Clamp(a, op, b) = min(max(a, op), b).
    """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], src_op_bias_list: Sequence[str]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list
    
    def xla_op_expr(self, min: xc.XlaOp, operand: xc.XlaOp, max: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        self.xla_op = xops.Clamp(min, operand, max)
        return self.xla_op


class CudnnConvOperator(HloOperator):
    """ 
    The class of the CUDNN Conv operator. This operator can be implemented with different GPU kernels 
    on different GPUs. (TODO) How to estimate this?
    """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], src_op_bias_list: Sequence[str], 
                 src_shapes: Sequence[Sequence[int]], window_sizes: Sequence[int], 
                 window_strides: Sequence[int], padding: Sequence[Tuple[int, int]], 
                 lhs_dilate: Sequence[int], rhs_dilate: Sequence[int]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list
        self.src_shapes = src_shapes
        self.window_sizes = window_sizes
        self.window_strides = window_strides                        # Window strides
        self.padding = padding                                      # Padding size
        self.lhs_dilate = lhs_dilate                                # Lhs dilation
        self.rhs_dilate = rhs_dilate                                # Rhs dilation
        self.micro_batch_size = int(os.environ.get("MICRO_BATCH_SIZE"))
    
    def _generate_conv_dim_numbers(self):
        """ Generate dimension numbers for convolution based on src and dst shapes. """
        assert len(self.src_shapes) == 2, "Mismatched num of src shapes."
        (input_shape, kernel_shape) = self.src_shapes

        # print(input_shape, kernel_shape)
        # print(self.shape)
        
        # Step 0. Spatial dims for input/kernel/output
        input_spatial_dims = list()
        for _i in range(len(input_shape) - 1):
            if input_shape[_i] == input_shape[_i + 1]:
                input_spatial_dims.extend([_i, _i + 1])
                break
        assert len(input_spatial_dims) > 0, "Input spatial dims are not init."

        kernel_spatial_dims = list()
        for _i in range(len(kernel_shape) - 1):
            if kernel_shape[_i] == self.window_sizes[0] \
                and kernel_shape[_i + 1] == self.window_sizes[0]:
                kernel_spatial_dims.extend([_i, _i + 1])
                break
        assert len(kernel_spatial_dims) > 0, "Kernel spatial dims are not init."

        output_spatial_dims = list()
        for _i in range(len(self.shape) - 1):
            if self.shape[_i] == self.shape[_i + 1]:
                output_spatial_dims.extend([_i, _i + 1])
                break
        assert len(output_spatial_dims) > 0, "Output spatial dims are not init."
        
        if input_shape[0] != kernel_shape[0]:
            # Ordinary cases
            # Example: [16, 7, 7, 3584] @ [3, 3, 448, 3584] -> [16, 7, 7, 448]
            # Step 1. Batch dim for input/output
            input_batch_dim = -1
            for _i in range(len(input_shape)):
                if (_i not in input_spatial_dims) and (input_shape[_i] == self.micro_batch_size):
                    input_batch_dim = _i
                    break
            if input_batch_dim == -1:
                # In multi-hosts cases with #dp > 1 and #mp > 1, this abnormal case appears 
                # and remains unhandled.
                _batch_size = self.micro_batch_size // 2
                while _batch_size >= 1 and input_batch_dim == -1:
                    for _i in range(len(input_shape)):
                        if (_i not in input_spatial_dims) and (input_shape[_i] == _batch_size):
                            input_batch_dim = _i
                            break
                    _batch_size = _batch_size // 2
            assert input_batch_dim != -1, "Input batch dim is not init."

            output_batch_dim = -1
            for _i in range(len(self.shape)):
                if (_i not in output_spatial_dims) and (self.shape[_i] == input_shape[input_batch_dim]):
                    output_batch_dim = _i
                    break
            assert output_batch_dim != -1, "Output batch dim is not init."

            # Step 2. Feature dim for input/output
            input_feature_dim = -1
            for _i in range(len(input_shape)):
                if (_i not in input_spatial_dims) and (_i != input_batch_dim):
                    input_feature_dim = _i
                    break
            assert input_feature_dim != -1, "Input feature dim is not init."

            output_feature_dim = -1
            for _i in range(len(self.shape)):
                if (_i not in output_spatial_dims) and (_i != output_batch_dim):
                    output_feature_dim = _i
                    break
            assert output_feature_dim != -1, "Output feature dim is not init."
            output_feature_num = self.shape[output_feature_dim]

            # Step 3. Kernel output (feature) and (input) feature dim
            kernel_output_dim = -1
            for _i in range(len(kernel_shape)):
                if (_i not in kernel_spatial_dims) and (kernel_shape[_i] == output_feature_num):
                    kernel_output_dim = _i
                    break
            assert kernel_output_dim != -1, "Kernel output (feature) dim is not init."

            kernel_feature_dim = -1
            for _i in range(len(kernel_shape)):
                if (_i not in kernel_spatial_dims) and (_i != kernel_output_dim):
                    kernel_feature_dim = _i
                    break
            assert kernel_feature_dim != -1, "Kernel (input) feature dim is not init."
        else:
            # Abnormal cases
            # Example: [16, 7, 7, 1792] @ [16, 7, 7, 1792] -> [1, 1, 1792, 1792]
            # Step 1. Batch dim for input and output feature dim for kernel
            input_batch_dim = -1
            for _i in range(len(input_shape)):
                if (_i not in input_spatial_dims) and (input_shape[_i] in self.shape):
                    input_batch_dim = _i
                    break
            assert input_batch_dim != -1, "Input batch dim is not init."

            kernel_output_dim = -1
            for _i in range(len(kernel_shape)):
                if (_i not in kernel_spatial_dims) and (kernel_shape[_i] in self.shape):
                    kernel_output_dim = _i
                    break
            assert kernel_output_dim != -1, "Kernel output (feature) dim is not init."

            # Step 2. Feature num for input/kernel
            input_feature_dim = -1
            for _i in range(len(input_shape)):
                if (_i not in input_spatial_dims) and (_i != input_batch_dim):
                    input_feature_dim = _i
                    break
            assert input_feature_dim != -1, "Input feature dim is not init."

            kernel_feature_dim = -1
            for _i in range(len(kernel_shape)):
                if (_i not in kernel_spatial_dims) and (_i != kernel_output_dim):
                    kernel_feature_dim = _i
                    break
            assert kernel_feature_dim != -1, "Kernel (input) feature dim is not init."

            # Step 3. Output batch and feature dim
            output_batch_dim = -1
            for _i in range(len(self.shape)):
                if (_i not in output_spatial_dims) and (self.shape[_i] in input_shape):
                    output_batch_dim = _i
                    break
            assert output_batch_dim != -1, "Output batch dim is not init."

            output_feature_dim = -1
            for _i in range(len(self.shape)):
                if (_i not in output_spatial_dims) and (_i != output_batch_dim):
                    output_feature_dim = _i
                    break
            assert output_feature_dim != -1, "Output feature dim is not init."

            # print(input_spatial_dims, input_batch_dim, input_feature_dim)
            # print(kernel_spatial_dims, kernel_output_dim, kernel_feature_dim)
            # print(output_spatial_dims, output_batch_dim, output_feature_dim)
            # print(self.bias)

            # if self.bias == "convolution.5":
            #     exit(0)

        _conv_dim_numbers = xc.ConvolutionDimensionNumbers()
        # Input dimensions (e.g., f32[16, 224, 224, 3] -> [224, 224] (spatial), [16] (batch), [3] (feature))
        _conv_dim_numbers.input_spatial_dimensions = input_spatial_dims
        _conv_dim_numbers.input_batch_dimension = input_batch_dim
        _conv_dim_numbers.input_feature_dimension = input_feature_dim
        # Kernel dimensions (e.g., f32[7, 7, 3, 112] -> [7, 7] (spatial), [3] (input), [112] (output))
        _conv_dim_numbers.kernel_spatial_dimensions = kernel_spatial_dims
        _conv_dim_numbers.kernel_input_feature_dimension = kernel_feature_dim
        _conv_dim_numbers.kernel_output_feature_dimension = kernel_output_dim
        # Output dimensions (e.g., f32[16, 112, 112, 112] -> [112, 112] (spatial), [16] (batch), [112] (feature))
        _conv_dim_numbers.output_spatial_dimensions = output_spatial_dims
        _conv_dim_numbers.output_batch_dimension = output_batch_dim
        _conv_dim_numbers.output_feature_dimension = output_feature_dim
        # Convolution dimension numbers
        dim_nums = xc.make_convolution_dimension_numbers(dimension_numbers=_conv_dim_numbers, 
                                                         num_spatial_dimensions=(len(input_shape) - 2))

        # LHS feature dimension / feature_group_count = RHS feature dimension
        feature_group_count = input_shape[input_feature_dim] // kernel_shape[kernel_feature_dim]
        batch_group_count = 1
        
        return dim_nums, feature_group_count, batch_group_count

    def xla_op_expr(self, builder: xc.XlaBuilder, lhs: xc.XlaOp, rhs: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        # XLA custom call: https://www.tensorflow.org/xla/custom_call
        # In XLA-GPU: ConvGeneralDilated (XLA) (custom call) -> cudnn-conv (cudnn library)
        assert len(self.src_shapes) == 2, "Mismatch operand num (expected 2)."
        
        _window_strides = self.window_strides if self.window_strides is not None \
                                              else [1 for _ in range(len(self.src_shapes[0]) - 2)]
        _padding = self.padding if self.padding is not None \
                                else [(0, 0) for _ in range(len(_window_strides))]
        _lhs_dilation = self.lhs_dilate if self.lhs_dilate is not None else [1 for _ in range(len(_padding))]
        _rhs_dilation = self.rhs_dilate if self.rhs_dilate is not None else [1 for _ in range(len(_padding))]
        dim_nums, feature_group_count, batch_group_count = self._generate_conv_dim_numbers()

        self.xla_op = xops.ConvGeneralDilated(lhs, rhs, _window_strides, _padding, _lhs_dilation, _rhs_dilation, 
                                              dim_nums, feature_group_count, batch_group_count)
        return self.xla_op


class CublasGEMMOperator(HloOperator):
    """ 
    The class of the Cublas GEMM operator. This operator can be implemented with different GPU kernels 
    on different GPUs. (TODO) How to estimate this?
    """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], src_op_bias_list: Sequence[str], 
                 src_shapes: Sequence[Sequence[int]], lhs_batch_dims: Sequence[int], 
                 rhs_batch_dims: Sequence[int], lhs_contracting_dims: Sequence[int], 
                 rhs_contracting_dims: Sequence[int]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias_list = src_op_bias_list
        self.src_shapes = src_shapes
        self.lhs_batch_dims = lhs_batch_dims
        self.rhs_batch_dims = rhs_batch_dims
        self.lhs_contracting_dims = lhs_contracting_dims
        self.rhs_contracting_dims = rhs_contracting_dims
        self.is_legacy_bug = False
    
    def xla_op_expr(self, builder: xc.XlaBuilder, lhs: xc.XlaOp, rhs: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        # assert len(self.shape) == 2, "Currently cublas-gemm only supports operands with 2-dimensions."
        # In XLA-GPU: DotGeneral (XLA) (custom call) -> cublas-gemm (cublas library)
        # Default to contract the second dim in lhs with the first dim in rhs
        num_devices_cur_stage = int(os.environ.get("NUM_DEVICES_CUR_STAGE"))
        lhs = xops.ConvertElementType(lhs, self.data_type)
        rhs = xops.ConvertElementType(rhs, self.data_type)
        dtype = xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.data_type]
        is_legacy_bug = (len(self.rhs_contracting_dims) > 0) and (self.rhs_contracting_dims[0] == 1) \
                        and (num_devices_cur_stage == 1)

        if not is_legacy_bug:
            # Odinary cases
            _lhs_batch_dims = tuple(self.lhs_batch_dims) if self.lhs_batch_dims else ()
            _rhs_batch_dims = tuple(self.rhs_batch_dims) if self.rhs_batch_dims else ()
            _lhs_contracting_dims = tuple(self.lhs_contracting_dims) if self.lhs_contracting_dims else ()
            _rhs_contracting_dims = tuple(self.rhs_contracting_dims) if self.rhs_contracting_dims else ()
            _dim_nums = xc.make_dot_dimension_numbers(((_lhs_contracting_dims, _rhs_contracting_dims), 
                                                    (_lhs_batch_dims, _rhs_batch_dims)))
            self.xla_op = xops.DotGeneral(lhs, rhs, _dim_nums)
        else:
            # Abnormal cases
            # A legacy bug will be triggered in GEMM operator when GPU num is set to 1. Source code of this error:
            #   https://github.com/openxla/xla/blob/c227585959ec96a4527b1b9f9023f8d6bbe976b3/xla/service/gpu/matmul_utils.cc#L360C20
            # Fake implementation, need to link lhs/rhs with the computation graph behind (operators behind may also call the 
            #   opeartors before this gemm)
            # assert len(self.shape) == 2
            self.is_legacy_bug = True

            # Reduce
            xla_computation = xc.XlaBuilder("add")
            init_vals = xops.ConstantLiteral(builder, np.array(0, dtype))
            shape = xc.Shape.array_shape(self.data_type, ())    # Scalar
            xops.Add(xops.Parameter(xla_computation, 0, shape), 
                    xops.Parameter(xla_computation, 1, shape))
            _reduce_xla_op = xops.Reduce(builder, [lhs], [init_vals], xla_computation.build(), self.lhs_contracting_dims)
            
            # Reshape
            _dims = [_i for _i in range(len(self.src_shapes[0]) - len(self.lhs_contracting_dims))]
            _shape = list()
            for _i, _v in enumerate(self.src_shapes[0]):
                if _i not in self.lhs_contracting_dims:
                    _shape.append(_v)
                else:
                    _shape.append(1)
            _re_shape = list()
            _traversed_elms = list()
            for _v in self.shape:
                if _v in _shape and (_traversed_elms.count(_v) < self.shape.count(_v)):
                    _re_shape.append(_v)
                    _traversed_elms.append(_v)
                else:
                    _re_shape.append(1)
            _reshape_xla_op = xops.Reshape(_reduce_xla_op, _dims, _re_shape)
            
            # Pad
            _padding_value = xops.Constant(builder, np.zeros((), dtype=dtype))
            _padding = list()
            for _i in range(len(self.shape)):
                if _re_shape[_i] == 1:
                    _padding.append((self.shape[_i] - 1, 0, 0))
                else:
                    _padding.append((0, 0, 0))
            _padding_configs = xc.make_padding_config(_padding)    
            self.xla_op = xops.Pad(_reshape_xla_op, _padding_value, _padding_configs) 

        return self.xla_op


class AllReduceOperator(HloOperator):
    """ The class of the all-reduce operator, performing a custom computation across replicas. """
    def __init__(self, bias: str, op_type: str, data_type: str, shape: Sequence[int], 
                 shape_dim_idx: Sequence[int], src_op_bias: str, channel_id: int,
                 replica_groups: Sequence[Sequence[int]]):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)
        self.src_op_bias = src_op_bias                   
        self.channel_id = channel_id                            # Optional channel ID for cross-module communication
        self.replica_groups = replica_groups                    # Groups between which the reductions are performed
    
    def xla_op_expr(self, operand: xc.XlaOp):
        """ Get the expression in format of XLA operator. """
        # Since there should exist at least one unfusable operator (e.g., cudnn-conv) between 
        # adjacent fused computations, we don't need to exploit optimization barrier here to 
        # disable further operator fusion that crosses this communication operator, which may 
        # harm other XLA optimizations (e.g., all-reduce-combiner).
        # Reshape
        _dims = [_i for _i in range(len(self.shape))]           # Default: [0, 1, ..., N]
        self.xla_op = xops.Reshape(operand, _dims, self.shape)

        return self.xla_op


class ReduceScatterOperator(HloOperator):
    """ The class of the reduce-scatter operator, performing a custom computation across replicas. """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 shape: Sequence[int], shape_dim_idx: Sequence[int], src_op_bias_list: Sequence[str],
                 src_data_types: Sequence[str], dst_shapes: Sequence[Sequence[int]], 
                 channel_id: int, replica_groups: Sequence[Sequence[int]], scatter_dimension: int):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)    
        self.src_op_bias_list = src_op_bias_list  
        self.src_data_types = src_data_types     
        self.dst_shapes = dst_shapes
        self.channel_id = channel_id                            
        self.replica_groups = replica_groups                    
        self.scatter_dimension = scatter_dimension                      # Dimension to scatter    

    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        xla_ops = list()
        for _i, _operand in enumerate(operands):
            # Since there should exist at least one unfusable operator (e.g., cudnn-conv) between 
            # adjacent fused computations, we don't need to exploit optimization barrier here to 
            # disable further operator fusion that crosses this communication operator, which may 
            # harm other XLA optimizations (e.g., all-reduce-combiner).
            # Slice
            _xla_op = xops.Slice(_operand, [0 for _ in range(len(self.dst_shapes[_i]))], 
                                 [self.dst_shapes[_i][_j] // 2 for _j in range(self.dst_shapes[_i])], 
                                 strides=[1 for _ in range(len(self.dst_shapes[_i]))])
            xla_ops.append(_xla_op)
        self.xla_op = xla_ops[0] if len(xla_ops) == 1 else xops.Tuple(builder, xla_ops)
        return self.xla_op


class AllGatherOperator(HloOperator):
    """ The class of the all-gather operator, performing concatenation across replicas. """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 shape: Sequence[int], shape_dim_idx: Sequence[int], src_op_bias_list: Sequence[str],
                 src_data_types: Sequence[str], dst_shapes: Sequence[Sequence[int]], 
                 channel_id: int, replica_groups: Sequence[Sequence[int]], all_gather_dimension: int):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)    
        self.src_op_bias_list = src_op_bias_list  
        self.src_data_types = src_data_types     
        self.dst_shapes = dst_shapes
        self.channel_id = channel_id                            
        self.replica_groups = replica_groups                    
        self.all_gather_dimension = all_gather_dimension        # Concatenation dimension

    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        xla_ops = list()
        _replica_num = len(self.replica_groups[0])

        for _i, _operand in enumerate(operands):
            # Since there should exist at least one unfusable operator (e.g., cudnn-conv) between 
            # adjacent fused computations, we don't need to exploit optimization barrier here to 
            # disable further operator fusion that crosses this communication operator, which may 
            # harm other XLA optimizations (e.g., all-reduce-combiner).
            # Padding
            _padding_value = xops.Constant(builder, np.zeros((), 
                                dtype=xc.XLA_ELEMENT_TYPE_TO_DTYPE[self.src_data_types[_i]]))
            _padding = list()
            for _j in range(len(self.dst_shapes[_i])):
                _tmp = (0, 0, 0) if _j != self.all_gather_dimension else (_replica_num - 1, 0, 0)
                _padding.append(_tmp)
            _padding_configs = xc.make_padding_config(_padding)
            _xla_op = xops.Pad(_operand, _padding_value, _padding_configs)
            xla_ops.append(_xla_op)
        self.xla_op = xla_ops[0] if len(xla_ops) == 1 else xops.Tuple(builder, xla_ops)
        return self.xla_op


class AllToAllOperator(HloOperator):
    """ 
    The class of the all-to-all operator, a collective operation that sends data from all 
    cores to all cores. This operator has two phases: Scatter and Gather.
    """
    def __init__(self, bias: str, op_type: str, data_type: str, 
                 shape: Sequence[int], shape_dim_idx: Sequence[int], src_op_bias_list: Sequence[str],
                 dst_shapes: Sequence[Sequence[int]], channel_id: int, replica_groups: Sequence[Sequence[int]], 
                 split_dimension: int, concate_dimension: int):
        super().__init__(bias, op_type, data_type, shape, shape_dim_idx)                 
        self.src_op_bias_list = src_op_bias_list
        self.dst_shapes = dst_shapes
        self.channel_id = channel_id                            
        self.replica_groups = replica_groups                    
        self.split_dimension = split_dimension                  # Split dimension
        self.concate_dimension = concate_dimension              # Concatenation dimension

    def xla_op_expr(self, builder: xc.XlaBuilder, operands: Sequence[xc.XlaOp]):
        """ Get the expression in format of XLA operator. """
        xla_ops = list()
        for _i, _operand in enumerate(operands):
            # Since there should exist at least one unfusable operator (e.g., cudnn-conv) between 
            # adjacent fused computations, we don't need to exploit optimization barrier here to 
            # disable further operator fusion that crosses this communication operator, which may 
            # harm other XLA optimizations (e.g., all-reduce-combiner).
            # Reshape
            _dims = [_i for _i in range(len(self.dst_shapes[_i]))]           # Default: [0, 1, ..., N]
            _xla_op = xops.Reshape(_operand, _dims, self.dst_shapes[_i])
            xla_ops.append(_xla_op)
        self.xla_op = xla_ops[0] if len(xla_ops) == 1 else xops.Tuple(builder, xla_ops)
        return self.xla_op


# Operator bias
operator_bias_table = {
    "convolution": "cudnn-conv",
    "convolution-base-dilated": "cudnn-conv",
    "convolution-window-dilated": "cudnn-conv",
    "cudnn-conv-bw-input": "cudnn-conv",
    "cudnn-conv-bw-filter": "cudnn-conv",
    "cudnn-conv-bias-activation": "cudnn-conv",
    "dot": "cublas-gemm",
    "cublas-batch-gemm": "cublas-gemm",
    "all-reduce-start": "all-reduce"
}

# Mapping from the operator type to the class name
cls_table = {
    # Element-wise
    "compare": ElementWiseOperator,
    "minimum": ElementWiseOperator,
    "maximum": ElementWiseOperator,
    "add": ElementWiseOperator,
    "subtract": ElementWiseOperator,
    "multiply": ElementWiseOperator,
    "divide": ElementWiseOperator,
    "power": ElementWiseOperator,
    "cosine": ElementWiseOperator,
    "exponential": ElementWiseOperator,
    "log": ElementWiseOperator,
    "negate": ElementWiseOperator,
    "rsqrt": ElementWiseOperator,
    "sqrt": ElementWiseOperator,
    "opt-barrier": ElementWiseOperator,
    "and": ElementWiseOperator,
    "or": ElementWiseOperator,
    "select": ElementWiseOperator,
    # Reshape
    "reshape": ReshapingOperator, 
    "concatenate": ReshapingOperator, 
    "slice": ReshapingOperator, 
    "dynamic-slice": ReshapingOperator,
    "pad": ReshapingOperator,
    "reverse": ReshapingOperator,
    # Fusable
    "param": ParamOperator,
    "constant": ConstantOperator,
    "convert": ConvertOperator,
    "transpose": TransposeOperator,
    "bitcast": BitCastOperator,
    "broadcast": BroadcastOperator,
    "reduce": ReduceOperator,
    "reduce-window": ReduceWindowOperator,
    "select-and-scatter": SelectAndScatterOperator,
    "copy": CopyOperator,
    "iota": IotaOperator,
    "tuple": TupleOperator,
    # Fusion hook
    "fusion": FusionHookOperator,
    # Unfusable
    "get-tuple-element": GTEOperator,
    "clamp": ClampOperator,
    "cudnn-conv": CudnnConvOperator,
    "cublas-gemm": CublasGEMMOperator,
    "all-reduce": AllReduceOperator,
    "reduce-scatter": ReduceScatterOperator,
    "all-gather": AllGatherOperator,
    "all-to-all": AllToAllOperator,
    # Useless
    "partition-id": None
}
