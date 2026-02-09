#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
Core implementations of pipeline planner to determine (1) stage partition and (2) GPU allocation 
in two phases with the given cell (i.e., specified GPUs and determined number of stages).
"""

from typing import (
    List, Tuple, Any, Sequence, Dict, Set, Optional)
from functools import (partial, lru_cache)
from collections import namedtuple
from itertools import (
    permutations, count)
import time
import jax.custom_derivatives
import numpy as np
import cvxpy as cp

import jax
from jax import lax
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import JaxprEqn, Var, DropVar, Jaxpr, ClosedJaxpr
from alpa.util import OrderedSet, jaxpr_to_hlo
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call)
from alpa.pipeline_parallel.stage_construction import get_stage_outvars

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell.cell import Cell
from pipeline.roofline import (
    RooflineProfiler, GPU_MEMORY_BW_GB, GPU_PEAK_GFLOPS)
from jaxpr.utils import (
    PipelinePlan, StageShape, AVAIL_MEM_MB_GPU_TYPES, CellConfigs, is_power_of,
)

# Non trival (i.e., heavy) opeartor types
non_trivial_primitive = [lax.dot_general_p, lax.conv_general_dilated_p]
non_trivial_primitive_include_custom = [
    lax.dot_general_p, lax.conv_general_dilated_p, jax.custom_derivatives.custom_jvp_call_p,
]
# Used for storing properties of a jaxpr equation
# str(eqn) -> properties
eqn_prop_cache = {}


class PipelinePlanner:
    """
    The class of pipeline planner to determine (1) stage partition and (2) GPU allocation in two 
    phases with the given cell (i.e., specified GPUs and determined number of stages):

    - Phase 1 (Allocate per-layer GPU fraction) (computation and memory). 

        - For each layer, allocate GPU fraction (e.g., 0.5 GPU) for it proportionally to its computation 
          load ( = `FLOPs` / `attainable performance` from roofline model) (i.e., more computation load, 
          more allocated GPU fraction), with the constraint of each layer's allocated GPU fraction can 
          satisfy its memory access. 

        - Since the allocated GPU quota of each stage follows the power of 2 rule, Phase 1 can lead to
          inter-stage compute-imbalance, i.e., the aggregated GPU fraction of all layers in stage i is
          not equal to the allocated number of GPU for stage i. This introduces an optimization objective,
          which is to minimize the gap (i.e., L2 norm) between aggregated GPU fraction of each stage and
          their allocated GPU quota.

    - Phase 2 (Determine stage boundary) (intra-stage/inter-stages communication). 

        - For intra-stage communication, infer it with varying stages, varying optimizations (e.g., 
          data/tensor parallelism, ZeRO) under varying stage partition plans. While minimizing the per-stage
          intra-stage communication, the optimization goal (denoted as `Minimize[f_a(x)]`) is to minimize 
          the maximum intra-stage communication among all stages.

        - For inter-stages communication, define it as last layer's output activation size divided by 
          min(#GPU-sender, #GPU-receiver) in each stage. The optimization goal (denoted as 
          `Minimize[f_r(x)]`) is to minimize the sum of stages' inter-stages communication and restrict 
          the varience of them.

        - Given the two objectives above, we construct a end-to-end optimization objective based on the 
          common structure of pipeline execution: `Minimize[ f_r(x) + (#micro-batches - 1) * f_a(x) ]`. Note that 
          for simplicity, we ignore the intra-stage comm of other stages and only count for that of the 
          longest stage, since the sum of each stage's intra-stage comm should be constant after the stage
          shape is determined (by choosing the shape with minimal intra-stage comm).
    """

    def __init__(
        self,
        cell: Cell,
        num_micro_batches: int,
        **kwargs,
    ) -> None:
        """ 
        Initialize a two-phase pipeline planner to determine stage partition 
        and GPU allocation for the given cell. 

        Args:
         - `cell`: Cell to be pipeline partitioned and GPU allocated.
         - `num_micro_batches`: Number of micro-batches as the execution data block.
        """

        self.cell = cell
        self.num_micro_batches = num_micro_batches

        # Global record of varying stage partition & stage logical shape
        # Partition plan hashkey -> stage idx -> stage logical shape hashkey -> (aggregated GPU fraction, 
        # allocated GPU num, computation loads, intra per-stage comm, additional comm, inter-stages comm).
        self._global_stage_all_logical_shapes = {}

        self._objective_val_cache = {}
        self._norm_gpu_sharding_cache = {}

        # Other options
        self._enable_mixed_parallel_in_stage = kwargs.get("enable_mixed_parallel_in_stage", False)
    
    ######################################
    #      Layer Analyzing Methods       #
    ######################################

    def _analyze_layer_flops(
        self,
        layers: Sequence[JaxPipelineComputation],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        layer_num: int,
        no_skip: bool = False,
        as_one_hlo: bool = False,
    ) -> List[float]:
        """ 
        Analyze the FLOPs of each layer, including its forward, backward and apply_grad computation. 

        Args:
         - `no_skip`: Do not skip the lightweight operators (eqns) when analyzing in per-eqn style.
         - `as_one_hlo`: Instead of analyzing the cost of each equation in each layer, analyze the 
                         overall cost of each layer by compiling it into one hlo module and profile.
        """

        # Get layer flops
        if not as_one_hlo:
            # Analyze in per-eqn style
            fb_layer_flops = [
                [eqn_flops_func(_eqn, no_skip) for _eqn in _layer.eqns] if _layer else [0]  
                    for _layer in layers
            ]
            apply_grad_layer_flops = [               
                [eqn_flops_func(_eqn, no_skip) for _eqn in _layer.eqns] if _layer else [0]               
                    for _layer in apply_grad_layers           
            ]
        else:
            # Analyze layer as one hlo
            # The result is almost similar to analyzing in per-eqn style, as most flops are 
            # generated by the heavy operators.
            fb_layer_flops = [
                [layer_flops_func(_layer)]
                    for _layer in layers
            ]
            apply_grad_layer_flops = [
                [layer_flops_func(_layer)]
                    for _layer in apply_grad_layers
            ]
        
        # Merge forward, backward and apply_grad flops of the same layer
        layer_flops = [
            fb_layer_flops[_i] +                        # Forward
            fb_layer_flops[2 * layer_num - _i - 1] +    # Backward
            apply_grad_layer_flops[_i]                  # Apply_grad
                for _i in range(layer_num)
        ]

        return layer_flops

    def _analyze_layer_outvar_size(
        self,
        layers: Sequence[JaxPipelineComputation], 
    ) -> List[int]:
        """ 
        Analyze the outvar size of each layer, treating as its output activation size. 
        Since apply_grad computation is not involved with any layer outvar, here only 
        considers forward and backward computation of each layer.

        Args:
         - `layers`: Forward and backward layer computations to be analyzed.
        """

        def __global_outvar_size(outvars: Set[Var], eqn: JaxprEqn):
            """ Get outvar size of one jaxpr eqn. """
            output_vars = {v for v in eqn.outvars if isinstance(v, Var)}
            size = sum((var.aval.size * var.aval.dtype.itemsize)
                    for var in outvars.intersection(output_vars))
            return size

        layer_outvar_size = []
        for layer in layers:
            cost_fn = partial(__global_outvar_size, set(layer.closed_jaxpr().jaxpr.outvars))
            layer_outvar_size.append(
                np.sum([cost_fn(_eqn) for _eqn in layer.eqns], dtype=np.float64)
            )
        
        return layer_outvar_size

    def _analyze_layer_memory_access(
        self,
        layers: Sequence[JaxPipelineComputation],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        layer_num: int,
        no_skip: bool = False,
        as_one_hlo: bool = False,
        include_custom_call: bool = True,
    ) -> int:
        """ 
        Analyze the required memory footprint (i.e., total memory access) of the layer. 

        Args:
         - `no_skip`: Do not skip the lightweight operators (eqns) when analyzing in per-eqn style.
         - `as_one_hlo`: Instead of analyzing the cost of each equation in each layer, analyze the 
                         overall cost of each layer by compiling it into one hlo module and profile.
         - `include_custom_call`: Include `custom_jvp_call_p` into the memory access analysis.
        """

        # Get layer mem access
        if not as_one_hlo:
            # Analyze in per-eqn style
            fb_layer_mems = [
                [eqn_mem_func(_eqn, no_skip, include_custom_call) for _eqn in _layer.eqns] if _layer else [0] 
                    for _layer in layers
            ]
            apply_grad_layer_mems = [                
                [eqn_mem_func(_eqn, no_skip, include_custom_call) for _eqn in _layer.eqns] if _layer else [0]                
                    for _layer in apply_grad_layers            
            ]
        else:
            # Analyze layer as one hlo
            # This is not recommended since it greatly over-estimate by returning the total memory 
            # footprint of the layer, ignoring variable donation.
            fb_layer_mems = [
                [layer_mem_func(_layer)]
                    for _layer in layers
            ]
            apply_grad_layer_mems = [
                [layer_mem_func(_layer)]
                    for _layer in apply_grad_layers
            ]
        
        # Merge forward, backward and apply_grad memory access of the same layer
        layer_mem_access = [
            fb_layer_mems[_i] +                         # Forward
            fb_layer_mems[2 * layer_num - _i - 1] +     # Backward
            apply_grad_layer_mems[_i]                   # Apply_grad
                for _i in range(layer_num)
        ]

        return layer_mem_access
    
    def _analyze_layer_param_memory_access(
        self,
        layers: Sequence[JaxPipelineComputation],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        layer_num: int,
    ) -> int:
        """ 
        Analyze the required memory footprint of the parameter weight of the layer.
        """

        # Get layer mem access
        fb_layer_mems = [
            sum(eqn_weight_mem(_eqn) for _eqn in _layer.eqns) if _layer else 0
                for _layer in layers
        ]
        apply_grad_layer_mems = [            
            sum(eqn_weight_mem(_eqn) for _eqn in _layer.eqns) if _layer else 0            
                for _layer in apply_grad_layers        
        ]
        
        # Merge forward, backward and apply_grad memory access of the same layer
        layer_mem_access = [
            (fb_layer_mems[_i] +                        # Forward
             fb_layer_mems[2 * layer_num - _i - 1] +    # Backward
             apply_grad_layer_mems[_i])                 # Apply_grad
                for _i in range(layer_num)
        ]

        return layer_mem_access
    
    def _analyze_op_outvar_size_in_layer(
        self,
        layers: Sequence[JaxPipelineComputation],
    ) -> int:
        """ 
        Analyze the intermediate outvar size of all operators in the layer. 
        """

        layer_op_outvar_size = []
        for layer in layers:
            # All outvar size of jaxpr equations in the layer
            op_outvar_size = []
            for eqn in layer.eqns:
                if eqn.primitive not in non_trivial_primitive:
                    # Skip lightweight eqn
                    continue

                # All outvars in the equation
                op_outvar_size.append(
                    np.sum([
                        (_v.aval.size * _v.aval.dtype.itemsize) 
                            for _v in eqn.outvars if isinstance(_v, Var)
                    ])
                )
            
            layer_op_outvar_size.append(op_outvar_size)
        
        return layer_op_outvar_size
    
    def _analyze_op_invar_size_in_layer(
        self,
        layers: Sequence[JaxPipelineComputation],
    ) -> int:
        """ 
        Analyze the intermediate invar size of all operators in the layer. 
        """

        layer_op_invar_size = []
        for layer in layers:
            # All invar size of jaxpr equations in the layer
            op_invar_size = []
            for eqn in layer.eqns:
                if eqn.primitive not in non_trivial_primitive:
                    # Skip lightweight eqn
                    continue

                # All invars in the equation
                op_invar_size.append(
                    np.sum([
                        (_v.aval.size * _v.aval.dtype.itemsize) 
                            for _v in eqn.invars if isinstance(_v, Var)
                    ])
                )
            
            layer_op_invar_size.append(op_invar_size)
        
        return layer_op_invar_size

    ######################################
    #        GPU Fraction Methods        #
    ######################################
    
    def _alloc_layer_gpu_fracion(
        self,
        layers: Sequence[JaxPipelineComputation],
        apply_grad_layers: Sequence[JaxPipelineComputation],
        layer_num: int,
        gpu_type: str,
        use_remat: bool = True,
    ) -> Tuple[Optional[List[float]], List[float], List[int], int]:
        """
        Allocate per-layer GPU fraction.

        For each layer, allocate GPU fraction (e.g., 0.5 GPU) for it proportionally to its computation 
        load ( = `FLOPs` / `attainable performance` from roofline model) (i.e., more computation load, 
        more allocated GPU fraction), with the constraint of each layer's allocated GPU fraction can 
        satisfy its memory access. 

        Args:
         - `layers`: Forward and backward layer computations.
         - `apply_grad_layers`: Apply_grad layer computations.
         - `layer_num`: Number of forward layers.
         - `gpu_type`: Type of the allocated GPU.
         - `use_remat`: Use rematerialization (i.e., gradient checkpointing) in model training.
        """

        # Layer flops
        layer_flops = self._analyze_layer_flops(layers, apply_grad_layers, layer_num) 
        # layer_flops = [np.sum(_l) / 1024**3 for _l in layer_flops]
        layer_flops = [
            [
                _l / 1024**3 for _l in layer_flops[_i]
            ] for _i in range(layer_num)
        ]
        
        # Layer memory access
        if use_remat:
            # Layer memory accesses of all heavy operators (e.g., gemm, conv)
            # As the remat is enabled, most intermediate tensors will not be stored. In this case, we only 
            # use the memory access of heavy operators (which are commonly not consecutive to each other) 
            # to estimate the global memory footprint of the stage. 
            layer_mem_access = self._analyze_layer_memory_access(layers, apply_grad_layers, layer_num)
        else:
            # Layer memory accesses of all operators
            # In this case, the intermediate tensors should be stored for backward computation. We use the 
            # memory access on weight and output of all operators, since each operator can be consecutive 
            # and the intermediate tensors are shared both by its producer and consumer operators.
            layer_mem_access = self._analyze_layer_memory_access(layers, apply_grad_layers, layer_num, no_skip=True)
        
        layer_mem_access_heavy_op = self._analyze_layer_memory_access(
            layers, apply_grad_layers, layer_num, include_custom_call=False,
        )
        layer_mem_access_heavy_op = [np.sum(_l) / 1024**3 for _l in layer_mem_access_heavy_op]

        # layer_mem_access = [np.sum(_l) / 1024**3 for _l in layer_mem_access]
        layer_mem_access = [
            [
                _l / 1024**3 for _l in layer_mem_access[_i]
            ] for _i in range(layer_num)
        ]

        # layer_flops_all = self._analyze_layer_flops(layers, apply_grad_layers, layer_num) 
        # layer_mem_access_all = self._analyze_layer_memory_access(layers, apply_grad_layers, layer_num)

        # for i in range(layer_num):
        #     print("")
        #     print(f"Layer {i}:")
        #     assert len(layer_flops_all[i]) == len(layer_mem_access_all[i])
        #     num_eqns = len(layer_flops_all[i])

        #     for j in range(num_eqns):
        #         if j < len(layers[i].eqns):
        #             eqn_primitive = layers[i].eqns[j].primitive
        #         elif j < len(layers[i].eqns) + len(layers[2 * layer_num - i - 1].eqns):
        #             eqn_primitive = layers[2 * layer_num - i - 1].eqns[j - len(layers[i].eqns)].primitive
        #         else:
        #             eqn_primitive = apply_grad_layers[i].eqns[j - len(layers[i].eqns) - len(layers[2 * layer_num - i - 1].eqns)]

        #         eqn_flops_gb = layer_flops_all[i][j] / 1024**3
        #         eqn_mem_access_gb = layer_mem_access_all[i][j] / 1024**3
        #         if eqn_flops_gb > 0.01 or eqn_mem_access_gb > 0.01:
        #             print(f"Eqn {j} in layer {i}: Primitive: {eqn_primitive} | FLOPs (GB): {eqn_flops_gb} | " + 
        #                 f"Memory access (GB): {eqn_mem_access_gb}")

        # exit(0)

        # for i in range(layer_num):
        #     print(f"Layer {i} FLOPs (GB): {layer_flops[i] / 1024**3}")
        #     print(f"Layer {i} memory access (GB): {layer_mem_access[i] / 1024**3}")
        #     print(f"Layer {i} eqn memory access (GB): {layer_eqn_mem_access[i] / 1024**3}")
        #     print("")

        # exit(0)

        def __get_computation_load(flops_gb: float, mem_access_gb: float, attn_perf: int):
            """ 
            Get the computation load with the specified FLOPs, memory access and attainable performance. 

            If the attainable performance is 0 (i.e., too small FLOPs), we use `memory access / maximum 
            GPU memory bandwidth` to estimate the computation load. 
            """
            return (flops_gb / attn_perf) if attn_perf > 0 else (mem_access_gb / GPU_MEMORY_BW_GB[gpu_type])

        # Roofline model
        assert gpu_type in GPU_MEMORY_BW_GB, \
            f"Mimatched GPU type '{gpu_type}' in the lookup table of " + \
            f"memory banwdith (keys: {GPU_MEMORY_BW_GB.keys()})."
        roofline_profiler = RooflineProfiler(gpu_type, GPU_MEMORY_BW_GB[gpu_type], GPU_PEAK_GFLOPS[gpu_type])
        # Attainable performance of each layer 
        # The roofline model is commonly used to evaluate the performance from application-level.
        # layer_attn_perfs = [
        #     roofline_profiler.query(
        #         gflops=_g,
        #         memory_access_gb=_m,
        #     ) for (_g, _m) in zip(layer_flops, layer_mem_access)
        # ]
        layer_attn_perfs = [
            [
                roofline_profiler.query(
                    gflops=_g, memory_access_gb=_m,
                ) for (_g, _m) in zip(layer_flops[i], layer_mem_access[i])
            ] for i in range(layer_num)
        ]

        for i in range(layer_num):
            print("")
            print(f"Layer {i}:")
            assert len(layer_flops[i]) == len(layer_mem_access[i])
            num_eqns = len(layer_flops[i])
            layer_comp_load = 0

            for j in range(num_eqns):
                if j < len(layers[i].eqns):
                    eqn_primitive = layers[i].eqns[j].primitive
                elif j < len(layers[i].eqns) + len(layers[2 * layer_num - i - 1].eqns):
                    eqn_primitive = layers[2 * layer_num - i - 1].eqns[j - len(layers[i].eqns)].primitive
                else:
                    eqn_primitive = apply_grad_layers[i].eqns[j - len(layers[i].eqns) - len(layers[2 * layer_num - i - 1].eqns)]

                eqn_flops_gb = layer_flops[i][j]
                eqn_mem_access_gb = layer_mem_access[i][j]
                attn_perf = layer_attn_perfs[i][j]
                comp_load = __get_computation_load(eqn_flops_gb, eqn_mem_access_gb, attn_perf)
                if eqn_flops_gb > 0.01 or eqn_mem_access_gb > 0.01:
                    print(f"Eqn {j} in layer {i}: Primitive: {eqn_primitive} | FLOPs (GB): {eqn_flops_gb} | " + 
                        f"Memory access (GB): {eqn_mem_access_gb} | Attainable performance: {attn_perf} | Computation load: {comp_load}")
                    layer_comp_load += comp_load

            print("Layer computation load:", layer_comp_load)

        # exit(0)

        # for i in range(layer_num):
        #     print(f"Layer {i} Flops:", layer_flops[i])
        #     print(f"Layer {i} attanable performance:", layer_attn_perfs[i])
        #     print(f"Layer {i} computation load:", layer_flops[i] / layer_attn_perfs[i])
        #     print("")
        
        # exit(0)

        # Computation load
        # layer_comp_loads = [
        #     _flops / _attn_perf 
        #         for (_flops, _attn_perf) in zip(layer_flops, layer_attn_perfs)
        #             if (_flops > 0.0 and _attn_perf > 0)
        # ]
        layer_comp_loads = [
            np.sum([
                __get_computation_load(_flops, _mem_access, _attn_perf)
                    for (_flops, _mem_access, _attn_perf) in zip(layer_flops[i], layer_mem_access[i], layer_attn_perfs[i]) 
            ]) for i in range(layer_num)
        ]

        # Factors
        if self.cell.model_name == "wide_resnet":
            # scale_factor = 0.15       # For 4-GPU
            scale_factor = 0.5          # For 8-GPU
        elif self.cell.model_name == "bert" or self.cell.model_name == "moe":
            scale_factor = 0.0

        # TODO(chunyu): An array of the scale factors for the computation load of each layer. The further back the layer is, 
        #               the larger scale factor it is given.
        comp_load_scale_factors = [1.0 + _i * scale_factor for _i in range(layer_num)]
        layer_comp_loads = [_comp_load * _f for (_comp_load, _f) in zip(layer_comp_loads, comp_load_scale_factors)]

        # print("")
        # for i in range(layer_num):
        #     print(layer_comp_loads[i])

        # print("")
        
        # Available memory
        assert self.cell.gpu_type in AVAIL_MEM_MB_GPU_TYPES, \
            f"GPU type '{self.cell.gpu_type}' is not recorded in available memory dict, " + \
            f"whose keys include: {AVAIL_MEM_MB_GPU_TYPES.keys()}"
        # Usually pre-allocate 80% memory for xla computation
        avail_mem_bytes = int(AVAIL_MEM_MB_GPU_TYPES[self.cell.gpu_type] * 1024**2 * 0.8)
        # Available gpu num
        num_gpus = self.cell.num_gpus
        # Per-layer memory access
        layer_mem_access = [np.sum(_l) for _l in layer_mem_access]

        print("Layer memory access:", layer_mem_access_heavy_op)

        layer_gpu_fractions = self._optimize_gpu_fraction(
            layer_comp_loads, layer_mem_access_heavy_op, avail_mem_bytes, num_gpus,
        )

        print("\nLayer GPU fractions:")
        print(layer_gpu_fractions)

        # exit(0)

        return layer_gpu_fractions, layer_comp_loads, layer_mem_access_heavy_op, avail_mem_bytes

    def _optimize_gpu_fraction(
        self,
        layer_comp_loads: List[float],
        layer_mem_access: List[int],
        avail_mem_bytes: int,
        num_gpus: int,
    ) -> Optional[List[float]]:
        """ 
        Optimize the allocation vector of per-layer GPU fraction. 

        (for dev) By manipulating layer mem access, user can manually specify the minimal allocated 
        GPU fraction for any layer. For example, if user sets `layer_mem_access[0] = 2 * avail_mem_bytes`,
        the planner will allocate at least fraction of 2 GPUs for the first layer.
        """

        # TODO(chunyu): Cancel constraint of memory access and directly equals layer gpu fraction to its computation load, 
        #               instead of using solver to optimize GPU fraction.

        # Normalize
        norm_loads = np.array([_load / np.sum(layer_comp_loads) for _load in layer_comp_loads])
        return [round(_v * num_gpus, 3) for _v in list(norm_loads)]
        
        # layer_mem_access = np.array(layer_mem_access)

        # # Fraction vector
        # x = cp.Variable(norm_loads.shape)
        # # Objective function
        # objective = cp.Minimize(cp.norm(x - norm_loads, 2))
        # # Constraints
        # constraints = []
        # constraints.append(cp.sum(x) == 1)  # Normalized fraction vector
        # constraints.append(
        #     layer_mem_access <= avail_mem_bytes * num_gpus * x
        # ) # Each layer's memory access must be satisfied

        # # Problem
        # cvxprob = cp.Problem(objective, constraints)
        # # Solve
        # _ = cvxprob.solve(solver=cp.CPLEX, verbose=False)

        # print(f"[TMP] Optimization status of GPU fraction: {cvxprob.status}")

        # if cvxprob.status != "optimal":
        #     return None

        # return [round(_v * num_gpus, 3) for _v in list(x.value)]

    ######################################
    #       Stage Partition Methods      #
    ######################################

    def cluster_layers_into_stages(
        self,
        strategy: str,
        layers: Sequence[JaxPipelineComputation], 
        apply_grad_layers: Sequence[JaxPipelineComputation],
        cell: Cell, 
        submesh_logical_shapes: Sequence[Sequence[int]],
        layer_num: int, 
        acc_grad_outvars: Sequence[Any], 
        accumulator_mapping: Dict[Var, Var],
    ) -> List[JaxPipelineComputation]:
        """ 
        Cluster forward and backward layers into pipeline stages. 

        Args:
         - `strategy`: The strategy of pipeline partition. Options: ["uniform", "auto"]. If set to "uniform", 
                       uniformly (i.e., proportional to the user-specified parallelism) slice forward and 
                       backward layers into pipeline stages; if set to "auto", generate pipeline plan by 
                       automatically slicing forward and backward layers into pipeline stages based on layer 
                       flops and minimizing cross-stages communication size. TODO(chunyu): Refine this.
         - `layers`: Forward and backward layer computations to be analyzed.
         - `apply_grad_layers`: Apply_grad layer computations to be analyzed.
         - `cell`: Cell to be pipeline partitioned and GPU allocated.
         - `submesh_logical_shapes`: List of intra-stage parallel plan for each stage (i.e., parallelism).
         - `layer_num`:  Number of forward layers to be pipeline-partitioned.
         - `acc_grad_outvars`: List of outvars.
         - `accumulator_mapping`: donation map of merged jaxpr, may have redundant items.
        """

        assert cell.pipeline_plan is None, \
            "This cell has been pipeline-partitioned."
        cell.pipeline_strategy = strategy

        if os.environ.get("CELL_FORCE_PLAN_SHAPE_HASHKEY", "none") != "none":
            # Forcibly set pipeline plan and cluster layers into stages
            return self._forcibly_cluster_layers_into_stages(
                layers, cell, layer_num, acc_grad_outvars, accumulator_mapping,
            )

        if strategy == "uniform":
            return self._uniform_cluster_layers_into_stages(
                layers, cell, submesh_logical_shapes, layer_num, acc_grad_outvars, accumulator_mapping,
            )
        elif strategy == "auto":
            return self._auto_cluster_layers_into_stages(
                layers, apply_grad_layers, cell, layer_num, acc_grad_outvars, accumulator_mapping,
            )
        else:
            raise ValueError(f"Invalid pipeline partition strategy: {strategy}")

    def _auto_cluster_layers_into_stages(
        self,
        layers: Sequence[JaxPipelineComputation], 
        apply_grad_layers: Sequence[JaxPipelineComputation],
        cell: Cell, 
        layer_num: int, 
        acc_grad_outvars: Sequence[Any], 
        accumulator_mapping: Dict[Var, Var],
    ) -> List[JaxPipelineComputation]:
        """ 
        Automatically cluster forward and backward layers into pipeline stages. The principles 
        and workflow are described in docstrings of `PipelinePlanner` class. 
        """

        # Allocate per-layer gpu fraction
        (layer_gpu_fractions, 
         layer_comp_loads,
         layer_mem_access, 
         avail_mem_bytes) = self._alloc_layer_gpu_fracion(layers, apply_grad_layers, layer_num, cell.gpu_type)
        assert layer_gpu_fractions is not None, \
            "The GPU fraction of the cell cannot be optimized."
        
        print("\nLayer GPU fractions:")
        print(layer_gpu_fractions)
        print("")
        
        # Get layer outvar size
        layer_outvar_size = self._analyze_layer_outvar_size(layers)
        # Get operator in/outvar size in layers
        layer_op_invar_size = self._analyze_op_invar_size_in_layer(layers)
        layer_op_outvar_size = self._analyze_op_outvar_size_in_layer(layers)
        # Get layer paramter size
        layer_param_size = self._analyze_layer_param_memory_access(layers, apply_grad_layers, layer_num)

        # Generate auto pipeline plan for cell
        self._gen_auto_pipeline_plan(
            cell=cell,
            layer_gpu_fractions=layer_gpu_fractions,
            layer_comp_loads=layer_comp_loads,
            layer_outvar_size=layer_outvar_size,
            layer_op_invar_size=layer_op_invar_size,
            layer_op_outvar_size=layer_op_outvar_size,
            layer_param_size=layer_param_size,
            layer_mem_access=layer_mem_access,
            avail_mem_bytes=avail_mem_bytes,
        )
        
        print(f"[I] Pipeline partition mode: auto | Layer-to-stage partition: " + 
            f"{cell.pipeline_plan.stage_layer_ids[:cell.num_stages]}")
        
        return self._gen_merged_stage_computations(
            layers, cell, acc_grad_outvars, accumulator_mapping,
        )

    def _uniform_cluster_layers_into_stages(
        self, 
        layers: Sequence[JaxPipelineComputation],
        cell: Cell, 
        submesh_logical_shapes: Sequence[Sequence[int]],
        layer_num: int, 
        acc_grad_outvars: Sequence[Any], 
        accumulator_mapping: Dict[Var, Var],
    ) -> List[JaxPipelineComputation]:
        """ 
        Uniformly cluster forward and backward layers into pipeline stages. If the user-specified 
        parallelism is asymmetric, generate clustering plan by slicing proportionally based on it.
        Only consider forward and backward stages, apply_grad will be considered later.

        Args:
         - `submesh_logical_shapes`: List of user-specified parallelism for each stage.
         -  Other arguments: described in docstrings of `auto_cluster_layers_into_stages()`.
        """

        # Generate auto pipeline plan for cell
        self._gen_uniform_pipeline_plan(
            cell=cell,
            num_layers=layer_num,
            submesh_logical_shapes=submesh_logical_shapes,
        )

        print(f"[I] Pipeline partition mode: uniform | Layer-to-stage partition: " + 
            f"{cell.pipeline_plan.stage_layer_ids[:cell.num_stages]}")
        
        return self._gen_merged_stage_computations(
            layers, cell, acc_grad_outvars, accumulator_mapping,
        )

    def _forcibly_cluster_layers_into_stages(
        self, 
        layers: Sequence[JaxPipelineComputation], 
        cell: Cell, 
        layer_num: int,
        acc_grad_outvars: Sequence[Any], 
        accumulator_mapping: Dict[Var, Var],
    ) -> None:
        """ Forcibly set the pipeline plan and cluster layers into stages. """

        plan_hashkey = os.environ.get("CELL_FORCE_PLAN_SHAPE_HASHKEY", "none").split("::")[0]
        stage_layer_ids = Cell.gen_hashkey_with_partition_plan(plan_hashkey=plan_hashkey, decode=True)
        shape_hashkeys = os.environ.get("CELL_FORCE_PLAN_SHAPE_HASHKEY", "none").split("::")[1].split("__")
        norm_sharding = [
            np.prod(Cell.gen_hashkey_with_stage_shape(shape_hashkey=_k, decode=True).stage_shape)
                for _k in shape_hashkeys
        ]
        # Backward stage layer ids
        backward_stage_layer_ids = [
            [
                2 * layer_num - 1 - _i 
                    for _i in reversed(_layer_ids)
            ] for _layer_ids in reversed(stage_layer_ids)
        ]
        # Combine forward and backward layer ids
        stage_layer_ids = stage_layer_ids + backward_stage_layer_ids
        # Stage idx -> mesh idx
        stage_to_mesh = list(range(cell.num_stages)) + list(reversed(range(cell.num_stages)))

        # Construct pipeline plan
        cell.pipeline_plan = PipelinePlan(stage_layer_ids, stage_to_mesh, norm_sharding)

        print(f"[I] Pipeline partition mode: forced | Layer-to-stage partition: " + 
            f"{cell.pipeline_plan.stage_layer_ids[:cell.num_stages]}")
        
        return self._gen_merged_stage_computations(
            layers, cell, acc_grad_outvars, accumulator_mapping,
        )

    def _gen_uniform_pipeline_plan(
        self, 
        cell: Cell,
        num_layers: int, 
        submesh_logical_shapes: Sequence[Sequence[int]] = None,
    ) -> None:
        """ 
        Generate pipeline plan by uniformly slicing forward and backward layers into pipeline stages. If the 
        user-specified parallelism is asymmetric, generate layer clustering plan by slicing proportionally 
        based on it.
        """
        
        assert submesh_logical_shapes is not None, \
            "`submesh_logical_shapes` must be set when perform " + \
            "uniform pipeline slicing."
        assert (num_layers > cell.num_gpus and 
                num_layers % cell.num_gpus == 0), \
            f"Layer num ({num_layers}) should be greater than and " + \
            f"divisible to GPU num ({cell.num_gpus})."

        # Layer ids in each stage
        # Proportionally cluster layers based on the generated
        # parallelism of each stage.
        num_layers_per_gpu = num_layers // cell.num_gpus
        num_layers_per_stage_list = [
            num_layers_per_gpu * np.prod(_submesh) 
                for _submesh in submesh_logical_shapes
        ]
        forward_stage_layer_ids = [
            [
                _i + _j * num_layers_per_stage_list[_j]
                    for _i in range(num_layers_per_stage_list[_j])
            ] for _j in range(cell.num_stages)
        ]
        backward_stage_layer_ids = [
            [
                2 * num_layers - 1 - _i for _i in reversed(layer_ids)
            ] for layer_ids in reversed(forward_stage_layer_ids)
        ]
        stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
        # Mapping from stage to mesh
        stage_to_mesh = list(range(cell.num_stages)) + list(reversed(range(cell.num_stages)))
        # Sharded GPU num of each pipeline stage
        gpu_sharding = [np.prod(_submesh) for _submesh in submesh_logical_shapes]

        # Construct pipeline plan
        cell.pipeline_plan = PipelinePlan(stage_layer_ids, stage_to_mesh, gpu_sharding)
    
    def _gen_auto_pipeline_plan(
        self, 
        cell: Cell,
        layer_gpu_fractions: List[float],
        layer_comp_loads: List[float],
        layer_outvar_size: List[int],
        layer_op_invar_size: List[List[int]],
        layer_op_outvar_size: List[List[int]],
        layer_param_size: List[int],
        layer_mem_access: List[int],
        avail_mem_bytes: int,
        overlap_grad_sync_with_bp: bool = False,
        _lambda: float = 0.5,
        _eta: float = 0.9,
        _gamma: float = 1.1,
    ) -> None:
        """
        Generate pipeline plan by automatically slicing forward and backward layers into pipeline stages
        with the optimization objective of (1) minimize max intra-stage communication and (2) minimize 
        the sum of inter-stages communication and their varience. 
        
        The principles and workflow are described in "Phase 2" in docstrings of `PipelinePlanner` class. 

        We use the following workflow to obtain the optimal plan as the target parallelism to be profiled 
        and used in cluster-level job scheduling:

        Workflow:

         - Step 1. Enumerate all candidate stage partition plans. We sort them with the l2 norm in the GPU
                   sharding normalization (i.e., the gap between sum GPU fraction and actually allocated 
                   GPU quota, selecting the top _eta% best-performing plans on l2 norm.

         - Step 2. Infer the intra-stage comm for each stage of varying plans, including (1) per-stage 
                   (aggregated) communication and (2) additional communication (for grad sync). We choose 
                   the minimal intra-stage comm of varying intra-stage parallelizations (e.g., data/tensor 
                   parallelism, ZeRO) for each stage, under the constraint of memory access.
        
         - Step 3. Optimize with the objective of minimizing e2e communication as described in docstrings of
                   `PipelinePlanner` class. We select the top _eta% best-performing plans on communication.
        
         - Step 4. Finally, we select the most compute-balance one in all candidate partition plans
                   from Step 3, i.e., the aggregated GPU fraction of each stage is the nearest to 
                   the actually allocated GPU quota (i.e., minimal l2 norm in the GPU sharding 
                   normalization).

        Args:
         - `layer_gpu_fractions`: List of per-layer GPU fractions determined by the pipeline planner.
         - `layer_comp_loads`: List of per-layer computation loads.
         - `layer_outvar_size`: List of layer outvar size (i.e., inter-layer communication size) for each 
                                layer.
         - `layer_op_invar_size`: List of operator invar size for each layer.
         - `layer_op_outvar_size`: List of operator outvar size for each layer.
         - `layer_param_size`: List of operator parameter size for each layer.
         - `layer_mem_access`: List of memory access amount (in bytes) for each layer.
         - `avail_mem_bytes`: Availble memory (in bytes) on the single GPU.
         - `overlap_grad_sync_with_bp`: For data parallelism, overlap the gradient sync of stage i + 1 
                                        with the backward computation of stage i.
        """

        # Candidate gpu shardings w.r.t. the power of 2 rule
        cand_shardings = _enum_cand_shardings(cell.num_gpus, cell.num_stages)

        # --------- Step 1. Enumerate candidate partition plans ---------

        cand_plans = _enum_all_partition_plans_dp(
            num_layers=len(layer_gpu_fractions), 
            num_stages=cell.num_stages,
            layer_gpu_fractions=layer_gpu_fractions,
            min_layers_per_stage=cell.cell_cfgs.min_layers_per_stage,
            only_symmetric_sharding=cell.cell_cfgs.only_symmetric_sharding,
        )

        print(cand_plans)
        print("")

        if os.environ.get("ENUM_ALL_PARALLELISM", "false") == "true":
            # Enumerate all parallelism, save and exit the program
            _enum_all_parallelism_and_save(
                cand_plans, layer_gpu_fractions, cell.num_gpus, cell.num_stages, cell.cell_cfgs,
            )

        def __cal_l2_norm(partition_plan: List[List[int]]):
            """ 
            Calculate the l2 normalization on the GPU fraction of all stages (i.e., the similarity of the
            aggregated GPU fraction and the actually allocated quota). 
            """
            _, l2_norm = _normalize_gpu_sharding(
                partition_plan, layer_gpu_fractions, cand_shardings, self._norm_gpu_sharding_cache,
            )
            return l2_norm

        # Sort by l2 norm
        if len(cand_plans) > 0:
            plans = sorted(cand_plans, key=__cal_l2_norm)
            # num_selected = max(int(len(plans) * _eta), 1)
            # cand_plans = plans[:num_selected]
            cand_plans = []
            best_val = __cal_l2_norm(plans[0])
            for plan in plans:
                if best_val / __cal_l2_norm(plan) >= (1 - _eta):
                    cand_plans.append(plan)
        
        print(cand_plans)
        print("")

        # --------- Step 2. Infer intra-stage communication ---------

        def __infer_intra_stage_objective_val(partition_plan: List[List[int]]):
            """ 
            Infer the intra-stage comm for each stage of the specified partition plan, choose the minimal 
            intra-stage comm of varying intra-stage parallelizations (e.g., data/tensor parallelism, ZeRO) 
            for each stage, under the constraint of memory access.

            For data parallelism, no matter how the stage partition is, the final gradient synchronization
            happens on all data parallelized stages (after the completion of all backward passes). If 
            supported, the gradient synchronization can be overlapped with the backward pass (i.e., 
            synchronize gradients of stage i + 1 when backwarding on stage i). Thus the communication of 
            these data parallelized stages should be added.

            For tensor parallelism, the intra-stage communication in every micro-batch during the forward
            pass, so the communication of these tensor parallelized stages should be seperately considered.

            Therefore, we use `all_stage_comm_tp` to evaluate the per-stage communication introduced by 
            tensor parallelism, and use `all_stage_comm_dp / #micro-batches` to evaluate that introduced by
            data parallelism. 
            
            The optimization goal is to minimize `f_a(x)`, where `f_a(x)` is defined as the maximum (aggregated) 
            per-stage communication of varying intra-stage parallelizations, plus the additional grad_sync 
            communication (from data parallelism if not overlapped).
            """
            # Normalized gpu sharding
            norm_sharding, _ = _normalize_gpu_sharding(
                partition_plan, layer_gpu_fractions, cand_shardings, self._norm_gpu_sharding_cache,
            )
            # Infer logical shapes and intra-stage communication
            (stage_logical_shapes, 
             all_per_stage_comm, 
             all_addn_comm) = self._infer_min_intra_stage_comm_fixed_plan(
                partition_plan, norm_sharding, layer_outvar_size, layer_op_invar_size, layer_op_outvar_size, 
                layer_param_size, layer_mem_access, layer_gpu_fractions, layer_comp_loads, avail_mem_bytes, 
                overlap_grad_sync_with_bp,
            )

            reshard_penalty = 0
            for (i, stage_shape) in enumerate(stage_logical_shapes):
                if i > 0 and stage_shape.stage_shape != stage_logical_shapes[i - 1].stage_shape:
                    # Penalty of cross-stages resharding
                    reshard_penalty += (all_per_stage_comm[i] + all_per_stage_comm[i]) * (_gamma - 1)

            return max(all_per_stage_comm) + np.sum(all_addn_comm) + reshard_penalty
        
        # --------- Step 2.5. Formulate inter-stages communication ---------

        def __cal_inter_stage_objective_val(partition_plan: List[List[int]]):
            """ Calculate the sum of inter-stages communication of all stages plus their standard deviation. """
            # All candidate partition plans must normalize their gpu fraction allocation
            # to find the nearest gpu sharding in these candidate ones.
            norm_sharding, _ = _normalize_gpu_sharding(
                partition_plan, layer_gpu_fractions, cand_shardings, self._norm_gpu_sharding_cache,
            )
            # Inter-stages communication
            inter_stages_comms = self._cal_inter_stages_comms(partition_plan, norm_sharding, layer_outvar_size)

            # return np.sum(inter_stages_comms) + _lambda * np.std(inter_stages_comms[:-1]) \
            #         if len(inter_stages_comms) > 1 else 0

            # TODO(chunyu): Currently we do not consider inter-stages communication in parallelism determination.

            return 0
        
        # --------- Step 3. Optimize end-to-end communication ---------

        def __cal_e2e_comm_objective_val(partition_plan: List[List[int]]):
            """ Calculate the end-to-end communication based on the common pipeline structure. """
            # Cache
            plan_hashkey = Cell.gen_hashkey_with_partition_plan(partition_plan)
            if plan_hashkey in self._objective_val_cache:
                return self._objective_val_cache[plan_hashkey]
    
            # Objective: Minimize[ f_r(x) + (#micro-batches - 1) * f_a(x) ]
            # objective_val = (
            #     __cal_inter_stage_objective_val(partition_plan) 
            #     + 
            #     (self.num_micro_batches - 1) * __infer_intra_stage_objective_val(partition_plan)
            # )

            # TODO(chunyu): Currently only consider intra-stage communication in parallelism determination.
            objective_val = __infer_intra_stage_objective_val(partition_plan)

            self._objective_val_cache[plan_hashkey] = objective_val

            return objective_val
        
        # if os.environ.get("CELL_FORCE_PLAN_SHAPE_HASHKEY", "none") != "none":
        #     # In case that the forcibly specified parallelism is pruned
        #     _eta = 0

        # Sort by objective val for intra-stage communication
        if len(cand_plans) > 0:
            plans = sorted(cand_plans, key=__cal_e2e_comm_objective_val)
            # Select best performing plans
            # num_selected = max(int(len(plans) * _eta), 1)
            # cand_plans = plans[:num_selected]
            cand_plans = []
            best_val = __cal_e2e_comm_objective_val(plans[0])
            for plan in plans:
                if best_val == 0 or best_val / __cal_e2e_comm_objective_val(plan) >= (1 - _eta):
                    cand_plans.append(plan)

        # -------- Step 4. Selecte the most compute-balance plan -------
        
        # TODO(chunyu): When determining the estimated best performing one, maybe can use the (1) communication 
        #               amount and offline profiled communication data, and (2) stage flops and roofline 
        #               attainable performance to compare in quite coarse-grained level. Specifically, construct
        #               a threshold to represent the bottleneck on computation or communication for the plan.

        # Sort by l2 norm
        if len(cand_plans) > 0:
            plans = sorted(cand_plans, key=__cal_l2_norm)
            # Select the best performing one
            stage_layer_ids = plans[0]

            for plan in plans:
                print(plan)
                print(__cal_e2e_comm_objective_val(plan))
                print(__cal_l2_norm(plan))
                print(self._norm_gpu_sharding_cache[Cell.gen_hashkey_with_partition_plan(plan)])
                print("")

        norm_sharding, _ = _normalize_gpu_sharding(
            stage_layer_ids, layer_gpu_fractions, cand_shardings, self._norm_gpu_sharding_cache,
            cell.cell_cfgs.only_symmetric_sharding,
        )
        # Backward stage layer ids
        backward_stage_layer_ids = [
            [
                2 * len(layer_gpu_fractions) - 1 - _i 
                    for _i in reversed(_layer_ids)
            ] for _layer_ids in reversed(stage_layer_ids)
        ]
        # Combine forward and backward layer ids
        stage_layer_ids = stage_layer_ids + backward_stage_layer_ids
        # Stage idx -> mesh idx
        stage_to_mesh = list(range(cell.num_stages)) + list(reversed(range(cell.num_stages)))

        # Construct pipeline plan
        cell.pipeline_plan = PipelinePlan(stage_layer_ids, stage_to_mesh, norm_sharding)
        # Set hints for cell
        cell.set_parallel_plan_hints(self._global_stage_all_logical_shapes)

    def _gen_merged_stage_computations(
        self,
        layers: Sequence[JaxPipelineComputation],
        cell: Cell, 
        acc_grad_outvars: Sequence[Any], 
        accumulator_mapping: Dict[Var, Var],
    ) -> List[JaxPipelineComputation]:
        """ Generate stage computations merged by cell's pipeline plan. """

        # Outvars of each stage
        stage_outvars = get_stage_outvars(
            layers, 
            cell.pipeline_plan.stage_layer_ids, 
            acc_grad_outvars,
        )
        # Cluster layers
        merged_stages = []
        for stage_id, layer_ids in enumerate(cell.pipeline_plan.stage_layer_ids):
            if len(layer_ids) == 1:
                # Single-layer stage
                merged_stages.append(layers[layer_ids[0]])
                continue
            # Multi-layers stage
            stage_layer_jaxprs = [layers[i].closed_jaxpr() for i in layer_ids]
            stage_name = str(stage_id)
            merged_stage_jaxpr = merge_marked_jaxprs_with_named_call(
                stage_layer_jaxprs,
                stage_outvars[stage_id],
                accumulator_mapping,
                stage_name,
                wrap_with_marker=True
            )
            merged_stage = JaxPipelineComputation.from_closed_jaxpr(
                stage_name, merged_stage_jaxpr,
            )
            merged_stages.append(merged_stage)
        
        return merged_stages

    def _cal_inter_stages_comms(
        self, 
        partition_plan: List[List[int]], 
        sharding: List[int],
        layer_outvar_size: Sequence[int],
    ) -> List[float]:
        """ 
        For a partition plan, calculate the inter-stages communication of all stages. 
        The inter-stages communication of a stage is defined by last layer's output size 
        / min(#GPU-sender, #GPU-receiver). 
        """

        inter_stages_comms = []
        # First n - 1 stages
        for i in range(len(partition_plan) - 1):
            outvar_size = layer_outvar_size[partition_plan[i][-1]]
            num_gpus_send = sharding[i]
            num_gpus_recv = sharding[i + 1]
            inter_stages_comms.append(
                outvar_size / min(num_gpus_send, num_gpus_recv)
            )
        # Last stage
        inter_stages_comms.append(0)

        return inter_stages_comms

    def _enum_stage_logical_shapes_fixed_plan(
        self,
        partition_plan: List[List[int]],
        sharding: List[int],
    ) -> List[List[Tuple[int]]]:
        """ 
        Enumerate all candidate intra-stage parallelizations for stages in the specified plan.
        """
        
        stage_logical_shapes = []

        # Case 1. Data, tensor parallelism and their hybrid combinations.
        for (stage, num_gpus) in zip(partition_plan, sharding):
            stage_logical_shapes.append(
                _enum_logical_shapes_one_stage(num_gpus)
            )
        
        # Case 2. Layer-granularity mixed optimizations within one stage.
        if self._enable_mixed_parallel_in_stage:
            for i in range(len(stage_logical_shapes)):
                stage_logical_shapes[i].append(StageShape(
                    type="mixed",
                    stage_shape=None,
                    layer_shape=None,   # To be inferred later
                ))

        return stage_logical_shapes

    def _infer_min_intra_stage_comm_fixed_plan(
        self,
        partition_plan: List[List[int]],
        sharding: List[int],
        layer_outvar_size: List[int],
        layer_op_invar_size: List[List[int]],
        layer_op_outvar_size: List[List[int]],
        layer_param_size: List[int],
        layer_mem_access: List[int],
        layer_gpu_fractions: List[float],
        layer_comp_loads: List[float],
        avail_mem_bytes: int,
        overlap_grad_sync_with_bp: bool = False,
        _gamma: float = 1.1,
    ) -> Tuple[List[StageShape], List[int], List[int]]:
        """ 
        Infer the minimal extra intra-stage communication introduced by the enumerated optimizations 
        for a given partition plan, under the constraint of layer memory access.

        Supported intra-stage parallelizations are as follows:

         - Case 1 ("single"). Data, tensor parallelism and their hybrid combinations. Data/tensor 
           parallelism ranks are represented as '(dp_rank, tp_rank)' in `stage_logical_shape`, while 
           other intra-stage parallelizations (e.g., ZeRO) need to be further specified. Note that in this 
           case, all layers within one stage shares the same logical shape.
        
         - Case 2 ("mixed"). Layer-granularity mixed optimizations within one stage. In this case, all 
           layers in one stage can be applied with different optimizations. For example, layer i uses 
           two-way tensor parallelism and layer i + 1 uses two-way data parallelism. 

        The intra-stage communication inferring rules include two scenarios:

         - Scenario 1. When we need to locally determine the optimal shape of one stage, we use 
           `comm_tp + comm_dp // num_micro_batches` to compare, plus the resharding penalty as we know the 
           optimal shape of its last stage.
        
         - Scenario 2. When we need to globally determine the total intra-stage communication of one list of 
           stage shapes, given the list of (1) per-stage comm and (2) additional comm, the result should be 
           `max per-stage comm + the sum of addition comm`, for each micro-batch. Note that for simplicity, 
           we ignore the intra-stage comm of other stages and only count for that of the longest stage, since 
           the sum of each stage's intra-stage comm should be constant after the stage shape is determined (by 
           choosing the shape with minimal intra-stage comm).
        
        Args:
         - `partition_plan`: The stage partition plan to be infered.
         - `sharding`: The normalized GPU sharding of the stage partition plan.
        """

        # Update global record
        plan_hashkey = Cell.gen_hashkey_with_partition_plan(partition_plan)
        if plan_hashkey not in self._global_stage_all_logical_shapes:
            self._global_stage_all_logical_shapes[plan_hashkey] = {}

        # Pre-compute inter-stages comm for global record
        inter_stages_comms = self._cal_inter_stages_comms(partition_plan, sharding, layer_outvar_size)

        # Candidate intra-stage parallelizations for each stage
        stage_cand_logical_shapes = self._enum_stage_logical_shapes_fixed_plan(partition_plan, sharding)

        # Infer intra-stage communication and select the shapes with the lowest one, 
        # under the constraint of layer memory access.
        stage_logical_shapes = []                       # List[(dp_rank, tp_rank)]
        all_per_stage_comm, all_addn_comm = [], []      # List[intra_stage_comm]
        last_stage_shape = None
        
        for (stage_idx, shapes) in enumerate(stage_cand_logical_shapes):
            # Update global comm record
            if stage_idx not in self._global_stage_all_logical_shapes[plan_hashkey]:
                self._global_stage_all_logical_shapes[plan_hashkey][stage_idx] = {}

            layer_ids = partition_plan[stage_idx]
            # In/outvar size of the stage
            stage_layer_invar_size, stage_layer_outvar_size, stage_layer_param_size = [], [], []
            for layer_id in layer_ids:
                stage_layer_invar_size.append(layer_op_invar_size[layer_id])
                stage_layer_outvar_size.append(layer_op_outvar_size[layer_id])
                stage_layer_param_size.append(layer_param_size[layer_id])
            
            # Memory footprint of the stage
            stage_mem = np.sum([layer_mem_access[_i] for _i in layer_ids])

            if len(shapes) == 1 and np.prod(shapes[0].stage_shape) == 1:
                # Only one GPU in this stage
                stage_logical_shapes.append(shapes[0])
                all_per_stage_comm.append(0)
                all_addn_comm.append(0)
                last_stage_shape = shapes[0]
               
                # Update global record
                shape_hashkey = Cell.gen_hashkey_with_stage_shape(shapes[0])
                sum_gpu_fraction = np.sum([layer_gpu_fractions[_i] for _i in layer_ids])
                comp_load = np.sum([layer_comp_loads[_i] for _i in layer_ids])
                alloc_gpu_num = sharding[stage_idx]
                # inter_stage_comm = inter_stages_comms[stage_idx]
                inter_stages_comm = 0
                self._global_stage_all_logical_shapes[plan_hashkey][stage_idx][shape_hashkey] = (
                    sum_gpu_fraction, alloc_gpu_num, comp_load, 0, 0, inter_stages_comm,
                )
                continue

            min_comm, best_shape, best_per_stage_comm, best_addn_comm = 1e12, None, None, None
            for (shape_idx, shape) in enumerate(shapes):
                # Update global record
                shape_hashkey = Cell.gen_hashkey_with_stage_shape(shape)

                if shape.type == "single":
                    # Case 1
                    stage_op_invar_size = [_e for _layer in stage_layer_invar_size for _e in _layer]
                    stage_op_outvar_size = [_e for _layer in stage_layer_outvar_size for _e in _layer]
                    comm_dp, comm_tp = self._infer_intra_stage_comm_fixed_stage_and_optimization(
                        stage_op_invar_size, stage_op_outvar_size, stage_layer_param_size, shape.stage_shape,
                    )
                    
                    # Total introduced communication amount of the stage
                    # The communication fo tensor parallelism happens in each micro-batch, while 
                    # that of data parallelism only happens at last.
                    stage_total_comm = comm_tp + comm_dp // self.num_micro_batches

                    if stage_idx > 0 and shape.stage_shape != last_stage_shape:
                        # Penalty of cross-stages resharding
                        stage_total_comm *= _gamma

                    if overlap_grad_sync_with_bp:
                        # Overlapped
                        # Communication of both tensor and adata parallelism are behaved as aggregated per-stage
                        # communication, while no additional grad_sync communication exists.
                        per_stage_comm = comm_tp + comm_dp // self.num_micro_batches
                        addn_comm = 0
                    else:
                        # Not overlapped
                        # Only communication of tensor parallelism is behaved as per-stage communication, while 
                        # the comunication of data parallelism seperately occurs after the completion of all backward
                        # passes.
                        per_stage_comm = comm_tp
                        addn_comm = comm_dp // self.num_micro_batches

                    # Update global record
                    if shape_hashkey not in self._global_stage_all_logical_shapes[plan_hashkey][stage_idx]:
                        if (stage_mem / shape.stage_shape[1]) <= avail_mem_bytes:
                            # Record: (sum GPU fraction, alloc GPU num, comp load, per-stage comm, additional comm, 
                            #          inter-stages comm).
                            sum_gpu_fraction = np.sum([layer_gpu_fractions[_i] for _i in layer_ids])
                            comp_load = np.sum([layer_comp_loads[_i] for _i in layer_ids])
                            alloc_gpu_num = sharding[stage_idx]
                            # inter_stage_comm = inter_stages_comms[stage_idx]
                            inter_stages_comm = 0
                            self._global_stage_all_logical_shapes[plan_hashkey][stage_idx][shape_hashkey] = (
                                sum_gpu_fraction, alloc_gpu_num, comp_load, per_stage_comm, addn_comm, inter_stages_comm,
                            )
                        else:
                            # Out-of-memory case
                            self._global_stage_all_logical_shapes[plan_hashkey][stage_idx][shape_hashkey] = (
                                None, None, None, None, None, None,
                            )
                    
                    if (
                        stage_total_comm < min_comm and                         # Smaller intra-stage communication
                        (stage_mem / shape.stage_shape[1]) < avail_mem_bytes    # Memory footprint is satisfied
                    ):
                        best_shape = shape
                        best_per_stage_comm = per_stage_comm
                        best_addn_comm = addn_comm
                        min_comm = stage_total_comm

                elif shape.type == "mixed":
                    # Case 2
                    raise NotImplementedError()

                else:
                    raise ValueError(f"Invalid stage shape for inferring intra-stage communication: {shape}")

            stage_logical_shapes.append(best_shape)     # Currently not used
            all_per_stage_comm.append(best_per_stage_comm)
            all_addn_comm.append(best_addn_comm)
            last_stage_shape = best_shape

        return stage_logical_shapes, all_per_stage_comm, all_addn_comm

    def _infer_intra_stage_comm_fixed_stage_and_optimization(
        self,
        stage_op_invar_size: List[int],
        stage_op_outvar_size: List[int],
        stage_layer_param_size: List[int],
        stage_logical_shape: Tuple[int],
    ) -> Tuple[int, int]:
        """ 
        Infer the intra-stage communication with the specified stage and optimization. 
        
        The rules are as follows:

         - For tensor parallelism, use `all-reduce` primitive to gather outvars of each 
           operator from all sharded GPUs.
        
         - For data parallelism, use `all-reduce` primitive to reduce/scatter gradients (
           same shape and dtype as layer parameters) across all replicas.
        
        It should be noted that the above inferring rules are some basic scenarios for collective
        communication, and the communication primitives can be varied in real-world application.

        Args:
         - `stage_op_invar_size`: List of invar size for all operators in the stage.
         - `stage_op_outvar_size`: List of outvar size for all operators in the stage.
         - `stage_layer_param_size`: List of param size for all layers in the stage.
         - `stage_logical_shape`: Logical shape of the stage to denote the parallel plan.
        """

        intra_stage_comm_dp, intra_stage_comm_tp = 0, 0

        if stage_logical_shape[0] > 1:
            # Data parallelism, only the last micro-batch leads to the synchronization among all stages
            for var_size in stage_layer_param_size:
                intra_stage_comm_dp += self._get_comm(
                    var_size, primitive="all-reduce", num_workers=stage_logical_shape[0],
                )
        
        if stage_logical_shape[1] > 1:
            # Tensor parallelism, each micro-batch leads to communication
            # for var_size in stage_op_invar_size:
            #     # All-to-all
            #     intra_stage_comm_tp += self._get_comm(
            #         var_size, primitive="all-to-all", num_workers=stage_logical_shape[1],
            #     )
            for var_size in stage_op_outvar_size:
                # All-gather
                intra_stage_comm_tp += self._get_comm(
                    var_size, primitive="all-reduce", num_workers=stage_logical_shape[1],
                )
        
        return intra_stage_comm_dp, intra_stage_comm_tp
    
    def _infer_intra_stage_comm_fixed_stage_and_mixed_optimizations(
        self,
        stage_layer_invar_size: List[List[int]],
        stage_layer_outvar_size: List[List[int]],
        stage_layer_param_size: List[int],
        layer_mem_access: List[int],
        avail_mem_bytes: int,
    ) -> Tuple[int, int]:
        """
        Infer the intra-stage communication with the specified stage, supporting optimize
        layer-granularity parallel plans in the stage (i.e., mixed optimizations within the stage). 

        The rules are as follows (following the objective in Alpa's intra-op ILP formulation):

         - Optimization goal: Minimize the sum of communication and resharding costs of all layers 
           in the stage. ignoring computation cost.
        
         - We use a ILP solver to solve this problem. Specifically, we first determine the shape of 
           the first layer to minimize its intra-layer communication cost; then, we sequentially 
           determine stage i to minimize the sum of (1) intra-layer communication cost and (2) resharding 
           cost; finally, need to be under the constraint of memory access.
        
        Args:
         - `stage_layer_invar_size`: List of invar size for all layers in the stage.
         - `stage_layer_outvar_size`: List of outvar size for all layers in the stage.
        """

        layer_shapes = []
        num_layers = len(stage_layer_invar_size)

        # TODO(chunyu): Use a ILP solver to solve this problem.

        for i in range(num_layers):
            layer_invar_size = stage_layer_invar_size[i]
            layer_outvar_size = stage_layer_outvar_size[i]

            self._optimize_mixed_optimizations_one_stage()

            raise NotImplementedError()
    
    def _optimize_mixed_optimizations_one_stage(
        self,
    ) -> Optional[List[int]]:
        """ TODO(chunyu) Implement this. """

    def _get_comm(self, var_size: int, primitive: str, num_workers: int):
        """ Get the communication with the specified communication primitive. """
        
        if primitive == "all-to-all":
            # One worker sends the input tensor to other (num_workers - 1) workers 
            return var_size * (num_workers - 1) / num_workers
        elif primitive == "all-gather":
            # Other (n - 1) workers send their output tensor to the main worker
            return var_size * (num_workers - 1) / num_workers
        elif primitive == "all-reduce":
            # All n workers perform ring all-reduce primitive
            return var_size * (num_workers - 1) / num_workers
        else:
            raise ValueError(f"Invalid primitive value: {primitive}")


def eqn_flops_func(eqn: JaxprEqn, no_skip: bool = False) -> float:
    """ Get the FLOPs of a jaxpr equation. """
    
    if "jaxpr" in eqn.params:
        return sum(eqn_flops_func(x, no_skip) for x in eqn.params["jaxpr"].eqns)
    
    properties = _analyze_eqn(eqn, no_skip)

    return properties["flops"] if ("flops" in properties and properties["flops"] > 1.0) else 0.0


def layer_flops_func(layer: JaxPipelineComputation) -> float:
    """ Get the FLOPs of a jaxpr pipeline computation. """
    properties = _analyze_layer(layer)
    return properties["flops"] if ("flops" in properties and properties["flops"] > 1.0) else 0.0


def eqn_mem_func(eqn: JaxprEqn, no_skip: bool = False, include_custom_call: bool = True) -> float:
    """
    Get the memory footprint (input, output memory access) of a jaxpr equation.
    """

    if "jaxpr" in eqn.params:
        return sum(eqn_mem_func(x, no_skip, include_custom_call) for x in eqn.params["jaxpr"].eqns)

    properties = _analyze_eqn(eqn, no_skip, include_custom_call)

    if False:
    # if no_skip:
        # Each operator can be consecutive and the intermediate tensors are shared both by its producer 
        # and consumer operators, thus only count the weight and output memory footprint of operators.
        mem_weight_bytes = properties["bytes accessed operand 1 {}"] \
            if ("bytes accessed operand 1 {}" in properties and 
                properties["bytes accessed operand 1 {}"] > 0) else 0
        mem_output_bytes = properties["bytes accessed output"] \
            if ("bytes accessed output" in properties and 
                properties["bytes accessed output"] > 0) else 0
        return mem_weight_bytes + mem_output_bytes
    else:
        # Heavy operators (e.g., gemm, conv) are commonly not consecutive to each other, thus we count
        # the total memory footprint of these operators.
        return properties["bytes accessed"] \
            if ("bytes accessed" in properties and properties["bytes accessed"] > 0) else 0


def layer_mem_func(layer: JaxPipelineComputation) -> float:
    """ Get the memory footprint (input, output memory access) of a jaxpr pipeline computation. """
    properties = _analyze_layer(layer)
    return properties["bytes accessed"] \
            if ("bytes accessed" in properties and properties["bytes accessed"] > 0) else 0


def eqn_weight_mem(eqn: JaxprEqn) -> float:
    """
    Get the required memory footprint of the parameter weight in a jaxpr equation.

    The parameter weight of the equation is corresponding to the operand 1.
    """

    if "jaxpr" in eqn.params:
        return sum(eqn_weight_mem(x) for x in eqn.params["jaxpr"].eqns)

    properties = _analyze_eqn(eqn, include_custom_call=False)

    return properties["bytes accessed operand 1 {}"] \
        if "bytes accessed operand 1 {}" in properties else 0.0


def _analyze_eqn(eqn: JaxprEqn, no_skip: bool = False, include_custom_call: bool = True) -> Dict[str, float]:
    """ Analyze properties of Jaxpr equation by HLO module cost analysis. """

    primitives = non_trivial_primitive if not include_custom_call else non_trivial_primitive_include_custom
    if (eqn.primitive not in primitives and not no_skip):
        # Skip lightweight eqn
        return {
            "flops": 0.0,
            "bytes accessed": 0,
            "bytes accessed operand 1 {}": 0,
            "bytes accessed output": 0,
        }
    
    if str(eqn) in eqn_prop_cache:
        # Return cache
        return eqn_prop_cache[str(eqn)]

    new_inv = [inv for inv in eqn.invars if isinstance(inv, Var)]
    jaxpr = Jaxpr([], new_inv, eqn.outvars, [eqn])
    closed_jaxpr = ClosedJaxpr(jaxpr, [])
    hlo_module = jaxpr_to_hlo("tmp", closed_jaxpr, [
        False,
    ] * len(jaxpr.invars)).get_module()
    backend = xb.get_backend("cpu")
    properties = xc._xla.hlo_module_cost_analysis(  # pylint: disable=protected-access
        backend, hlo_module)
    
    # Cache
    eqn_prop_cache[str(eqn)] = properties
    
    return properties


def _analyze_layer(layer: JaxPipelineComputation) -> Dict[str, float]:
    """ Analyze properties of Jaxpr pipeline computation by HLO module cost analysis. """

    closed_jaxpr = layer.closed_jaxpr()
    hlo_module = jaxpr_to_hlo("tmp", closed_jaxpr, [
        False,
    ] * len(closed_jaxpr.jaxpr.invars)).get_module()
    backend = xb.get_backend("cpu")
    properties = xc._xla.hlo_module_cost_analysis(  # pylint: disable=protected-access
        backend, hlo_module)
    
    return properties


@lru_cache(maxsize=None)
def _enum_cand_shardings(gpu_num: int, forward_stage_num: int) -> List[Any]:
    """ Given stage num, enumerate all candidate gpus shardings with constraint of power of 2. """

    if forward_stage_num == 1:
        return [[gpu_num]]
    
    # All candidate elms
    pow_two_list = [2**_i for _i in range(gpu_num.bit_length())]
    shardings = list()

    def __gen_shardings(cur_sharding: Sequence[int], s_idx: int, cur_sum: int):
        """ Generate candidate shardings. """
        if len(cur_sharding) == forward_stage_num and cur_sum == gpu_num:
            # Candidate sharding
            shardings.append(cur_sharding[:])
            return
        if len(cur_sharding) >= forward_stage_num or cur_sum > gpu_num:
            # Invalid
            return
        for _i in range(s_idx, len(pow_two_list)):
            cur_sharding.append(pow_two_list[_i])
            # Next elm
            __gen_shardings(cur_sharding, _i, cur_sum + pow_two_list[_i])
            # Backtrack
            cur_sharding.pop()
    
    __gen_shardings([], 0, 0)

    time.sleep(1)
    
    # Permutate
    gpu_shardings = []
    for sharding in shardings:
        gpu_shardings.extend([list(_s) for _s in list(set(permutations(sharding)))])
    
    return gpu_shardings


def _normalize_gpu_sharding(
    partition_plan: List[List[int]],
    layer_gpu_fractions: Sequence[float],
    cand_shardings: List[List[int]],
    norm_gpu_sharding_cache: Dict[str, Tuple[List[int], float]] = None,
    only_symmetric_sharding: bool = False,
) -> Tuple[List[int], float]:
    """ 
    Normalize the raw GPU sharding of the partition plan by approximating the nearest GPU sharding 
    w.r.t. the power of 2 rule. The similarity is evaluated by the L2 norm of two vectors.
    """

    plan_hashkey = Cell.gen_hashkey_with_partition_plan(partition_plan)
    if (norm_gpu_sharding_cache is not None and plan_hashkey in norm_gpu_sharding_cache):
        return norm_gpu_sharding_cache[plan_hashkey]

    raw_sharding = [
        np.sum([layer_gpu_fractions[_l] for _l in _s]) for _s in partition_plan
    ]

    best_sharding, best_l2_norm = None, 1e12
    for sharding in cand_shardings:
        assert len(sharding) == len(raw_sharding), \
            f"Mismatched stage num between candidate sharding ({len(sharding)}) " + \
            f"and raw sharding ({len(raw_sharding)})."
    
        l2_norm = np.sqrt(np.sum((np.array(raw_sharding) - np.array(sharding)) ** 2))
        if (
            l2_norm < best_l2_norm and
            (not only_symmetric_sharding or 
             all([_s == sharding[0] for _s in sharding]))
        ):
            best_sharding = sharding
            best_l2_norm = l2_norm
    
    if norm_gpu_sharding_cache is not None:
        norm_gpu_sharding_cache[plan_hashkey] = (best_sharding, best_l2_norm)

    return best_sharding, best_l2_norm


def _enum_all_partition_plans_dp(
    num_layers: int,
    num_stages: int,
    layer_gpu_fractions: Sequence[float],
    _sigma: float = 0.7,
    min_layers_per_stage: int = 1,
    only_symmetric_sharding: bool = False,
) -> List[List[List[int]]]:
    """
    Enumerate all candidate stage partition plans with dynamic programming that satisfies: 

    The accumulated GPU fraction of all partitioned stages must be larger than \_sigma GPUs. This is 
    for avoiding significant imbalance between FLOPs and allocated GPUs (since #GPU must be 
    approximated to the power of 2 later).
    """

    # Adjust _sigma with num_gpus and num_stages with the rules:
    # Larger num_gpus, smaller num_stages, _sigma should be larger.
    _sigma = round(_sigma * (np.sum(layer_gpu_fractions) / num_stages), 1)
    # Construct partition plans
    plans = []

    # 2D dp array
    # Structure: `dp[i][j]` means the collection of partition plans by 
    #            spliting the first i layers into j stages.
    dp = [[[] for _ in range(num_stages + 1)] for _ in range(num_layers + 1)]
    dp[0][0] = [[]]

    # Dp for-loop
    for i in range(1, num_layers + 1):
        for j in range(1, min(i, num_stages) + 1):
            for k in range(i):
                for plan in dp[k][j - 1]:
                    if len(plan) == 0 or plan[-1][-1] < i - 1:
                        low = 0 if len(plan) == 0 else plan[-1][-1] + 1
                        new_plan = plan + [[m for m in range(low, i)]]
                        dp[i][j].append(new_plan)

    # Candidate shardings
    cand_shardings = _enum_cand_shardings(int(np.sum(layer_gpu_fractions) + 0.2), num_stages)
    
    # Look for valid plans
    for plan in dp[num_layers][num_stages]:
        if any([len(_s) < min_layers_per_stage for _s in plan]):
            # At least min_layer_per_stage layers in each stage
            continue
        
        stage_gpu_fractions = [
            np.sum([layer_gpu_fractions[_l] for _l in _s]) for _s in plan
        ]
        stage_alloc_gpu_nums, _ = _normalize_gpu_sharding(
            plan, layer_gpu_fractions, cand_shardings, only_symmetric_sharding=only_symmetric_sharding,
        )

        if (
            only_symmetric_sharding and
            not all([_s == stage_alloc_gpu_nums[0] for _s in stage_alloc_gpu_nums])
        ):
            # Skip asymmetric case
            continue

        if all([_f >= _sigma for _f in stage_gpu_fractions]):
            # A valid plan
            plans.append(plan)

    return plans


def _enum_logical_shapes_one_stage(num_gpus: int):
    """ Enumerate logical shapes for one stage. """
    assert is_power_of(base=2, target=num_gpus), \
        f"Total device num ({num_gpus}) should be the power of 2."
    shapes = []
    log_nd = int(np.log2(num_gpus))
    for dp_rank in range(0, log_nd + 1, 1):
        tp_rank = log_nd - dp_rank
        shapes.append(
            StageShape(
                type="single", 
                stage_shape=(pow(2, dp_rank), pow(2, tp_rank)), 
                layer_shape=None,
        ))
    return shapes


def _enum_all_parallelism_and_save(
    partition_plans: List[List[List[int]]],
    layer_gpu_fractions: List[float],
    num_gpus: int, 
    num_stages: int,
    cell_cfgs: CellConfigs,
    saved_path: str = "./tmp_all_parallelism.csv",
) -> None:
    """ 
    Enumerate all parallelism w.r.t. stage partition and intra-stage parallelism, saving to the
    specified file path.
    """

    plan_shape_hashkeys = []
    cand_shardings = _enum_cand_shardings(num_gpus, num_stages)

    if num_stages == 1:
        # Only one stage
        assert len(partition_plans) == 1, \
            f"Should be only one stage partition plan."
        plan_hashkey = Cell.gen_hashkey_with_partition_plan(partition_plans[0])
        stage_logical_shapes = _enum_logical_shapes_one_stage(num_gpus)
        for shape in stage_logical_shapes:
            shape_hashkey = Cell.gen_hashkey_with_stage_shape(shape)
            plan_shape_hashkeys.append(f"{plan_hashkey}::{shape_hashkey}")
        # Save
        _save_enumerated_parallelism(plan_shape_hashkeys, saved_path)

        return

    for plan in partition_plans:
        plan_hashkey = Cell.gen_hashkey_with_partition_plan(plan)
        gpu_sharding, _ = _normalize_gpu_sharding(plan, layer_gpu_fractions, cand_shardings)
        if (
            cell_cfgs.only_symmetric_sharding and
            not all([_n_g == gpu_sharding[0] for _n_g in gpu_sharding])
        ):
            # Skip asymmetric case to reduce parallelism num
            continue
        
        # Queue for storing intermediate partial plans in plan enumeration
        plan_queue = []
        # Queue in the first stage
        stage_logical_shapes = _enum_logical_shapes_one_stage(gpu_sharding[0])
        for shape in stage_logical_shapes:
            plan_queue.append([shape])
        
        for stage_idx in range(1, num_stages, 1):
            # Loop for all partial plans with the length of `stage_idx`
            while len(plan_queue) > 0 and len(plan_queue[0]) == stage_idx:
                # Pop the element in the queue head
                partial_plan = plan_queue.pop(0)
            
                num_gpus_this_stage = gpu_sharding[stage_idx]
                stage_logical_shapes = _enum_logical_shapes_one_stage(num_gpus_this_stage)
                for shape in stage_logical_shapes:
                    new_partial_plan = partial_plan + [shape]

                    if len(new_partial_plan) == num_stages:
                        # Complete plan construction and record
                        shape_hashkeys = [
                            Cell.gen_hashkey_with_stage_shape(_s) 
                                for _s in new_partial_plan
                        ]

                        if (
                            cell_cfgs.only_universal_shape and
                            not all([_s == shape_hashkeys[0] for _s in shape_hashkeys])
                        ):
                            # Skip shape-different case to reduce parallelism num
                            continue

                        if (
                            cell_cfgs.max_universal_shape_num > 0 and
                            (shape_hashkeys.count(max(set(shape_hashkeys), key=shape_hashkeys.count)) > 
                             cell_cfgs.max_universal_shape_num)
                        ):
                            # Skip shape-same case to reduce parallelism num
                            continue

                        if (
                            cell_cfgs.universal_shape_stage_num > 0 and
                            not all([_s == shape_hashkeys[0] for _s in shape_hashkeys[:cell_cfgs.universal_shape_stage_num]])
                        ):
                            # The first several stages must share the same logical shape
                            continue
                        
                        shape_hashkeys = "__".join(shape_hashkeys)
                        plan_shape_hashkey = f"{plan_hashkey}::{shape_hashkeys}"
                        plan_shape_hashkeys.append(plan_shape_hashkey)
                    else:
                        # Queue in 
                        plan_queue.append(new_partial_plan)

    # Save
    _save_enumerated_parallelism(plan_shape_hashkeys, saved_path)
    # Exit the program
    exit(0)


def _save_enumerated_parallelism(
    plan_shape_hashkeys: List[str],
    saved_path: str,
) -> None:
    """ Save all enumerated parallelism to the specified path. """

    import csv

    print(f"[TMP] The enumerated parallelism is saved to: '{saved_path}'")
    with open(saved_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([[_k] for _k in plan_shape_hashkeys])
        # for hashkey in plan_shape_hashkeys:
        #     # f.write(hashkey + "\n")
        #     writer.writerow(hashkey)
