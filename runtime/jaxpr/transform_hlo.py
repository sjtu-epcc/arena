#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to transform Jaxpr Intermediate Representation (IR) into optimized XLA HLO IR. """

from typing import Sequence, Any, Union
import numpy as np
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.core import subjaxprs
from alpa.util import jaxpr_to_hlo, get_compile_options, XlaPassContext
from alpa.global_env import global_config
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call)
from alpa.wrapped_hlo import WrappedHlo
from alpa.shard_parallel.auto_sharding import (
    AutoShardingOption, LogicalDeviceMesh, run_auto_sharding_pass, run_spmd_partitioner_pass)

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crius_worker.jax.trainer.wide_resnet_trainer import WideResNetTrainer
from crius_worker.jax.utils import (TrainState, create_learning_rate_fn)
from jaxpr.transform_jaxpr import prepare_model_and_transform_jaxpr
from jaxpr.testing import get_dummy_input_cfgs
from jaxpr.utils import ALL_GATHER_THRESHOLD, ALL_REDUCE_THRESHOLD


def transform_jaxpr_stages_to_hlo_modules(jax_all_stages: Sequence[JaxPipelineComputation]):
    """ Transform pipeline stages in the format of Jaxpr IR into unoptimized XLA HLO modules. """
    wrapped_hlos, invars_list, outvars_list = list(), list(), list()
    for _i, stage in enumerate(jax_all_stages):
        _name = f"hlo_{_i}"
        _donated_invars = (False,) * len(stage.invars)
        wrapped_hlo = jaxpr_to_hlo(name=_name, closed_jaxpr=stage.closed_jaxpr(), donated_invars=_donated_invars)
        wrapped_hlos.append(wrapped_hlo)
        invars_list.append(stage.invars)
        outvars_list.append(stage.outvars)

    return wrapped_hlos, invars_list, outvars_list


def compile_one_sharded_hlo_module(hlo_module: WrappedHlo, num_devices: int, backend: Any, 
                                   bypass_device: bool = False):
    """ 
    Inline function to compile one sharded hlo module. 
    ------------------------------------------------------
    Args:
        - bypass_device: Allow device assignment to ensure execution on devices in compiled.execute().
    """
    # Compile options
    _num_replicas = 1
    _num_partitions = num_devices
    compile_options = get_compile_options(num_replicas=_num_replicas, num_partitions=_num_partitions, 
                                          device_assignment=np.arange(num_devices).reshape((1, -1)),
                                        #   device_assignment=(device.id,) if device else None,
                                          use_spmd_partitioning=hlo_module.is_sharding_annotated(), 
                                          parameter_is_tupled_arguments=False,
                                          build_random_seed=global_config.compile_random_seed)
    # Compilation
    with XlaPassContext({
            # Build options
            "build_option::bypass_device_assignment_check": bypass_device,
            # Communication combiner options
            "combiner::all_gather_threshold": ALL_GATHER_THRESHOLD,
            "combiner::all_reduce_threshold": ALL_REDUCE_THRESHOLD,
            "done-event::enable": global_config.enable_overlapping,
    }):
        compiled = backend.compile(hlo_module.get_computation(), compile_options)

    return compiled


def shard_one_hlo_module(hlo_module: WrappedHlo, logical_mesh_shape: Sequence[int], force_batch_dim_to_mesh_dim: int, 
                     num_micro_batches: int, num_devices: int, run_spmd_partition: bool = False):
    """ Shard hlo_module with given parallelizing method, which will be fed into backend compilation later. """
    # Enable ILP in intra-stage sharding
    as_option = AutoShardingOption(enable_auto_sharding=True, 
                                   force_batch_dim_to_mesh_dim=force_batch_dim_to_mesh_dim, 
                                   prefer_reduce_scatter=True, 
                                   allow_mixed_mesh_shape=True)
    # Logical device mesh
    assert len(logical_mesh_shape) == 2
    _id_mesh = np.arange(logical_mesh_shape[0] * logical_mesh_shape[1]).reshape(logical_mesh_shape)
    logical_device_mesh = LogicalDeviceMesh(physical_mesh=None, id_mesh=_id_mesh, mesh_alpha=None, mesh_beta=None)
    # Sharding annotated pass
    sharded_hlo_module, stage_plan = run_auto_sharding_pass(hlo_module, logical_device_mesh, return_mode="single", 
                                                            num_micro_batches=num_micro_batches, as_option=as_option)
    # SPMD partition pass
    if run_spmd_partition:
        sharded_hlo_module = run_spmd_partitioner_pass(sharded_hlo_module, num_devices)
    
    return sharded_hlo_module, stage_plan


# def optimize_hlo_modules(hlo_modules: Sequence[WrappedHlo], num_devices: int = 1, log_hlo_text: bool = False, log_pth: str = "./jaxpr/tmp"):
#     """ Run device-independent compilation passes to optimize the computation graph of HLO modules. """
#     optimized_hlo_modules = list()
#     for hlo_module in hlo_modules:
#         compiled = compile_one_sharded_hlo_module(hlo_module, num_devices)
#         optimized_hlo_module = compiled.hlo_modules()[0]
#         optimized_hlo_modules.append(optimized_hlo_module)
#         if log_hlo_text:
#             _pth = os.path.join(log_pth, f"optimized_stage_hlo_text_{hlo_modules.index(hlo_module)}.hlo")
#             with open(_pth, "w") as f:
#                 f.write(optimized_hlo_module.to_string())
        
#     return optimized_hlo_modules


# def test_hlo_transform():
#     """ Test the function of hlo transformation without sharding. """
#     # Input
#     input_cfgs = get_dummy_input_cfgs()
#     input_cfgs["is_dp_only"] = True
#     # Prepare model and transform to Jaxpr stages
#     jax_pipeline_stages = prepare_model_and_transform_jaxpr(input_cfgs=input_cfgs, forward_stage_num=1)
#     # Transform to HLO modules
#     wrapped_hlos, _, _ = transform_jaxpr_stages_to_hlo_modules(jax_pipeline_stages=jax_pipeline_stages)
#     # Device-independent compilation to optimize HLOs
#     optimized_hlo_modules = optimize_hlo_modules(hlo_modules=wrapped_hlos, num_devices=2, log_hlo_text=True)


# if __name__ == "__main__":
#     test_hlo_transform()

#     # parse_hlo_text(hlo_pth="./jaxpr/tmp/optimized_stage_hlo_text_0.hlo")
