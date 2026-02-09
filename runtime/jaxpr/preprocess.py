#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to the preprocess functions in runtime profiling. """

from typing import Sequence, Any, Dict
import jax
import jax.numpy as jnp
from jax.core import ClosedJaxpr, Var, gensym
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
import alpa
from alpa.util import GradFuncTransformContext
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call)
from alpa.pipeline_parallel.stage_construction import get_stage_outvars
from alpa.pipeline_parallel.layer_construction import ManualLayerOption, AutoLayerOption, automatic_layer_construction
from alpa.pipeline_parallel.compile_executable import (
    split_and_process_layers, slice_apply_grad_for_stage_construction)

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crius_worker.jax.configs import benchmark_cfgs
from crius_worker.jax.trainer.wide_resnet_trainer import WideResNetTrainer
from crius_worker.jax.trainer.gpt_trainer import GPTTrainer
from crius_worker.jax.trainer.moe_trainer import MoETrainer
from crius_worker.jax.utils import (TrainState, create_learning_rate_fn)
from jaxpr.utils import InputConfigs
from jaxpr.testing import get_dummy_trainer_cfgs


def load_trainer_configs(model_name: str, param_num: str, batch_size: int, num_micro_batches: int, 
                         num_pipeline_layers: int, niter: int):
    """ Load trainer configurations. """
    # Parse
    model_name = model_name
    param_num = param_num
    # Check
    assert (model_name in benchmark_cfgs.keys()) and (param_num in benchmark_cfgs[model_name].keys())
    # Model cfgs
    model_cfgs = benchmark_cfgs[model_name][param_num]

    trainer_cfgs = {
        # Basic
        'model_name': model_name,
        'param_num': param_num,
        'dataset_name': "none",
        'batch_size': batch_size,
        'lr': 1e-3,
        'momentum': 0.9,
        'rand_seed': 123,
        'dtype': None,
        # For WideResNet
        'resnet_layer_num': -1,
        'width_factor': -1,
        'num_classes': -1,
        'image_size': -1,
        'num_channels': -1,
        # For Bert & MoE
        'seq_len': -1,
        'hidden_size': -1,
        'num_layers': -1,
        'num_heads': -1,
        'vocab_size': -1,
        # For MoE
        'expert_group_size': -1,
        'num_experts': -1,
        # Other
        'num_micro_batches': num_micro_batches,
        'num_pipeline_layers': num_pipeline_layers,
        'parallel_mode': 'search',
        'niter': niter,
        'profile_driver_time': True,
    }
    
    # Common items
    trainer_cfgs['dtype'] = model_cfgs['dtype']
    # Model-specified items
    if model_name == 'wide_resnet':
        # Wide-ResNet
        trainer_cfgs['resnet_layer_num'] = model_cfgs['layer_num']
        trainer_cfgs['width_factor'] = model_cfgs['width_factor']
        trainer_cfgs['num_classes'] = model_cfgs['num_classes']
        trainer_cfgs['image_size'] = model_cfgs['image_size']
        trainer_cfgs['num_channels'] = model_cfgs['num_channels']
    elif model_name == 'bert':
        # Bert
        trainer_cfgs['seq_len'] = model_cfgs['seq_len']
        trainer_cfgs['hidden_size'] = model_cfgs['hidden_size']
        trainer_cfgs['num_layers'] = model_cfgs['num_layers']
        trainer_cfgs['num_heads'] = model_cfgs['num_heads']
        trainer_cfgs['vocab_size'] = model_cfgs['vocab_size']
    else:
        # MoE
        trainer_cfgs['seq_len'] = model_cfgs['seq_len']
        trainer_cfgs['hidden_size'] = model_cfgs['hidden_size']
        trainer_cfgs['num_layers'] = model_cfgs['num_layers']
        trainer_cfgs['num_heads'] = model_cfgs['num_heads']
        trainer_cfgs['vocab_size'] = model_cfgs['vocab_size']
        trainer_cfgs['expert_group_size'] = model_cfgs['expert_group_size']
        trainer_cfgs['num_experts'] = model_cfgs['num_experts']
    
    return trainer_cfgs


def prepare_state_and_batch(trainer: Any, model_type: str):
    """ Prepare the state and batch of the given model. """
    if model_type == "wide_resnet":
        batches = [{
                "images": 
                    jnp.ones((trainer.batch_size, trainer.image_size, trainer.image_size, 3), dtype=jnp.int32),
                "labels":
                    jnp.ones((trainer.batch_size), dtype=jnp.int32),
        }]
        learning_rate_fn = create_learning_rate_fn()
        state = trainer._get_init_state(batched_data=batches[0]['images'], 
                                   learning_rate_fn=learning_rate_fn)
        return state, batches, learning_rate_fn
    elif model_type == "bert":
        _batch_size = trainer.batch_size
        _seq_len = trainer.seq_len
        batches = [{
            "input_ids": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "attention_mask": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "position_ids": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "labels": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
        }]
        state = trainer._get_init_state(batch=batches[0], is_state_avail=False)
        return state, batches, None
    elif model_type == "moe":
        _batch_size = trainer.batch_size
        _seq_len = trainer.seq_len
        batches = [{
            "input_ids": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "attention_mask": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "position_ids": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
            "labels": jnp.ones((_batch_size, _seq_len), dtype=jnp.int32),
        }]
        state = trainer._get_init_state(batch=batches[0])
        return state, batches, None


def prepare_flax_model(input_cfgs: InputConfigs, enable_alpa: bool = False, 
                       disable_alpa_profiling_db: bool = False):
    """ Prepare flax model to be transformed. """
    
    # Configurations
    trainer_cfgs = input_cfgs.trainer_cfgs
    file_path = input_cfgs.tmp_pth
    model_name = trainer_cfgs["model_name"]
    is_dp_only = input_cfgs.is_dp_only
    is_pp_only = input_cfgs.is_pp_only
    is_mp_only = input_cfgs.is_mp_only
    devices_name = input_cfgs.devices_name
    num_nodes = input_cfgs.hardware_configs.num_nodes
    num_devices_per_node = input_cfgs.hardware_configs.num_devices_per_node

    # Only for measuring iter time with alpa. If estimated with crius profiler, 
    # the input parallel method should be empty.
    is_manual_config_test = input_cfgs.is_manual_config_test \
        if enable_alpa else False       # Not mean auto-search
    forward_stage_layer_id = input_cfgs.parallel_method.forward_stage_layer_id \
        if enable_alpa else None
    submesh_physical_shapes = input_cfgs.parallel_method.submesh_physical_shapes \
        if enable_alpa else None
    submesh_logical_shapes = input_cfgs.parallel_method.submesh_logical_shapes \
        if enable_alpa else None
    auto_sharding_option = input_cfgs.parallel_method.auto_sharding_option \
        if enable_alpa else None

    # Trainer
    if model_name == "wide_resnet":
        trainer_cls = WideResNetTrainer
    elif model_name == "bert":
        trainer_cls = GPTTrainer
    elif model_name == "moe":
        trainer_cls = MoETrainer
    else:
        raise RuntimeError("Invalid model name.")

    trainer = trainer_cls(trainer_cfgs=trainer_cfgs, file_path=file_path, 
                          is_dp_only=is_dp_only, is_pp_only=is_pp_only, is_mp_only=is_mp_only, 
                          is_manual_config_test=is_manual_config_test, 
                          devices_name=devices_name, 
                          num_nodes=num_nodes, 
                          num_devices_per_node=num_devices_per_node, 
                          is_jaxpr_transform=(not enable_alpa),
                          forward_stage_layer_id=forward_stage_layer_id, 
                          submesh_physical_shapes=submesh_physical_shapes,
                          submesh_logical_shapes=submesh_logical_shapes, 
                          auto_sharding_option=auto_sharding_option, 
                          disable_alpa_profiling_db=disable_alpa_profiling_db)

    if enable_alpa:
        return trainer, None, None, None, None
    
    state, batches, learning_rate_fn = prepare_state_and_batch(trainer=trainer, model_type=trainer.model_name)
    grad_func = alpa.grad
    method = trainer._get_pipshard_parallel_method()
    # Train step func
    if model_name == "wide_resnet":
        train_step_fn = trainer._get_train_step_func(learning_rate_fn=learning_rate_fn, method=method, grad_func=grad_func)
    elif model_name == "bert" or model_name == "moe":
        train_step_fn = trainer._get_train_step_func(method=method, grad_func=grad_func)
    # Micro batch
    microbatch_size = trainer.batch_size // trainer.num_micro_batches
    micro_batch = {k: v[:microbatch_size] for k, v in batches[0].items()}

    return trainer, train_step_fn, state, micro_batch, batches[0]


def get_stage_layer_ids_uniform(forward_stage_num: int, layer_num: int):
    """ 
    Get the list of stage layer ids and stage-to-mesh mapping, considering forward and backward stages. 
    --------------------------------------------------------------
    Note:
        - Within homogenous hardwares, we assume uniform slicing of layers into stages.
    """
    assert layer_num % forward_stage_num == 0
    layer_num_per_stage = layer_num // forward_stage_num
    forward_stage_layer_ids = [[_i + _j * layer_num_per_stage for _i in range(layer_num_per_stage)] for _j in range(forward_stage_num)]
    backward_stage_layer_ids = [[2 * layer_num - 1 - _i for _i in reversed(layer_ids)] for layer_ids in reversed(forward_stage_layer_ids)]
    # Layer ids in each stage
    stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
    # Mapping from stage to mesh
    stage_to_mesh = list(range(forward_stage_num)) + list(reversed(range(forward_stage_num)))
    return stage_layer_ids, stage_to_mesh
