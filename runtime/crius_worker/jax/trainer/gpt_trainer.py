#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Trainer for GPT & Bert.
"""

import os
import time
import numpy as np
from abc import ABC, abstractmethod
import alpa
from alpa import global_config
from alpa.testing import assert_allclose
from alpa.util import disable_tqdm_globally
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.pipeline_parallel.stage_construction import get_last_dp_result
import copy
from flax import linen as nn
from flax.training import common_utils, train_state
# from flax.optim import dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray
from typing import Any
import argparse
from contextlib import nullcontext

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loader.model_loader import ModelLoader
from loader.data_loader import DataLoader
from configs import model_cfgs_proto, dataset_cfgs_proto
from utils import TrainState, TrainState2, create_learning_rate_fn, output_and_save_distributed_trainer, get_file_path, \
                  compute_gpt_tflops, compute_moe_tflops, compute_gpt_parameter_count, compute_moe_parameter_count
from benchmark_parallel_utils import (
    compile_and_benchmark_pipeshard_training_executable,
    benchmark_training_executable)
from macro.macro_def import BYTE_TO_GB


class GPTTrainer:
    """
    Trainer for GPT & Bert.
    """
    def __init__(self, trainer_cfgs, file_path, is_dp_only, is_pp_only, is_mp_only, is_manual_config_test, 
                 devices_name, num_nodes, num_devices_per_node, is_jaxpr_transform=False, 
                 forward_stage_layer_id=None, submesh_physical_shapes=None, submesh_logical_shapes=None, auto_sharding_option=None, 
                 disable_alpa_profiling_db=False):
        # Hardwares
        self.devices_name = devices_name
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        # Settings
        self.model_name = trainer_cfgs['model_name']
        self.dataset_name = trainer_cfgs['dataset_name']
        self.batch_size = trainer_cfgs['batch_size']
        self.lr = trainer_cfgs['lr']
        self.momentum = trainer_cfgs['momentum']
        self.rand_seed = trainer_cfgs['rand_seed']
        self.dtype = trainer_cfgs['dtype']
        # For GPT & Bert
        self.seq_len = trainer_cfgs['seq_len']
        self.hidden_size = trainer_cfgs['hidden_size']
        self.num_layers = trainer_cfgs['num_layers']
        self.num_heads = trainer_cfgs['num_heads']
        self.vocab_size = trainer_cfgs['vocab_size']
        # Load
        self.model = self._get_flax_model(model_cfgs=model_cfgs_proto[self.model_name])
        self.data_loader = self._get_data_loader(dataset_cfgs=dataset_cfgs_proto[self.dataset_name])
        # Pipeline
        # NOTE: Divide the batch size into [x] micro-batches. 
        #       The number of micro batches for gradient accumulation. 
        self.num_micro_batches = trainer_cfgs['num_micro_batches']
        self.num_pipeline_layers = trainer_cfgs['num_pipeline_layers']
        # Parallel Settings
        self.parallel_mode = trainer_cfgs['parallel_mode']
        self.niter = trainer_cfgs['niter']
        # File path
        self.file_path = file_path
        # Profile Settings
        # NOTE: If false, just profile the pure iteration time on workers; 
        #       else, profile the iteration time + sync time (driver overhead)
        self.profile_driver_time = trainer_cfgs['profile_driver_time']
        # Force DP
        self.is_dp_only = is_dp_only
        # Force PP
        self.is_pp_only = is_pp_only
        # Force MP
        self.is_mp_only = is_mp_only
        # Whether manual config
        self.is_manual_config_test = is_manual_config_test
        self.forward_stage_layer_id = forward_stage_layer_id
        self.submesh_physical_shapes = submesh_physical_shapes
        self.submesh_logical_shapes = submesh_logical_shapes
        self.auto_sharding_option = auto_sharding_option
        # Whether to apply jaxpr transform in profiling
        self.is_jaxpr_transform = is_jaxpr_transform
        self.disable_alpa_profiling_db = disable_alpa_profiling_db
    
    def _get_flax_model(self, model_cfgs):
        # Loader func
        loader_func = ModelLoader()
        # Modify model cfgs based on training cfgs
        model_cfgs['dtype'] = self.dtype
        model_cfgs['vocab_size'] = self.vocab_size
        model_cfgs['hidden_size'] = self.hidden_size
        model_cfgs['num_attention_heads'] = self.num_heads
        model_cfgs['intermediate_size'] = self.hidden_size * 4
        model_cfgs['num_hidden_layers'] = self.num_layers
        model_cfgs['type_vocab_size'] = 0
        model_cfgs['tie_word_embeddings'] = False
        model_cfgs['gradient_checkpointing'] = None
        model_cfgs['add_manual_pipeline_markers'] = None
        model_cfgs['pipeline_mp_size'] = None

        return loader_func(model_cfgs=model_cfgs)
    
    def _get_data_loader(self, dataset_cfgs):
        # Loader func
        loader_func = DataLoader()
        # Modify dataset cfgs based on training cfgs
        dataset_cfgs['batch_size'] = self.batch_size

        return loader_func(dataset_cfgs=dataset_cfgs)
    
    def _get_init_state(self, batch, is_state_avail):
        print("[I] Initialize training state...")

        # Rngkey
        rngkey = jax.random.PRNGKey(self.rand_seed)
        
        def __get_context_for_model_loading(type: str):
            """ The context is one-off and needs to be re-get for each usage. """
            # Context for model loading
            load_ctx = nullcontext()
            if os.environ.get("CPU_LOAD_MODEL", "false") == "true":
                print(f"Total CPU cores: {os.cpu_count()}")
                print("XLA_FLAGS env variable:", os.environ.get(f"XLA_FLAGS", ""))
                print(f"Available CPU cores: {len(jax.devices('cpu'))}")
                print(f"[I] Enabling CPU (host memory) loading on: {type} ...")
                load_ctx = jax.default_device(jax.devices('cpu')[0])

            return load_ctx
        
        # Parameters
        load_ctx = __get_context_for_model_loading("model params")
        if not is_state_avail:
            with load_ctx:
                params = self.model.init_dummy(rngkey, batch['input_ids'], batch['attention_mask'],
                                            batch['token_type_ids'], batch['position_ids'])
        else:
            with load_ctx:
                params = jax.eval_shape(self.model.init, rngkey, batch['input_ids'], batch['attention_mask'],
                                        batch['token_type_ids'], batch['position_ids'])
        
        def weight_decay_mask(pytree):
            # Do not use weight decay on layer norm and bias.
            return jax.tree_map(lambda x: x.ndim > 1, pytree)
        
        # Optmizer in Jax
        load_ctx = __get_context_for_model_loading("jax optimizer")
        with load_ctx:
            tx = optax.chain(optax.adamw(learning_rate=self.lr, mask=weight_decay_mask))
        use_master_copy = (self.dtype == jnp.float16)
        
        # State
        load_ctx = __get_context_for_model_loading("training state")
        with load_ctx:
            state = TrainState2.create(
                apply_fn=self.model.apply, params=params, tx=tx, use_master_copy=use_master_copy, dynamic_scale=None,
            )

        print("[I] Training state initialization is completed.")

        return state

    def _get_train_step_func(self, method, grad_func=alpa.grad):
        """ Get the traning step function. """
        _str = "[I] Constructing Alpa parallelized train step func..." if not self.is_jaxpr_transform \
                    else "[I] Constructing general train step func..."
        print(_str)

        # Whether to perform jaxpr transformation
        if not self.is_jaxpr_transform:
            # Decorate with alpa parallelization
            @alpa.parallelize(method=method)
            def train_step_func(state, batch, rng_key=jax.random.PRNGKey(self.rand_seed)):
                """ Parallelized train step function. """
                # Loss func
                def lossFunc(params):
                    rngs = {'dropout': rng_key}
                    logits = state.apply_fn(params, 
                                            batch['input_ids'],
                                            batch['attention_mask'],
                                            batch['token_type_ids'],
                                            batch['position_ids'],
                                            deterministic=True,
                                            rngs=rngs)[0]
                    label_mask = jnp.where(batch['labels'] > 0, 1.0, 0.0)
                    labels = jax.nn.one_hot(batch['labels'], logits.shape[-1])
                    loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
                    loss = (label_mask * loss).sum() / label_mask.sum()
                    metrics = {
                        'loss': loss,
                    }
                    return loss, (metrics, )
                
                grads, (metrics, ) = grad_func(lossFunc, has_aux=True)(state.params)
                # New state
                new_state = state.apply_gradients(grads=grads)

                return new_state, metrics
        else:
            # Naive train step function
            def train_step_func(state, batch, rng_key=jax.random.PRNGKey(self.rand_seed)):
                """ Normal train step function. """
                # Loss func
                def lossFunc(params):
                    rngs = {'dropout': rng_key}
                    logits = state.apply_fn(params, 
                                            batch['input_ids'],
                                            batch['attention_mask'],
                                            batch['token_type_ids'],
                                            batch['position_ids'],
                                            deterministic=True,
                                            rngs=rngs)[0]
                    label_mask = jnp.where(batch['labels'] > 0, 1.0, 0.0)
                    labels = jax.nn.one_hot(batch['labels'], logits.shape[-1])
                    loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
                    loss = (label_mask * loss).sum() / label_mask.sum()

                    return loss
                
                grads = grad_func(lossFunc)(state.params)
                # New state
                new_state = state.apply_gradients(grads=grads)

                return new_state

        _str = "[I] Alpa parallelized train step func construction is completed." if not self.is_jaxpr_transform \
                    else "[I] General train step func construction is completed."
        print(_str)

        return train_step_func

    def _get_pipshard_parallel_method(self):    
        # Allow mixed 1d mesh and 2d mesh shape.
        allow_mixed_mesh_shape = True
        # Prefer reduce scatter
        prefer_reduce_scatter = True
        # Use remat
        use_remat = True
        remat_mode = "coarse_grained_remat" if use_remat else "none"
        # Pipeline schedule
        pipeline_schedule = os.environ.get("PIPELINE_SCHEDULE", "1f1b")

        # Stage option
        # - `stage_option="auto"` means we enable the auto stage construction algorithm.
        # - `stage_option=alpa.ManualStageOption()` means we apply manual pipeline (stage assignment) & sharding spec of each stage.
        
        # # Part 1. Layer -> stage mapping: '[[0, 1], [2]]' means assign layer_0 and layer_1 onto the first submesh as a stage, 
        # #         layer_2 onto the second submesh as a stage.
        # forward_stage_layer_id = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
        
        # # Part 2. Submesh physical shapes: '[(1, 2), (1, 2)]' means there are 2 nodes (each with 2 GPUs), we formulate each node as a submesh.
        # submesh_physical_shapes = [(self.num_nodes, self.num_devices_per_node)]
        
        # # Part 3. Submesh logical shapes: '[(2, 1), (1, 2)]' means on the first submesh, use 2-replica data parallelism; 
        # #         on the second submesh, use 2-shard model parallelism.
        # submesh_logical_shapes = [(self.num_nodes * self.num_devices_per_node, 1)]
        
        # # Part 4. Submesh autosharding option dicts: The auto-sharding options of each stage. 
        # #         `autosharding_option_dicts = {"force_batch_dim_to_mesh_dim": 0} or {}`：Forcibly map the batch dimension to a mesh dimension. 
        # #         If set to 0, then force the batch tensor dim to match the first mesh dim.
        # submesh_autosharding_option_dicts = [{"force_batch_dim_to_mesh_dim": 0}]

        if self.is_manual_config_test:
            # Apply manual specified parallelism
            forward_stage_layer_id = self.forward_stage_layer_id
            submesh_physical_shapes = self.submesh_physical_shapes
            submesh_logical_shapes = self.submesh_logical_shapes
            submesh_autosharding_option_dicts = self.auto_sharding_option
        elif self.is_dp_only:
            # Only apply data parallelism
            forward_stage_layer_id = [[i for i in range(self.num_pipeline_layers)]]
            submesh_physical_shapes = [(self.num_nodes, self.num_devices_per_node)]
            submesh_logical_shapes = [(self.num_nodes * self.num_devices_per_node, 1)]
            submesh_autosharding_option_dicts = [{"force_batch_dim_to_mesh_dim": 0}]
        elif self.is_pp_only:
            # Only apply pipeline parallelism
            _gpu_num = self.num_nodes * self.num_devices_per_node
            assert (self.num_pipeline_layers >= _gpu_num) and (self.num_pipeline_layers % _gpu_num == 0)
            # Uniform clustering layers by default
            forward_stage_layer_id = [[(j + i * int(self.num_pipeline_layers / _gpu_num)) for j in range(int(self.num_pipeline_layers / _gpu_num))] for i in range(_gpu_num)]
            submesh_physical_shapes = [(1, 1) for i in range(_gpu_num)]
            submesh_logical_shapes = [(1, 1) for i in range(_gpu_num)]
            # submesh_autosharding_option_dicts = [{"force_batch_dim_to_mesh_dim": 0} for i in range(_gpu_num)]
            submesh_autosharding_option_dicts = [{} for i in range(_gpu_num)]
        elif self.is_mp_only:
            # Only apply model parallelism
            forward_stage_layer_id = [[i for i in range(self.num_pipeline_layers)]]
            submesh_physical_shapes = [(self.num_nodes, self.num_devices_per_node)]
            submesh_logical_shapes = [(1, self.num_nodes * self.num_devices_per_node)]
            submesh_autosharding_option_dicts = [{"force_batch_dim_to_mesh_dim": 0}]
        else:
            # Only apply data parallelism
            forward_stage_layer_id = None
            submesh_physical_shapes = None
            submesh_logical_shapes = None
            submesh_autosharding_option_dicts = None
        
        # Construct stage option.
        manual_stage_option = alpa.ManualStageOption(forward_stage_layer_ids=forward_stage_layer_id, 
                                              submesh_physical_shapes=submesh_physical_shapes, 
                                              submesh_logical_shapes=submesh_logical_shapes, 
                                              submesh_autosharding_option_dicts=submesh_autosharding_option_dicts)
        
        if isinstance(manual_stage_option, alpa.ManualStageOption) and (self.is_manual_config_test or self.is_dp_only or self.is_pp_only or self.is_mp_only):
            print("[I] Manually slice pipeline and sharding, the related specs are as follows:")
            print("    - 'forward_stage_layer_id':", forward_stage_layer_id)
            print("    - 'submesh_physical_shapes':", submesh_physical_shapes)
            print("    - 'submesh_logical_shapes':", submesh_logical_shapes)
            print("    - 'submesh_autosharding_option_dicts':", submesh_autosharding_option_dicts)

        # Auto stage option
        _alpa_prof_pth = os.environ.get("ALPA_PROF_PATH", ".")
        _filename = f"{_alpa_prof_pth}/prof_database/prof_database_{self.devices_name}_{self.num_devices_per_node}_d.pkl"
        (_use_hlo_cost_model, _profiling_database_filename) = (True, _filename) if not self.disable_alpa_profiling_db else (False, None)
        auto_stage_option = alpa.AutoStageOption(
            submesh_physical_shape_space="small_power_of_two",
            submesh_logical_shape_space="single_node_model_parallel",
            stage_imbalance_tolerance=0.25,
            use_hlo_cost_model=_use_hlo_cost_model,
            profiling_database_filename=_profiling_database_filename,
            cached_profile_result=None,
        )
        
        # Layer option
        # - `alpa.AutoLayerOption(layer_num=2)` means we use the auto layer construcion
        #   algorithm to cluster primitive operators into two layers.
        layer_option = alpa.AutoLayerOption(layer_num=self.num_pipeline_layers, remat_mode=remat_mode)
        
        # Parallel method
        if self.is_manual_config_test or self.is_dp_only or self.is_pp_only or self.is_mp_only:
            method = alpa.PipeshardParallel(num_micro_batches=self.num_micro_batches,
                                            default_auto_sharding_option=AutoShardingOption(
                                                prefer_reduce_scatter=prefer_reduce_scatter,
                                                allow_mixed_mesh_shape=allow_mixed_mesh_shape),
                                            pipeline_schedule=pipeline_schedule,
                                            layer_option=layer_option,
                                            stage_option=manual_stage_option)
        else:
            method = alpa.PipeshardParallel(num_micro_batches=self.num_micro_batches,
                                            default_auto_sharding_option=AutoShardingOption(
                                                prefer_reduce_scatter=prefer_reduce_scatter,
                                                allow_mixed_mesh_shape=allow_mixed_mesh_shape),
                                            pipeline_schedule=pipeline_schedule,
                                            layer_option=layer_option,
                                            stage_option=auto_stage_option)
        
        return method
    
    def _get_gpt_stat(self, partial_cfgs, latencies, num_devices):
        """
        Compute statistics of GPT & Bert.
        """
        batch_size = partial_cfgs['batch_size']
        seq_len = partial_cfgs['seq_len']
        hidden_size = partial_cfgs['hidden_size']
        num_layers = partial_cfgs['num_layers']
        hidden_size = partial_cfgs['hidden_size']
        vocab_size = partial_cfgs['vocab_size']
        use_remat = True    # See https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/suite_manual_gpt.py

        tflops = compute_gpt_tflops(batch_size,
                                    seq_len,
                                    num_layers,
                                    hidden_size,
                                    vocab_size,
                                    num_devices,
                                    np.mean(latencies),
                                    checkpoint_activations=use_remat)
        parameter_count = compute_gpt_parameter_count(num_layers, hidden_size,
                                                    vocab_size)
        
        return tflops, parameter_count
    
    def train(self, dump_debug_file_path=None, job_id=None, try_idx=None):
        """ Training entrypoint. """
        # Overwrite in case that the container cannot detact devices through nividia-smi
        assert global_config.has_cuda, f"Alpa does not detact gpu backend through `nvidia-smi`..."
        
        global_config.xla_gpu_autotune_level = 4
        # Check whether XLA auto-tuning is enabled by crius profiler when measuring 
        # e2e iteration time with alpa.
        _xla_autotune_level = os.environ.get("XLA_AUTO_TUNE_LEVEL", "")
        if _xla_autotune_level != "":
            global_config.xla_gpu_autotune_level = int(_xla_autotune_level)

        print("[I] Manually constructing dummy batch...")

        # Batch
        batch = {
            'input_ids': jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            'attention_mask': jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            'token_type_ids': jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            'position_ids': jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            'labels': jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
        }

        # State
        # avail_train_state = True
        avail_train_state = False
        state = self._get_init_state(batch=batch, is_state_avail=avail_train_state)

        # Parallel method
        method = self._get_pipshard_parallel_method()
        
        # Train step func
        train_step_func = self._get_train_step_func(method=method)

        # Use the benchmark tool provided by alpa team.
        (latencies, e2e_total_time, niter, local_lats, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_pipeshard_training_executable(
         parallel_mode=self.parallel_mode,
         niter=self.niter,
        #  niter=min(self.niter, len(batches)),
         train_step=train_step_func,
         state=state,
         other_train_step_inputs=(batch,),
         batches=None,
         dump_debug_file_path=dump_debug_file_path,
         job_id=job_id, try_idx=try_idx,
         profile_driver_time=self.profile_driver_time)
        
        # # Get GPT statistics
        # partial_cfgs = {
        #     'batch_size': self.batch_size,
        #     'seq_len': self.seq_len,
        #     'hidden_size': self.hidden_size,
        #     'num_layers': self.num_layers,
        #     'hidden_size': self.hidden_size,
        #     'vocab_size': self.vocab_size,
        # }
        # tflops, parameter_count = self._get_gpt_stat(partial_cfgs=partial_cfgs, latencies=latencies, num_devices=(num_nodes * num_devices_per_node))
        
        if latencies is not None:
            # Calculate average latencies
            # Benchmark step time: warmup = 2 if niter >= 5 else 1
            avg_lat = jnp.mean(jnp.array(latencies))
            # Max allocated memory among mesh devices
            max_mem_gb = max_mem_allocated / BYTE_TO_GB
            # Local time info (without sync (driver overhead))
            local_total_time = jnp.sum(jnp.array(local_lats))
            local_avg_lat = jnp.mean(jnp.array(local_lats))
            # Need save profile result or not
            is_need_save = not self.is_manual_config_test

            # # Get DP result
            # (compute_cost_file_name, forward_stage_layer_ids, submesh_shapes,
            #     logical_mesh_shapes, autosharding_option_dicts) = get_last_dp_result()
            # # Metadata
            # metadata = {
            #     "compilation_times": compilation_times,
            #     "compute_cost_file_name": compute_cost_file_name,
            #     "forward_stage_layer_ids": forward_stage_layer_ids,
            #     "submesh_shapes": submesh_shapes,
            #     "logical_mesh_shapes": logical_mesh_shapes,
            #     "autosharding_option_dicts": autosharding_option_dicts,
            # }

            print("")
            print("[I] Performance metrics:")
            print(" - Iteration count: {}.".format(niter))
            print(" - Total e2e training time : {} s.".format(round(e2e_total_time, 3)))
            print(" - Average e2e iteration time: {} s.".format(round(avg_lat, 3)))
            print(" - Total local training time: {} s.".format(round(local_total_time, 3)))
            print(" - Average local iteration time: {} s.".format(round(local_avg_lat, 3)))
            print(" - Max allocated memory among devices: {} GB.".format(round(max_mem_gb, 3)))
            print(" - Compilation times: ", compilation_times)
            print(" - Metadata: ", [])
            print(" - Is need save result: ", is_need_save)
            print("")
        else:
            # Unexpected error
            print("")
            print("[E] Unexpected error occurred in compiling or profiling executables...")
            alpa.shutdown()
            return -1.0

        # # Output and save
        # output_and_save_distributed_trainer(file_path=self.file_path, 
        #                                 niter=niter, 
        #                                 e2e_total_time=e2e_total_time, 
        #                                 avg_lat=avg_lat, 
        #                                 local_total_time=local_total_time, 
        #                                 local_avg_lat=local_avg_lat, 
        #                                 max_mem_gb=max_mem_gb, 
        #                                 compilation_times=compilation_times,
        #                                 # metadata=metadata, 
        #                                 is_need_save=is_need_save)        

        # Shutdown
        alpa.shutdown()

        return local_avg_lat
