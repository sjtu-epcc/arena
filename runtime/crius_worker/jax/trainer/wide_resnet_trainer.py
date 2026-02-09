#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Trainer for Wide-ResNet.
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


class WideResNetTrainer:
    """
    Trainer for Wide ResNet.
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
        # For Wide-ResNet
        self.resnet_layer_num = trainer_cfgs['resnet_layer_num']
        self.width_factor = trainer_cfgs['width_factor']
        self.num_classes = trainer_cfgs['num_classes']
        self.image_size = trainer_cfgs['image_size']
        self.num_channels = trainer_cfgs['num_channels']
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
    
    def cross_entropy_loss(self, logits, labels):
        """
        Cross-entropy loss definition.
        """
        num_classes = logits.shape[-1]
        one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
        xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
        return jnp.mean(xentropy)
    
    def _get_flax_model(self, model_cfgs):
        # Loader func
        loader_func = ModelLoader()
        # Modify model cfgs based on training cfgs
        model_cfgs['dtype'] = self.dtype
        model_cfgs['resnet_layer_num'] = self.resnet_layer_num
        model_cfgs['width_factor'] = self.width_factor
        model_cfgs['num_classes'] = self.num_classes
        model_cfgs['num_filters'] = self.num_channels

        return loader_func(model_cfgs=model_cfgs)
    
    def _get_data_loader(self, dataset_cfgs):
        # Loader func
        loader_func = DataLoader()
        # Modify dataset cfgs based on training cfgs
        dataset_cfgs['batch_size'] = self.batch_size

        return loader_func(dataset_cfgs=dataset_cfgs)
    
    def _get_init_state(self, batched_data, learning_rate_fn):
        print("[I] Initialize training state...")

        # Rngkey
        rngkey = jax.random.PRNGKey(self.rand_seed)
        # Params
        params = self.model.init_dummy(rngkey, batched_data)
        params, batch_stats = params["params"], params["batch_stats"]

        # Optmizer in Jax
        tx = optax.sgd(learning_rate=learning_rate_fn,
                       momentum=self.momentum,
                       nesterov=True)
        # State
        state = TrainState.create(apply_fn=self.model.apply, 
                                 params=params, 
                                 tx=tx,
                                 batch_stats=batch_stats,
                                 dynamic_scale=None)

        print("[I] Training state initialization is completed.")

        return state

    def _get_train_step_func(self, learning_rate_fn, method, grad_func=alpa.grad):
        """ Get the traning step function. """
        _str = "[I] Constructing Alpa parallelized train step func..." if not self.is_jaxpr_transform \
                    else "[I] Constructing general train step func..."
        print(_str)

        # Whether to perform jaxpr transformation
        if not self.is_jaxpr_transform:
            # Decorate with alpa parallelization
            @alpa.parallelize(method=method)
            def train_step_func(state, batch):
                """ Parallelized train step function. """
                # Loss func
                def loss_func(params):
                    # NOTE: Logits is the output of the fully-connected layer.
                    logits, new_model_state = state.apply_fn(
                        {
                            'params': params,
                            'batch_stats': state.batch_stats,
                        },
                        batch['images'],
                        mutable=['batch_stats']
                    )
                    loss = self.cross_entropy_loss(logits=logits, labels=batch['labels'])
                    metrics = {
                        'loss': loss,
                        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch["labels"]),
                        'lr': learning_rate_fn(step),
                    }
                    return loss, (new_model_state, metrics)

                step = state.step
                # NOTE: Without dynamic_scale mentioned in alpa benchmark (also not compepled)
                # NOTE: (jax.grad) has_aux (bool) – Optional, bool. Indicates whether fun returns a pair 
                #       where the first element is considered the output of the mathematical function to be 
                #       differentiated and the second element is auxiliary data. Default False.
                grads, aux = grad_func(loss_func, has_aux=True)(state.params)
                new_model_state, metrics = aux
                new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])

                return new_state, metrics
        else:
            # Naive train step function
            def train_step_func(state, batch):
                """ Normal train step function. """
                # Loss func
                def loss_func(params):
                    # NOTE: Logits is the output of the fully-connected layer.
                    logits, new_model_state = state.apply_fn(
                        {
                            'params': params,
                            'batch_stats': state.batch_stats,
                        },
                        batch['images'],
                        mutable=['batch_stats']
                    )
                    loss = self.cross_entropy_loss(logits=logits, labels=batch['labels'])
                    metrics = {
                        'loss': loss,
                        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch["labels"]),
                        'lr': learning_rate_fn(step),
                    }
                    return loss, (new_model_state, metrics)

                step = state.step
                # NOTE: Without dynamic_scale mentioned in alpa benchmark (also not compepled)
                # NOTE: (jax.grad) has_aux (bool) – Optional, bool. Indicates whether fun returns a pair 
                #       where the first element is considered the output of the mathematical function to be 
                #       differentiated and the second element is auxiliary data. Default False.
                grads, aux = grad_func(loss_func, has_aux=True)(state.params)
                new_model_state, metrics = aux
                new_state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])

                return new_state, metrics

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
        
        # Due to legacy issues, we turn off auto-tuning. Although the
        # performance will be much better if we turn it on
        # global_config.xla_gpu_autotune_level = 0
        
        # Batches
        batches = []
        # Manual construct batch with specified size.
        batch = {
            'images': 
                jnp.ones((self.batch_size, self.image_size, self.image_size, 3), dtype=jnp.int32),
            'labels':
                jnp.ones((self.batch_size), dtype=jnp.int32),
        }
        batches.append(batch)

        print("[I] Manually construct dummy batch with the shape of {}.".format(batch['images'].shape))
        
        # Learning rate func
        learning_rate_fn = create_learning_rate_fn()
        # Init state
        state = self._get_init_state(batched_data=batch['images'], 
                                   learning_rate_fn=learning_rate_fn)
        grad_func = alpa.grad
        # Parallel method
        method = self._get_pipshard_parallel_method()
        # Train step func
        train_step_func = self._get_train_step_func(learning_rate_fn=learning_rate_fn, 
                                                    method=method, 
                                                    grad_func=grad_func)
    
        # Use the benchmark tool provided by alpa team.
        (latencies, e2e_total_time, niter, local_lats, max_mem_allocated, compilation_times,
     executable) = compile_and_benchmark_pipeshard_training_executable(
         parallel_mode=self.parallel_mode,
         niter=self.niter,
        #  niter=min(self.niter, len(batches)),
         train_step=train_step_func,
         state=state,
         other_train_step_inputs=(batch,),
         batches=batches,
         dump_debug_file_path=dump_debug_file_path,
         job_id=job_id, try_idx=try_idx,
         profile_driver_time=self.profile_driver_time)
        
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
        #                                 is_need_save=is_need_save)        

        # Shutdown
        alpa.shutdown()

        return local_avg_lat
