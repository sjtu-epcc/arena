#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os
import pandas as pd
import numpy as np
from typing import Any, Callable, Optional
import optax
import flax
from flax.training import common_utils
# from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import train_state, dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp


def create_learning_rate_fn():
    """
    Create learning rate schedule.
    ---------------------------------
    Copied from Alpa Wide-ResNet benchmark.
    """
    base_learning_rate = 0.1
    warmup_epochs = 5.0
    steps_per_epoch = 10000
    num_epochs = 100.0

    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=warmup_epochs *
                                      steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs *
                                            steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


class TrainState(train_state.TrainState):
    """
    Customized train state in Alpa Wide-ResNet benchmark.
    """
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


class TrainState2(train_state.TrainState):
    """This is an extended version of flax.training.train_state.TrainState.

    This class wraps the logic for creating the master weight copy in
    mixed precision training.
    """
    master_copy: flax.core.FrozenDict[str, Any]
    dynamic_scale: Optional[dynamic_scale_lib.DynamicScale]

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.
        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.
        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.
        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        if self.master_copy is None:
            master_params = self.params
        else:
            master_params = self.master_copy

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                master_params)
        new_master_params = optax.apply_updates(master_params, updates)

        if self.master_copy is None:
            new_master_copy = None
            new_params = new_master_params
        else:
            new_master_copy = new_master_params
            new_params = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float16), new_master_params)

            # A hack to make the donation works perfectly in gradient accumulation:
            # We need the accumulate_grad to take the old params as input.
            new_params_flat, tree = jax.tree_util.tree_flatten(new_params)
            old_params_flat, _ = jax.tree_util.tree_flatten(self.params)
            new_params_flat = [
                x + 0.0 * y for x, y in zip(new_params_flat, old_params_flat)
            ]
            new_params = jax.tree_util.tree_unflatten(tree, new_params_flat)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            master_copy=new_master_copy,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, use_master_copy=False, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        if use_master_copy:
            master_copy = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), params)
            params = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float16), params)
            opt_state = tx.init(master_copy)
        else:
            master_copy = None
            opt_state = tx.init(params)

        return cls(
            step=np.array(0, dtype=np.int32),
            apply_fn=apply_fn,
            params=params,
            master_copy=master_copy,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    @classmethod
    def create_aval(cls,
                    *,
                    apply_fn,
                    params,
                    tx,
                    use_master_copy=False,
                    **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = jax.eval_shape(tx.init, params)

        if use_master_copy:
            master_copy = params
            params = jax.eval_shape(
                lambda p: jax.tree_util.tree_map(
                    lambda x: jnp.asarray(x, dtype=jnp.float16), p), params)
        else:
            master_copy = None

        return cls(
            step=np.array(0, dtype=np.int32),
            apply_fn=apply_fn,
            params=params,
            master_copy=master_copy,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )



def get_file_path(partial_cfgs):
    """
    Get the file path of the output .csv file.
    """
    
    # Parse
    num_nodes = partial_cfgs['num_nodes']
    num_devices_per_node = partial_cfgs['num_devices_per_node']
    devices_name = partial_cfgs['devices_name']
    model_name = partial_cfgs['model_name']
    dataset_name = partial_cfgs['dataset_name']
    batch_size = partial_cfgs['batch_size']
    num_micro_batches = partial_cfgs['num_micro_batches']
    num_pipeline_layers = partial_cfgs['num_pipeline_layers']
    param_num = partial_cfgs['param_num']
    try_idx = partial_cfgs['try_idx']
    is_manual_config_test = partial_cfgs['is_manual_config_test']

    # Print output and save as .csv file
    subfolder = str(num_nodes) + '_nodes_' + str(num_devices_per_node) + '_devices_per_node'
    subsubfolder = devices_name
    subsubsubfolder = str(model_name) + '_' + str(dataset_name)
    file_name = 'bs_' + str(batch_size) + '_nmb_' + str(num_micro_batches) + '_pln_' + str(num_pipeline_layers) + '_param_num_' + str(param_num) + '_try_' + str(try_idx) + '.csv'
    # Check path
    if not os.path.exists('./profile_result'):
        os.mkdir('./profile_result')
    if not os.path.exists('./profile_result/' + subfolder):
        os.mkdir('./profile_result/' + subfolder)
    if not os.path.exists('./profile_result/' + subfolder + '/' + subsubfolder):
        os.mkdir('./profile_result/' + subfolder + '/' + subsubfolder)
    if not os.path.exists('./profile_result/' + subfolder + '/' + subsubfolder + '/' + subsubsubfolder):
        os.mkdir('./profile_result/' + subfolder + '/' + subsubfolder + '/' + subsubsubfolder)
    # .csv file path
    file_path = './profile_result/' + subfolder + '/' + subsubfolder + '/' + subsubsubfolder + '/' + file_name
    # Check
    if os.path.exists(file_path) and not is_manual_config_test:
        assert not os.path.exists(file_path)


def output_and_save_distributed_trainer(file_path, niter, e2e_total_time, avg_lat, local_total_time, \
                                    local_avg_lat, max_mem_gb, compilation_times, metadata=None, is_need_save=True):
    """
    Print the output of the train() in distributed trainer and save as .csv file.
    """
    print("")
    print("[I] Performance metrics:")
    print(" - Iteration count: {}.".format(niter))
    print(" - Total e2e training time : {} s.".format(round(e2e_total_time, 3)))
    print(" - Average e2e iteration time: {} s.".format(round(avg_lat, 3)))
    print(" - Total local training time: {} s.".format(round(local_total_time, 3)))
    print(" - Average local iteration time: {} s.".format(round(local_avg_lat, 3)))
    print(" - Max allocated memory among devices: {} GB.".format(round(max_mem_gb, 3)))
    print(" - Compilation times: ", compilation_times)
    print(" - Metadata: ", metadata)
    print(" - Is need save result: ", is_need_save)
    print("")
    
    # if is_need_save:
    #     if metadata is not None:
    #         output = {
    #             "Iteration count": [str(niter)],
    #             "Total e2e training time (s)": [str(round(e2e_total_time, 3))],
    #             "Average e2e iteration time (s)": [str(round(avg_lat, 3))],
    #             "Total local training time (s)": [str(round(local_total_time, 3))],
    #             "Average local iteration time (s)": [str(round(local_avg_lat, 3))],
    #             "Max allocated memory among devices (GB)": [str(round(max_mem_gb, 3))],
    #             "Compilation times": [str(compilation_times)],
    #             "compilation_times (meta)": [metadata['compilation_times']],
    #             "compute_cost_file_name": [metadata['compute_cost_file_name']],
    #             "forward_stage_layer_ids": [metadata['forward_stage_layer_ids']],
    #             "submesh_shapes": [metadata['submesh_shapes']],
    #             "logical_mesh_shapes": [metadata['logical_mesh_shapes']],
    #             "autosharding_option_dicts": [metadata['autosharding_option_dicts']],
    #         }
    #     else:
    #         output = {
    #             "Iteration count": [str(niter)],
    #             "Total e2e training time (s)": [str(round(e2e_total_time, 3))],
    #             "Average e2e iteration time (s)": [str(round(avg_lat, 3))],
    #             "Total local training time (s)": [str(round(local_total_time, 3))],
    #             "Average local iteration time (s)": [str(round(local_avg_lat, 3))],
    #             "Max allocated memory among devices (GB)": [str(round(max_mem_gb, 3))],
    #             "Compilation times": [str(compilation_times)],
    #         }
    #     # Write
    #     df = pd.DataFrame(output)
    #     df.to_csv(path_or_buf=file_path, sep=',', header=True, index=False, \
    #             mode='w', encoding='utf-8')


def compute_gpt_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       vocab_size,
                       num_gpus,
                       latency,
                       backward=True,
                       checkpoint_activations=False):
    """
    Copied from https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/util.py
    """
    factor = 24
    if backward:
        factor += 48
    if checkpoint_activations:
        factor += 24

    total_flop = factor * batch_size * seq_len * (hidden_size ** 2) * num_layers * \
          (1 + seq_len / (6 * hidden_size)) \
          + 6 * batch_size * seq_len * hidden_size * vocab_size
    # Note: The above formula does not count the first embedding table lookup
    # because it is a sparse operation.
    # If we use dense dot to compute the first embedding table lookup,
    # then the last term in total_flops should be
    # "+ 10 * batch_size * seq_len * hidden_size * vocab_size".
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_moe_tflops(batch_size,
                       seq_len,
                       num_layers,
                       hidden_size,
                       group_size,
                       vocab_size,
                       num_expert,
                       num_gpus,
                       latency,
                       mlp_factor=8,
                       checkpoint_activations=False):
    """
    Copied from https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/util.py
    """
    factor = 4 if checkpoint_activations else 3
    # num_layers / 2 attention block
    pure_transformer = batch_size * seq_len * (hidden_size ** 2) * (8 + 4 * mlp_factor) +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    pure_transformer = pure_transformer * factor

    # num_layers / 2 attention-moe block
    # transformer
    moe_transformer = batch_size * seq_len * (hidden_size ** 2) * 8  +\
        4 * batch_size * (seq_len ** 2) * hidden_size
    # expert FFNs:
    # moe_transformer += 2 * batch_size * seq_len * (hidden_size ** 2) * mlp_factor * 2
    moe_transformer += 8 * batch_size * seq_len * (hidden_size**2) * mlp_factor

    # softmax
    moe_transformer += 2 * batch_size * seq_len * hidden_size * num_expert
    # top-2 gating
    moe_transformer += 2 * (batch_size * seq_len) * 2 * group_size
    # dispatch + combine
    moe_transformer += 2 * batch_size * seq_len * hidden_size * 2 * group_size * 2

    moe_transformer = moe_transformer * factor

    # vocab
    embedding = 6 * batch_size * seq_len * hidden_size * vocab_size

    total_flop = pure_transformer * num_layers / 2 + \
                 moe_transformer * num_layers / 2 + embedding
    tflops = total_flop / latency / num_gpus / 1e12
    return tflops


def compute_gpt_parameter_count(num_layers, hidden_size, vocab_size):
    """
    Copied from https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/util.py
    """
    return num_layers * (
        # self-attention
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) +
        # mlp
        hidden_size * (4 * hidden_size + 1) + hidden_size * 4 *
        (hidden_size + 1) +
        # layer norm
        hidden_size * 4) + vocab_size * (hidden_size + 1)


def compute_moe_parameter_count(num_layers,
                                hidden_size,
                                vocab_size,
                                num_expert,
                                mlp_factor=8,
                                tie_embedding=True):
    """
    Copied from https://github.com/alpa-projects/alpa/blob/main/benchmark/alpa/util.py
    """
    pure_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1) + \
        hidden_size * 4
    moe_transformer = \
        hidden_size * (3 * hidden_size + 1) + hidden_size * (hidden_size + 1) + \
        num_expert * (hidden_size * (mlp_factor * hidden_size + 1) + hidden_size * mlp_factor * (hidden_size + 1)) + \
        hidden_size * 4

    # embedding
    embedding_factor = 1 if tie_embedding else 2
    embedding = embedding_factor * vocab_size * (hidden_size + 1)

    if num_expert == 1:
        return pure_transformer * num_layers + embedding
    else:
        half = num_layers / 2
        return half * pure_transformer + half * moe_transformer + embedding
