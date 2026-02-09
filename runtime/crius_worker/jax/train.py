#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Long-term Training Process with automated pipeline stages slicing & OPs/tensors sharding provided by Alpa.
-----------------------------------------
- Ref: https://github.com/alpa-projects/alpa
"""

import os
import argparse
import alpa
from alpa.util import disable_tqdm_globally
import jax.numpy as jnp
import ray

from trainer.wide_resnet_trainer import WideResNetTrainer
from trainer.gpt_trainer import GPTTrainer
from trainer.moe_trainer import MoETrainer
from configs import benchmark_cfgs
from utils import get_file_path


# Args 
parser = argparse.ArgumentParser()
# Job
parser.add_argument("--job_id", default="default", type=str)
# Configurations of Ray cluster
parser.add_argument("--ray_address", default='auto', type=str)
parser.add_argument("--devices_name", default="1_1080ti", type=str)
parser.add_argument("--num_devices_per_node", default=4, type=int)
parser.add_argument("--num_nodes", default=2, type=int)
parser.add_argument("--is_ray_cluster_existed", default=False, action='store_true')
# Configurations of model training
parser.add_argument("--model_name", default='wide_resnet', type=str)
parser.add_argument("--param_num", default='500M', type=str)
parser.add_argument("--dataset_name", default='none', type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--resnet_layer_num", default=50, type=int)
parser.add_argument("--num_micro_batches", default=16, type=int, help="The num of micro batches for pipeline. \
                                                                       Local bs of each stage = bs / num_mb at each time slot.")
parser.add_argument("--num_pipeline_layers", default=16, type=int, help="The num of layers for operators clustering.")
parser.add_argument("--niter", default=100, type=int)
parser.add_argument("--try_idx", default=1, type=int)
# Others
parser.add_argument("--is_dp_only", default=False, action='store_true', help="Force to apply data parallelism.")
parser.add_argument("--is_pp_only", default=False, action='store_true', help="Force to apply pipeline parallelism.")
parser.add_argument("--is_mp_only", default=False, action='store_true', help="Force to apply model parallelism.")
parser.add_argument("--is_manual_config_test", default=False, action='store_true', help="Run a test with manually specified configuration on pipeline & sharding.")
parser.add_argument("--is_dummy_test", default=False, action='store_true', help="Run a customized dummy test (only resnet).")
parser.add_argument("--disable_alpa_profiling_db", default=False, action='store_true', help="Disable exploiting the profiling database in Alpa for kernel-profiling.")
parser.add_argument("--verbose", default=False)
args = parser.parse_args()


# Devices num per node
num_devices_per_node = args.num_devices_per_node
# Nodes num
num_nodes = args.num_nodes
# Disable tqdm
disable_tqdm_globally()


def init_backend(ray_cluster_existed: bool = False):
    """ Initializing Ray Cluster & Alpa backend. """
    # Connect to or construct a Ray Cluster
    if ray_cluster_existed:
        # No init log will be printed. 
        ray.init(address=args.ray_address)
    else: 
        ray.init()

    # NOTE: To specify the deivces num per node or the nodes num, use:
    # - `alpa.init('ray', num_devices_per_node=2, num_nodes=2)`
    # alpa.init(cluster="ray")
    alpa.init(cluster="ray", num_devices_per_node=num_devices_per_node, num_nodes=num_nodes)

    print("[I] Ray Cluster & Alpa backend initialization is completed.")
    print("[I] Device Info:")
    print("    - Devices num per node: {}".format(num_devices_per_node))
    print("    - Nodes num: {}".format(num_nodes))


def load_trainer_configs():
    """ Load trainer configurations. """
    # Parse
    model_name = args.model_name
    param_num = args.param_num
    # Check
    assert (model_name in benchmark_cfgs.keys()) and (param_num in benchmark_cfgs[model_name].keys())
    # Model cfgs
    model_cfgs = benchmark_cfgs[model_name][param_num]

    trainer_cfgs = {
        # Basic
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'batch_size': args.batch_size,
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
        'num_micro_batches': args.num_micro_batches,
        'num_pipeline_layers': args.num_pipeline_layers,
        'parallel_mode': 'search',
        'niter': args.niter,
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
    
    if args.is_dummy_test:
        trainer_cfgs = {
            # Basic
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'batch_size': args.batch_size,
            'lr': 1e-3,
            'momentum': 0.9,
            'rand_seed': 123,
            'dtype': None,
            # For WideResNet
            'resnet_layer_num': 50,
            'width_factor': 1,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 10,
            # For Bert & MoE
            'seq_len': 1024,
            'hidden_size': 16,
            'num_layers': 4,
            'num_heads': 4,
            'vocab_size': 51200,
            # For MoE
            'expert_group_size': 2048,
            'num_experts': 4,
            # Other
            'num_micro_batches': args.num_micro_batches,
            'num_pipeline_layers': args.num_pipeline_layers,
            'parallel_mode': 'search',
            'niter': args.niter,
            'profile_driver_time': True,
        }
    
    return trainer_cfgs


def format_file_path():
    """ Get the file path of the output .csv file. """
    # Get the file path of the output .csv file.
    partial_cfgs = {
        'num_nodes': num_nodes,
        'num_devices_per_node': num_devices_per_node,
        'devices_name': args.devices_name,
        'model_name': args.model_name,
        'dataset_name': args.dataset_name,
        'batch_size': args.batch_size,
        'num_micro_batches': args.num_micro_batches,
        'num_pipeline_layers': args.num_pipeline_layers,
        'param_num': args.param_num,
        'try_idx': args.try_idx,
        'is_manual_config_test': args.is_manual_config_test,
    }

    return get_file_path(partial_cfgs=partial_cfgs)


def trainer_func(trainer_cfgs, file_path):
    """ Instantiate trainer and functionalize it. """
    # _disable_alpa_profiling_db = args.disable_alpa_profiling_db
    _disable_alpa_profiling_db = os.environ.get("DISABLE_ALPA_PROFILING_DB") == "True"
    
    # Trainer
    if args.model_name == 'wide_resnet':
        trainer = WideResNetTrainer(trainer_cfgs=trainer_cfgs, 
                                    file_path=file_path, 
                                    is_dp_only=args.is_dp_only, 
                                    is_pp_only=args.is_pp_only, 
                                    is_mp_only=args.is_mp_only, 
                                    is_manual_config_test=args.is_manual_config_test, 
                                    devices_name=args.devices_name, 
                                    num_nodes=num_nodes, 
                                    num_devices_per_node=num_devices_per_node, 
                                    disable_alpa_profiling_db=_disable_alpa_profiling_db)
    elif args.model_name == 'bert' or args.model_name == 'gpt':
        trainer = GPTTrainer(trainer_cfgs=trainer_cfgs, 
                             file_path=file_path, 
                             is_dp_only=args.is_dp_only, 
                             is_pp_only=args.is_pp_only, 
                             is_mp_only=args.is_mp_only, 
                             is_manual_config_test=args.is_manual_config_test, 
                             devices_name=args.devices_name, 
                             num_nodes=num_nodes, 
                             num_devices_per_node=num_devices_per_node, 
                             disable_alpa_profiling_db=_disable_alpa_profiling_db)
    elif args.model_name == 'moe':
        trainer = MoETrainer(trainer_cfgs=trainer_cfgs, 
                             file_path=file_path, 
                             is_dp_only=args.is_dp_only, 
                             is_pp_only=args.is_pp_only, 
                             is_mp_only=args.is_mp_only, 
                             is_manual_config_test=args.is_manual_config_test, 
                             devices_name=args.devices_name, 
                             num_nodes=num_nodes, 
                             num_devices_per_node=num_devices_per_node,
                             disable_alpa_profiling_db=_disable_alpa_profiling_db)
    else:
        raise NotImplementedError("Error: Unsupported model type.")
    
    # Dump debug file path
    subfolder = str(num_nodes) + '_nodes_' + str(num_devices_per_node) + '_devices_per_node'
    subsubfolder = args.devices_name
    subsubsubfolder = str(args.model_name) + '_' + str(args.dataset_name)
    _path = 'bs_' + str(args.batch_size) + '_nmb_' + str(args.num_micro_batches) + '_pln_' + str(args.num_pipeline_layers) + '_param_num_' + str(args.param_num)
    # Check path
    if not os.path.exists('./profile_result/debug/'):
        os.mkdir('./profile_result/debug/')
    if not os.path.exists('./profile_result/debug/' + subfolder):
        os.mkdir('./profile_result/debug/' + subfolder)
    if not os.path.exists('./profile_result/debug/' + subfolder + '/' + subsubfolder):
        os.mkdir('./profile_result/debug/' + subfolder + '/' + subsubfolder)
    if not os.path.exists('./profile_result/debug/' + subfolder + '/' + subsubfolder + '/' + subsubsubfolder):
        os.mkdir('./profile_result/debug/' + subfolder + '/' + subsubfolder + '/' + subsubsubfolder)
    # Path
    dump_debug_file_path = './profile_result/debug/' + subfolder + '/' + subsubfolder + '/' + subsubsubfolder + '/' + _path    
    
    print("[I] Dump debug file path:", dump_debug_file_path)

    # For loss convergence evaluation
    evaluate_loss = True
    if evaluate_loss:
        os.environ["EVAL_LOSS"] = "true"

    # Train
    trainer.train(dump_debug_file_path=dump_debug_file_path, job_id=args.job_id, try_idx=args.try_idx)


def main():
    """ Main function. """
    # Xla envs
    os.environ["XLA_AUTO_TUNE_LEVEL"] = "0"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    # Others
    #  Compared to 1f1b, "gpipe" consumes more gpu memory.
    os.environ["PIPELINE_SCHEDULE"] = "gpipe"
    os.environ["PROFILING_WARMUP_NUM"] = "2"
    
    # Init backend
    init_backend(args.is_ray_cluster_existed)
    # Load trainer cfgs
    trainer_cfgs = load_trainer_configs()
    # File path
    file_path = format_file_path()    
    # Train
    trainer_func(trainer_cfgs=trainer_cfgs, file_path=file_path)


if __name__ == "__main__":
    main()
