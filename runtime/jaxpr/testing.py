#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

import os

import jax.numpy as jnp
import ray


def get_dummy_trainer_cfgs():
    """ Dummy configurations for test. """
    trainer_cfgs = {
            # Basic
            'model_name': 'wide_resnet',
            'dataset_name': 'none',
            'batch_size': 256,
            'lr': 1e-3,
            'momentum': 0.9,
            'rand_seed': 123,
            'dtype': jnp.float32,
            # For WideResNet
            'resnet_layer_num': 50,
            'width_factor': 2,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 224,
            # 'num_channels': 320,
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
            'num_micro_batches': 16,
            'num_pipeline_layers': 16,
            'parallel_mode': 'search',
            'niter': 5,
            'profile_driver_time': True,
        }
    return trainer_cfgs


def get_dummy_input_cfgs():
    """ Dummy configurations for test. """
    # Input
    input_cfgs = {
        "trainer_cfgs": get_dummy_trainer_cfgs(),
        "file_path": "./tmp/tmp.csv",
        "is_dp_only": False,
        "is_pp_only": False,
        "is_mp_only": False,
        "is_manual_config_test": False,
        "forward_stage_layer_id": None,
        "submesh_physical_shapes": None,
        "submesh_logical_shapes": None,
        "auto_sharding_option": None,
        "devices_name": "1_a40",
        "num_nodes": 1,
        "num_devices_per_node": 2,
    }
    return input_cfgs


def test_create_virtual_placement_group():
    """ Dummy test for creating virtual placement group without enough underlying Ray resources. """
    num_hosts = int(os.environ.get('CRIUS_NUM_HOSTS'))
    num_devices_per_host = int(os.environ.get('CRIUS_NUM_DEVICES_PER_HOST'))
    assert num_hosts is not None and num_devices_per_host is not None, \
        "Environment variables of the virtual cluster specs are not properly set."
    
    # Create Ray placement group
    bundles = [{
        "CPU": 1,
        "GPU": num_devices_per_host,
    } for _ in range(num_hosts)]

    # Create placement group
    placement_group = ray.util.placement_group(bundles, strategy="SPREAD", name="test_pg")

    # Remove placement group
    ray.util.remove_placement_group(placement_group)
