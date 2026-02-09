#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue


import jax.numpy as jnp


"""
Benchmark configurations. Indexed by: model_name -> param_num -> cfgs.
"""
benchmark_cfgs = {
    'wide_resnet': {
        # Expected for 2 GPUs
        '500M': {
            'layer_num': 50,
            'width_factor': 2,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 224,
            'dtype': jnp.float32,
        },
        # Expected for 4 GPUs
        '1B': {
            'layer_num': 50,
            'width_factor': 2,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 320,
            'dtype': jnp.float32,
        },
        # Expected for 8 GPUs
        '2B': {
            'layer_num': 50,
            'width_factor': 2,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 448,
            'dtype': jnp.float32,
        },
        # Expected for 16 GPUs
        '4B': {
            'layer_num': 50,
            'width_factor': 2,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 640,
            'dtype': jnp.float32,
        },
        # Expected for 32 GPUs
        '6.8B': {
            'layer_num': 50,
            'width_factor': 16,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 320,
            'dtype': jnp.float32,
        },
        # Expected for 64 GPUs
        '13B': {
            'layer_num': 101,
            'width_factor': 16,
            'num_classes': 1024,
            'image_size': 224,
            'num_channels': 320,
            'dtype': jnp.float32,
        },
    },
    'bert': {
        '350M': {
            'seq_len': 1024,
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
        # Expected for 2 GPUs
        '760M': {
            'seq_len': 1024,
            'hidden_size': 1536,
            'num_layers': 24,
            'num_heads': 16,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
        # Expected for 4 GPUs
        '1.3B': {
            'seq_len': 1024,
            'hidden_size': 2048,
            'num_layers': 24,
            'num_heads': 32,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
        # Expected for 8 GPUs
        '2.6B': {
            'seq_len': 1024,
            'hidden_size': 2560,
            'num_layers': 32,
            'num_heads': 32,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
        # Expected for 16 GPUs
        '6.7B': {
            'seq_len': 1024,
            'hidden_size': 4096,
            'num_layers': 32,
            'num_heads': 32,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
        # Expected for 32 GPUs
        '15B': {
            'seq_len': 1024,
            'hidden_size': 5120,
            'num_layers': 48,
            'num_heads': 40,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
        # Expected for 64 GPUs
        '39B': {
            'seq_len': 1024,
            'hidden_size': 8192,
            'num_layers': 48,
            'num_heads': 64,
            'vocab_size': 51200,
            'dtype': jnp.float16,
        },
    },
    'moe': {
        # Expected for 2 GPUs
        '690M': {
            'seq_len': 1024,
            'hidden_size': 768,
            'num_layers': 8,
            'num_heads': 16,
            'vocab_size': 32000,
            'expert_group_size': 2048,
            'num_experts': 16,
            'dtype': jnp.float16,
        },
        # Expected for 4 GPUs
        '1.3B': {
            'seq_len': 1024,
            'hidden_size': 768,
            'num_layers': 16,
            'num_heads': 16,
            'vocab_size': 32000,
            'expert_group_size': 2048,
            'num_experts': 16,
            'dtype': jnp.float16,
        },
        # Expected for 8 GPUs
        '2.4B': {
            'seq_len': 1024,
            'hidden_size': 1024,
            'num_layers': 16,
            'num_heads': 16,
            'vocab_size': 32000,
            'expert_group_size': 2048,
            'num_experts': 16,
            'dtype': jnp.float16,
        },
        # Expected for 16 GPUs
        '10B': {
            'seq_len': 1024,
            'hidden_size': 1536,
            'num_layers': 16,
            'num_heads': 16,
            'vocab_size': 32000,
            'expert_group_size': 2048,
            'num_experts': 32,
            'dtype': jnp.float16,
        },
        # Expected for 32 GPUs
        '27B': {
            'seq_len': 1024,
            'hidden_size': 2048,
            'num_layers': 16,
            'num_heads': 16,
            'vocab_size': 32000,
            'expert_group_size': 2048,
            'num_experts': 48,
            'dtype': jnp.float16,
        },
        # Expected for 64 GPUs
        '70B': {
            'seq_len': 1024,
            'hidden_size': 2048,
            'num_layers': 32,
            'num_heads': 16,
            'vocab_size': 32000,
            'expert_group_size': 2048,
            'num_experts': 64,
            'dtype': jnp.float16,
        },
    },
}


"""
Default model configurations.
"""
model_cfgs_proto = {
    'wide_resnet': {
        'model_name': 'wide_resnet',
        'dtype': None,
        # For Wide-ResNet
        'resnet_layer_num': None,
        'width_factor': None,
        'num_filters': None,  
        'num_classes': None,
    },
    'bert': {
        'model_name': 'bert',
        'dtype': None,
        # For bert
        'vocab_size': None,
        'hidden_size': None,
        'num_attention_heads': None,
        'intermediate_size': None,
        'num_hidden_layers': None,
        'type_vocab_size': None,
        'tie_word_embeddings': None,
        'gradient_checkpointing': None,
        'add_manual_pipeline_markers': None,
        'pipeline_mp_size': None,
    },
    'gpt': {
        'model_name': 'gpt',
        'dtype': None,
        # For gpt
        'vocab_size': None,
        'hidden_size': None,
        'num_attention_heads': None,
        'intermediate_size': None,
        'num_hidden_layers': None,
        'type_vocab_size': None,
        'tie_word_embeddings': None,
        'gradient_checkpointing': None,
        'add_manual_pipeline_markers': None,
        'pipeline_mp_size': None,
    },
    'moe': {
        'model_name': 'moe',
        'dtype': None,
        # For moe
        'num_hidden_layers': None,
        'hidden_size': None,
        'intermediate_size': None,
        'num_attention_heads': None,
        'max_position_embeddings': None,
        'vocab_size': None,
        'expert_group_size': None,
        'expert_number': None,
        'tie_word_embeddings': None,
        'gradient_checkpointing': None,
        'add_manual_pipeline_markers': None,
        'pipeline_mp_size': None,
    },
}


"""
Default dataset configurations.
"""
dataset_cfgs_proto = {
    'CIFAR10': {
        'dataset_name': 'CIFAR10',
        'batch_size': 256,
        'use_cuda': True
    },
    'none': {
        'dataset_name': 'none',
    }
}


