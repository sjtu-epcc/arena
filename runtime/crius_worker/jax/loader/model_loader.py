#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Model loader which loads flax-implemented models.
------------------------------------
- Available models: ['wide_resnet', ]
"""

from model.wide_resnet import WideResNet, wide_resnet_cfgs
from model.bert_model import FlaxBertForMaskedLMModule, BertConfig
from model.gpt_model import FlaxGPTForLMModule
from model.moe_model import FlaxMoEForLMModule, MoEConfig


class ModelLoader:
    def __call__(self, model_cfgs):
        # Load flax models
        print("[I] Loading flax model....")

        model = self.load_flax_model(model_cfgs)

        print("[I] Model '{}' loads sucessfully. Model Info: \n".format(model_cfgs['model_name']))
        print(model, '\n')

        return model

    def load_flax_model(self, model_cfgs):
        if model_cfgs['model_name'] == 'wide_resnet':
            # Wide-ResNet configs
            wrn_cfgs = wide_resnet_cfgs[model_cfgs['resnet_layer_num']]
            wrn_cfgs['dtype'] = model_cfgs['dtype']
            wrn_cfgs['width_factor'] = model_cfgs['width_factor']
            wrn_cfgs['num_filters'] = model_cfgs['num_filters']
            wrn_cfgs['num_classes'] = model_cfgs['num_classes']

            return WideResNet(**wrn_cfgs)
        elif model_cfgs['model_name'] == 'bert' or model_cfgs['model_name'] == 'gpt':   
            # Bert configs
            bert_cfgs = BertConfig(
                vocab_size=model_cfgs['vocab_size'],
                hidden_size=model_cfgs['hidden_size'],
                num_attention_heads=model_cfgs['num_attention_heads'],
                intermediate_size=model_cfgs['intermediate_size'],
                num_hidden_layers=model_cfgs['num_hidden_layers'],
                type_vocab_size=model_cfgs['type_vocab_size'],
                tie_word_embeddings=model_cfgs['tie_word_embeddings'],
                gradient_checkpointing=model_cfgs['gradient_checkpointing'],
                add_manual_pipeline_markers=model_cfgs['add_manual_pipeline_markers'],
                pipeline_mp_size=model_cfgs['pipeline_mp_size'],
            )

            if model_cfgs['model_name'] == 'bert':
                return FlaxBertForMaskedLMModule(bert_cfgs, dtype=model_cfgs['dtype'])
            else:
                return FlaxGPTForLMModule(bert_cfgs, dtype=model_cfgs['dtype'])
        elif  model_cfgs['model_name'] == 'moe':
            # MoE configs
            moe_cfgs = MoEConfig(
                num_hidden_layers=model_cfgs['num_hidden_layers'],
                hidden_size=model_cfgs['hidden_size'],
                intermediate_size=model_cfgs['intermediate_size'],
                num_attention_heads=model_cfgs['num_attention_heads'],
                max_position_embeddings=model_cfgs['max_position_embeddings'],
                vocab_size=model_cfgs['vocab_size'],    
                expert_group_size=model_cfgs['expert_group_size'],
                expert_number=model_cfgs['expert_number'],
                tie_word_embeddings=model_cfgs['tie_word_embeddings'],
                gradient_checkpointing=model_cfgs['gradient_checkpointing'],
                add_manual_pipeline_markers=model_cfgs['add_manual_pipeline_markers'],
                pipeline_mp_size=model_cfgs['pipeline_mp_size'],
            )

            return FlaxMoEForLMModule(moe_cfgs, dtype=model_cfgs['dtype'])
        else:
            raise NotImplementedError("Unsupported flax model type. Please recheck the list of available models.")
