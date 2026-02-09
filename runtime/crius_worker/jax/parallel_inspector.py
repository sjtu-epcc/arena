#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
An inspector to parse and analyze the parallel strategies of optimized models in alpa.
"""

import os
import re
import json


def mkdir(file_path: str):
    """ To construct the dir of each level (if not existed) in the target file path. """
    _folders = file_path.split('/')
    _path = output_dir
    for _folder in _folders:
        _path = _path + _folder
        if not os.path.exists(_path):
            os.mkdir(_path)


class ParallelInspector:
    """ The class of the inspector to inspect the parallel strategies in alpa. """
    def __init__(self, input_dir: str, output_dir: str, target_file_name: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_file_name = target_file_name
    
    def _parse_internal(self, file_path: str, output_path: str):
        """ To parse a .txt file and get the parallel strategies of each operators in the model. """
        


    
    def parse(self):
        """ Parse all debug info in parallel strategies. """
        # Loop 1: node num & device num
        _node_device_folders = os.listdir(self.input_dir)
        for _nd_folder in _node_device_folders:
            if _nd_folder[0] == '.':
                continue
            _node_device_dir = self.input_dir + '/' + _nd_folder
            # Loop 2: device name
            _device_name_folders = os.listdir(_node_device_dir)
            for _dn_folder in _device_name_folders:
                if _dn_folder[0] == '.':
                    continue
                _device_name_dir = _node_device_dir + '/' + _dn_folder
                # Loop 3: model name & dataset name
                _model_name_folders = os.listdir(_device_name_dir)
                for _mn_folder in _model_name_folders:
                    if _mn_folder[0] == '.':
                        continue
                    _model_name_dir = _device_name_dir + '/' + _mn_folder
                    # Loop 4: model configuration
                    _model_config_folders = os.listdir(_model_name_dir)
                    for _mc_folder in _model_config_folders:
                        if _mc_folder[0] == '.':
                            continue
                        _model_config_dir = _model_name_dir + '/' + _mc_folder
                        # Loop 5: Find the target file
                        _files = os.listdir(_model_config_dir)
                        if _file == self.target_file_name:
                            _path = _model_config_dir + '/' + _file
                            _output_path = './parallel_strategies/' + _nd_folder + '/' + _dn_folder + '/' + _mn_folder + '/' + _mc_folder + '.csv'
                            # Check and make dir
                            mkdir(file_path=_output_path)
                            # Parse
                            self._parse_internal(file_path=_path, output_path=_output_path)
        

