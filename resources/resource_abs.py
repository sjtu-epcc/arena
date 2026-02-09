#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to resource abstraction in the cluster. """

from typing import Sequence
from dataclasses import dataclass
import copy
import numpy as np

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import deepcopy
from macro.macro_def import IDLE_STATUS, EMPTY_JOB_ID, GPU_STATUS_TABLE


@dataclass
class GPUStatus:
    """ The dataclass to collect general GPU runtime status. """
    status: str = None      # Options: ["USED", "RESERVED", "IDLE"]
    used_mem: float = None
    gpu_util: float = None


class GPU:
    """ The abstraction class of one GPU. """
    def __init__(self, uuid: str, alias: str, type: str, 
                 node_id: str, gpu_specs: dict):
        # Basic info
        self.uuid = uuid
        self.alias = alias
        self.type = type
        self.node_id = node_id                             
        # Hardware specs
        self.max_mem = gpu_specs["max_mem"]
        self.max_bw = gpu_specs["max_bw"]
        self.sm_num = gpu_specs["sm_num"]
        # Runtime status
        self.status = IDLE_STATUS
        self.used_mem = 0.0
        self.gpu_util = 0.0
    
    def update_status(self, new_status: GPUStatus):
        """ Update GPU runtime status. """
        assert new_status.status in GPU_STATUS_TABLE
        self.status = new_status.status
        self.used_mem = new_status.used_mem if new_status.used_mem else self.used_mem
        self.gpu_util = new_status.gpu_util if new_status.gpu_util else self.gpu_util


class Node:
    """ The abstraction class of one node. """
    def __init__(self, uuid: str, alias: str, type: str, capacity: int, 
                 gpu_list: Sequence[GPU], ip_addr: str = None, port: int = None):
        # Basic info
        self.uuid = uuid
        self.alias = alias
        self.type = type
        self.capacity = capacity
        # Resources
        self.gpu_list = gpu_list
        # IP address and exposed port of daemon process for crius runtime
        self.ip_addr = ip_addr
        self.port = port
        # records occupied job id of each gpu
        self.gpu_to_job_table = self._init_gpu_to_job_table()   # GPU id -> job id
    
    @property
    def gpu_type(self):
        return self.gpu_list[0].type
    
    @property
    def gpu_ids(self):
        return [_gpu.uuid for _gpu in self.gpu_list]
    
    @property
    def gpu_status(self):
        _stat = dict()
        for _gpu in self.gpu_list:
            _stat[_gpu.uuid] = {
                "type": _gpu.type,
                "status": _gpu.status, 
                "used_job_id": self.gpu_to_job_table[_gpu.uuid]["used"], 
            }
        return _stat

    @property
    def idle_gpu_num(self):
        _num = 0
        for _gpu in self.gpu_list:
            if _gpu.status == IDLE_STATUS:
                _num += 1
        return _num
    
    def _init_gpu_to_job_table(self):
        """ Init table that records occupied job id of each gpu. """
        _table = dict()
        for _gpu in self.gpu_list:
            _table[_gpu.uuid] = { 
                "reserved": EMPTY_JOB_ID, 
                "used": EMPTY_JOB_ID, 
            }
        return _table
    
    def get_gpu_by_uuid(self, uuid):
        """ Return a GPU instance of the uuid if existed in this node. """
        for _gpu in self.gpu_list:
            if _gpu.uuid == uuid:
                return _gpu
        return None


class Resources:
    """
    The wrapper class of various hardware typies (in uuid). When a node 
    appears in related uuid list, PART of the underlying resources are 
    provided for the owner of this instance.
    """
    def __init__(self, node_to_gpu_table: dict):
        # Node - GPU mapping table: {"node_id": ["gpu_id_1", "gpu_id_2", ...], ...}
        self.node_to_gpu_table = node_to_gpu_table

    @property
    def node_id_list(self):
        return [_nid for _nid in self.node_to_gpu_table.keys()]

    @property
    def node_num(self):
        return len(list(self.node_to_gpu_table.keys()))
    
    @property
    def gpu_id_list(self):
        _list = list()
        for _nid in self.node_to_gpu_table.keys():
            _list.extend(self.node_to_gpu_table[_nid])
        return _list
    
    @property
    def gpu_num(self):
        return int(np.sum([
            len(_list) for _list in self.node_to_gpu_table.values()
        ]))
    
    # def __getstate__(self):
    #     """ Customize serialization and deserialization of the class. """
    #     return {
    #         "node_to_gpu_table": deepcopy(self.node_to_gpu_table),
    #     }
    
    # def __setstate__(self, state):
    #     """ Customize serialization and deserialization of the class. """
    #     self.node_to_gpu_table = state["node_to_gpu_table"]

    def __deepcopy__(self, memo):
        """ Hook function for copy.deepcopy(). """
        return self.__class__(deepcopy(self.node_to_gpu_table))
    
    # def deepcopy(self):
    #     """ Deepcopy hook API. """
    #     return Resources(node_to_gpu_table=deepcopy(self.node_to_gpu_table))
