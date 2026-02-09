#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
Implementations of the distributed training job submitted to the cluster scheduler.
"""

from typing import Any
import pickle
from dataclasses import dataclass
from copy import deepcopy as _deepcopy

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resources.resource_abs import Resources
from resources.hardware_specs import supported_gpu_type, NODE_CAPACITY
from utils import deepcopy, deepcopy_pickle
from macro.macro_def import JOB_STATUS_TABLE, JOB_SUBMITTED_STATUS, IS_SUCCEED, IS_FAILED


class ResourceQuota:
    """
    The encapsulated class of the required resource quota of the submmited job specified by the user, 
    maybe modified by the global scheduler and then submit to the resource interface to handle real hardware dispatch.
    -----------------------------------------------------------------------------
    Entries:
        - Job ID
        - GPU num (only support 2^x num)
        - GPU type (only support requiring GPUs with the same type)
        - locality requirement (only support uniformly distributed locality)
    Note:
        - Format of locality requirement: e.g., If 4 GPUs are requested, we only support: 
            - [4]: (on the same node)
            - [2, 2] (uniformly distributed on 2 nodes)
            - [1, 1, 1, 1]: (uniformly distributed on 4 nodes)
        - Memory and CPU core num are not considered here, following GPU-proportional allocation (mentioned in Synergy)
    """
    def __init__(self, 
                 job_id: str = None, 
                 gpu_num: int = 1, 
                 gpu_type: str = None, 
                 locality: list = None
                 ):
        # Job info
        self.job_id = job_id
        # Resources
        self.gpu_num = gpu_num
        self.gpu_type = gpu_type
        self.locality = locality if locality is not None \
            else self._best_locality(gpu_num, NODE_CAPACITY[gpu_type])
        # Inline check
        # self.__check()
    
    def _best_locality(self, gpu_num: int, node_cap: int):
        """ Initialize the job with the best locality. """
        assert node_cap % gpu_num == 0 or gpu_num % node_cap == 0, \
            f"Mismatched spec (node capacity = {node_cap}, GPU num = {gpu_num})"
        return [gpu_num] if gpu_num <= node_cap else [node_cap for _ in range(gpu_num // node_cap)]
    
    # def __check(self):
    #     assert self.job_id is not None
    #     assert (self.gpu_num > 0) and (self.gpu_type in supported_gpu_type), \
    #         f"GPU num: {self.gpu_num} | GPU type: {self.gpu_type}"
    #     assert (self.gpu_num % 2 == 0) or (self.gpu_num == 1)
    #     assert all(_v == self.locality[0] for _v in self.locality)
    
    def __getstate__(self):
        """ Customize serialization and deserialization of the class. """
        return {
            "job_id": self.job_id,
            "gpu_num": self.gpu_num,
            "gpu_type": self.gpu_type,
            "locality": deepcopy(self.locality),
        }
    
    def __setstate__(self, state):
        """ Customize serialization and deserialization of the class. """
        self.job_id = state["job_id"]
        self.gpu_num = state["gpu_num"]
        self.gpu_type = state["gpu_type"]
        self.locality = state["locality"]

    def __deepcopy__(self, memo):
        """ Hook function for copy.deepcopy(). """
        return self.__class__(self.job_id, self.gpu_num, self.gpu_type, 
                              _deepcopy(self.locality, memo))


class Job:
    """
    The class of the dsitributed training job, which encapsulates multiple job workload entries.
    -----------------------------------------------------------------------------
    Entries: 
        ####### Job information #######
        - Job ID
        - Job user ID
        - Job VC (Virtual Cluster) ID 
        - Job submission time
        - Job iter_num (only known when simulating with trace as input)
        - Job required resource quota (in ResourceQuota style, init by user and modified by the scheduler)
        - Job model name
        - Job batch size
        ####### Runtime entries #######
        - Job status
        - Job priority
        - Job allocated resources (in Resources style)
        - Is blocker: Whether the job is the blocker of the cluster.
        - Is opportunism: Whether the job is an opportunism job.
        - Is half-opportunism: Whether the job is a half-opportunism job.
    """
    def __init__(self, job_cfgs):
        # Job info
        self.uuid: str = job_cfgs["job_id"]
        self.alias = job_cfgs["alias"]
        # self.user_id: str = job_cfgs["user_id"]
        # self.vc_id: str = job_cfgs["vc_id"]
        self.sub_time: Any = job_cfgs["sub_time"]
        self.deadline: Any = job_cfgs["deadline"] if "deadline" in job_cfgs else None
        self.iter_num: float = job_cfgs["iter_num"]
        self.resource_quota: ResourceQuota = job_cfgs["resource_quota"]
        self.model_name: str = job_cfgs["model_name"]
        self.batch_size: int = job_cfgs["batch_size"]
        # Runtime entries
        self.status = JOB_SUBMITTED_STATUS
        self.priority = None
        self.resource_alloc = None
        self.exec_time = 0
        self.remained_iter_num = self.iter_num
        self.is_blocker = False
        self.is_opportunism = False
        # Inline check
        # self.__check()

        self.profile_time_budget = None   # The time budget for profiling the job
    
    @property
    def is_cross_nodes(self):
        """ Whether cross nodes. """
        return len(self.resource_quota.locality) > 1
    
    # def __check(self):
    #     assert (self.uuid is not None) and (self.user_id is not None) and (self.vc_id is not None) 
    #     assert (self.sub_time is not None) and (self.iter_num > 0) and (self.model_name is not None) and (self.batch_size is not None)
        
    def __getstate__(self):
        """ Customize serialization and deserialization of the class. """
        return {
            "uuid": self.uuid,
            "alias": self.alias,
            # "user_id": self.user_id,
            # "vc_id": self.vc_id,
            "sub_time": self.sub_time,
            "deadline": self.deadline,
            "iter_num": self.iter_num,
            "resource_quota": deepcopy_pickle(self.resource_quota),
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "status": self.status,
            "priority": self.priority,
            "resource_alloc": deepcopy_pickle(self.resource_alloc),
            "exec_time": self.exec_time,
            "remained_iter_num": self.remained_iter_num,
            "is_blocker": self.is_blocker,
            "is_opportunism": self.is_opportunism,
        }
    
    def __setstate__(self, state):
        """ Customize serialization and deserialization of the class. """
        self.uuid = state["uuid"]
        self.alias = state["alias"]
        # self.user_id = state["user_id"]
        # self.vc_id = state["vc_id"]
        self.sub_time = state["sub_time"]
        self.deadline = state["deadline"]
        self.iter_num = state["iter_num"]
        self.resource_quota = state["resource_quota"]
        self.model_name = state["model_name"]
        self.batch_size = state["batch_size"]
        self.status = state["status"]
        self.priority = state["priority"]
        self.resource_alloc = state["resource_alloc"]
        self.exec_time = state["exec_time"]
        self.remained_iter_num = state["remained_iter_num"]
        self.is_blocker = state["is_blocker"]
        self.is_opportunism = state["is_opportunism"]
        
    def __deepcopy__(self, memo):
        """ Hook function for copy.deepcopy(). """
        # obj = self.__class__({
        #     "job_id": self.uuid, "alias": self.alias, "user_id": self.user_id, 
        #     "vc_id": self.vc_id, "sub_time": self.sub_time, "deadline": self.deadline,
        #     "iter_num": self.iter_num, "resource_quota": _deepcopy(self.resource_quota, memo),
        #     "model_name": self.model_name, "batch_size": self.batch_size
        # })
        obj = self.__class__({
            "job_id": self.uuid, "alias": self.alias, "sub_time": 
            self.sub_time, "deadline": self.deadline,
            "iter_num": self.iter_num, 
            "resource_quota": _deepcopy(self.resource_quota, memo),
            "model_name": self.model_name, "batch_size": self.batch_size
        })
        obj.status = self.status
        obj.priority = self.priority
        obj.resource_alloc = _deepcopy(self.resource_alloc, memo)
        obj.exec_time = self.exec_time
        obj.remained_iter_num = self.remained_iter_num
        obj.is_blocker = self.is_blocker
        obj.is_opportunism = self.is_opportunism

        return obj

    # def deepcopy(self):
    #     """ Deepcopy hook API. """
    #     _job_cfgs = {
    #         "job_id": self.uuid,
    #         "alias": self.alias,
    #         "user_id": self.user_id,
    #         "vc_id": self.vc_id,
    #         "sub_time": self.sub_time,
    #         "deadline": self.deadline,
    #         "iter_num": self.iter_num, 
    #         "resource_quota": _deepcopy(self.resource_quota) if self.resource_quota is not None else None,
    #         "model_name": self.model_name,
    #         "batch_size": self.batch_size,
    #     }
    #     _job = Job(job_cfgs=_job_cfgs)
    #     # Update runtime entries
    #     _job.status = self.status
    #     _job.priority = self.priority
    #     _job.resource_alloc = _deepcopy(self.resource_alloc) if self.resource_alloc is not None else None
    #     _job.remained_iter_num = self.remained_iter_num
    #     _job.is_blocker = self.is_blocker
    #     _job.is_opportunism = self.is_opportunism
    #     return _job
    
    def update_resource_alloc(self, new_resources: Resources):
        """ Update allocated resources of the job. """
        self.resource_alloc = new_resources
    
    def update_status(self, new_status: str):
        """ Update job status. """
        assert new_status in JOB_STATUS_TABLE
        self.status = new_status
    
    def update_remained_iter_num(self, delta_exec_time: int, iter_time: float):
        """ Update the remained iteration num of the job. """
        self.exec_time += delta_exec_time
        self.remained_iter_num = self.iter_num - self.exec_time // iter_time
        if self.remained_iter_num == self.iter_num:
            # Drop
            self.deadline = 0
