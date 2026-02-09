#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to the global scheduler to generate scheduling decision with runtime job workload.
"""

import os
import time
import copy
import pickle
import numpy as np
import threading
from operator import eq
from typing import Sequence, Any, List
from dataclasses import dataclass
from collections import namedtuple

from db_querier import (
    DatabaseQuerier, QueryConfigs)
from job.job import Job, ResourceQuota
from resources.hardware_specs import (
    NODE_CAPACITY, forbid_cross_nodes_gpu_type)
from resources.resource_abs import Resources
from resources.resource_interface import ResourceInterface
from utils import (
    search_list_with_uuid, dict_counter_add, create_entry_and_append_queue, deepcopy, 
    deepcopy_pickle, get_dummy_delta_thr, unique_append, remove_if_exist, 
    is_power_of)
from macro.macro_def import (
    INIT_JOB_NUM, JOB_SUBMITTED_STATUS, JOB_INIT_READY_STATUS, JOB_PENDING_STATUS,
    JOB_RUNNING_STATUS, JOB_COMPLETED_STATUS, JOB_ERROR_STATUS, JOB_STATUS_TABLE,
    IDLE_STATUS, USED_STATUS, EMPTY_JOB_ID, EMPTY_JOB_ALIAS, FAKE_JOB_ID,
    INFEASIBLE_THR, IS_NO_MODIFIED, IS_SHRINKED, IS_HTC, PREC, MAX_SUPPORTED_GPU_NUM,
    ITER_NUM_OF_DOWNGRADE_SEARCH, MAX_SHRINK_TIMES, SCHEDULING_INTERVAL, INFEASIBLE_ITER_TIME,
    MAX_UPGRADE_STEP_NUM, MAX_RESTART_TRIAL_NUM, LOCAL_ETA, GLOBAL_ETA,
    RESCHED_OVERHEAD_WITH_PRUNE, RESCHED_OVERHEAD_WITHOUT_PRUNE, MAX_RESCHED_OVERHEAD_WITH_PRUNE,
    MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE, CKPT_RESUME_OVERHEAD, MAX_CKPT_RESUME_OVERHEAD, AP_PROFILE_OVERHEAD_PER_CONFIG,
)


@dataclass
class AblationOptions:
    """ The dataclass of enabling/disabling partial functions of crius scheduler. """
    # Force using data parallelism in job scheduling
    force_dp: bool = False
    # Disable job gpu num up/down scaling
    disable_scaling: bool = False
    # Disable job gpu type switching
    disable_htc: bool = False
    # Disable opportunistic execution
    disable_opportunistic: bool = False
    # Disable migration
    disable_migration: bool = False
    # Apply ddl-aware in crius
    enable_ddl: bool = False


"""
A collection of job shrink and hadware type change search plan (might be incompleted).
-------------------------------------------------------------------
Entries:
    - crs_table: Current cluster resource status table.
    - target_job: Current instance of the target job (e.g., new arrival job).
    - gain: Total structured marginal gain of the plan.
    - global_mgrt_num: Current remained global migration num.
    - modified_flag: A flag to indicate whether the current step of the plan is shrinked, type-changed or none.
    - mdf_queue: A list of multiple modifying info (job shrink or GPU type change).
"""
SearchPlan = namedtuple("SearchPlan", [
    "crs_table", "target_job", "gain", "global_mgrt_num", "modified_flag", "mdf_queue",
])


class Scheduler:
    """ 
    The class of global scheduler, which retrieve cluster status and job workload to make 
    runtime scheduling decision. 
    """
    def __init__(self, node_pool: dict, supported_gpu_types: List[str], ablation_options: AblationOptions, 
                 is_runtime: bool = False, verbose: bool = False, 
                 dummy_test: bool = False, sched_with_opt: bool = False):
        # Global timer
        self.global_timer = None
        # Job related
        self.submit_init_job_queue = list()
        self.running_job_queue = list()
        self.pending_job_queue = list()
        self.in_profile_job_queue = list()
        self.ended_job_queue = list()
        self.job_registry = dict()              # Job uuid -> instance of this job
        self.decision_queue = list()            # Maintain order of pending and opportunistic executed jobs (FIFO)
        self.resched_time_debt_table = dict()   # Job uuid -> time debt 
        self.resched_num_table = dict()         # Job uuid -> rescheduled times
        self.job_sched_overhead_table = dict()  # Job uuid -> sched overhead
        # Resource interface
        self.resource_interface = ResourceInterface(node_pool)
        self.supported_gpu_types = supported_gpu_types
        # Database querier 
        self.db_querier = DatabaseQuerier(is_runtime)
        # Ablation options
        self.force_dp = ablation_options.force_dp
        self.disable_scaling = ablation_options.disable_scaling
        self.disable_htc = ablation_options.disable_htc
        self.disable_opportunistic = ablation_options.disable_opportunistic
        self.enable_ddl = ablation_options.enable_ddl
        self.ddl_satisfied_job_num = 0
        self.timeout_job_id_list = list()           # Uuid of timeout job
        self.prepend_profile_overhead = os.environ.get("CRIUS_PREPEND_PROFILE_OVERHEAD", "0") == "1"
        self.per_job_profile_overhead = 60 * 4 * 4  # Assume each job can be allocated 16 GPUs at most
        self.required_gpu_num_for_profile = 1       # Assume each job needs 1 GPU to be profiled
        self.profiled_job_to_node_ids = {}          # Job uuid -> [node ids, ], where the job has been profiled
        self.disable_single_device_profiler = os.environ.get("CRIUS_DISABLE_SINGLE_DEVICE_PROFILER", "0") == "1"
        if self.disable_single_device_profiler:
            self.per_job_profile_overhead = 60 * 4
            self.required_gpu_num_for_profile = 4       # FIXME(chunyu): A workaround, should require 16 GPUs but conflict with best-GPU-first scheduling logic

        self.disable_ap_prune = os.environ.get("CRIUS_DISABLE_AP_PRUNE", "0") == "1"
        if self.disable_ap_prune:
            global RESCHED_OVERHEAD_WITHOUT_PRUNE, MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE
            RESCHED_OVERHEAD_WITHOUT_PRUNE=300
            MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE=1200

        # Runtime related
        self.is_runtime = is_runtime            # Runtime scheduling (instead of simulation)
        # Performance
        self.queue_time_table = dict()          # Job uuid -> queuing time
        self.in_profile_time_table = dict()     # Job uuid -> profiled time
        self.exec_time_table = dict()           # Job uuid -> execution time
        self.in_profile_job_queue_end = {}
    
        # Debug mode
        self.verbose = verbose
        # Dummy test
        self._dummy_test = dummy_test
        self._sched_with_opt = sched_with_opt
        self._jau_table = dict()                # Job alias -> job uuid table (for test)
        self._job_init_gpu_num_table = dict()   # Job initial gpu num table (for test)
    
    ######################################
    #     Profiling Related Functions    #
    ######################################

    def _get_job_priority(self, job: Job):
        """ Get job init priority based on job submission timestamp (FIFO order). """
        return job.sub_time
    
    def _is_alloc_feasible(self, target_job: Job, gpu_type: str, locality: Sequence[int],
                           force_opt: bool = False):
        """ 
        Check whether the allocated resources (gpu num, gpu type, locality) is feasible 
        to execute the target job based on profiling results. 
        """
        if self._dummy_test:
            # Dummy test for debug
            return True
        
        if self.enable_ddl and not self._is_crius_ddl_satisfied(target_job, gpu_type, locality):
            # Not satisfiy crius ddl-aware requirement
            return False
        
        # About job
        (model_name, param_num) = target_job.model_name.split("__")
        batch_size = target_job.batch_size
        
        if self._sched_with_opt:
            # (for debug) Make scheduling decisions with throughput of optimal parallelism
            is_fsb, _, _ = self.db_querier.query_db(
                QueryConfigs(len(locality), locality[0], gpu_type, model_name, param_num, 
                            batch_size, only_opt=True),
                force_opt,
            )
        else:
            # Make scheduing decisions with profiled data
            is_fsb, _, _ = self.db_querier.query_db(
                QueryConfigs(len(locality), locality[0], gpu_type, model_name, param_num, 
                            batch_size, only_dp=self.force_dp, search_max=(not self.force_dp)),
                force_opt,
            )
        
        return is_fsb
    
    def _get_delta_thr(self, target_job: Job, new_gpu_type: str, new_gpu_num: int, 
                       new_locality: list, prev_gpu_type: str = None, 
                       prev_gpu_num: int = None, prev_locality: Sequence[int] = None, 
                       query_with_opt: bool = False):
        """ 
        Get the throughput variation of switching gpu type & gpu num & locality based 
        on profiling results. 
        """
        if self._dummy_test:
            # Dummy test for debug
            return get_dummy_delta_thr(target_job, new_gpu_type, new_gpu_num, new_locality, 
                                       prev_gpu_type, prev_gpu_num, prev_locality,
                                       self._job_init_gpu_num_table)
        
        if self.enable_ddl and not self._is_crius_ddl_satisfied(target_job, new_gpu_type, 
                                                                new_locality):
            # Not satisfiy crius ddl-aware requirement
            return INFEASIBLE_THR
        
        # About job
        (model_name, param_num) = target_job.model_name.split("__")
        batch_size = target_job.batch_size
        
        if query_with_opt:
            # Query optimal throughput of running jobs with adaptive parallelism
            assert not prev_gpu_type, \
                "When querying throughput with adaptive parallelism, no prev information " + \
                "should be provided."
            is_fsb, new_thr, _ = self.db_querier.query_db(
                QueryConfigs(len(new_locality), new_locality[0], new_gpu_type, model_name, param_num, 
                            batch_size, only_opt=True), 
                force_opt=True,
            )
            assert is_fsb, \
                f"Got infeasible throughput after adaptive parallelism on allocated resources, " + \
                f"this should not happen: Model name: {model_name} | Param num: {param_num} | " + \
                f"Batch size: {batch_size} | GPU type: {new_gpu_type} | GPU num: {new_gpu_num} | " + \
                f"Locality: {new_locality}."
            
            return new_thr
        
        if self._sched_with_opt:
            # Query database for new resources with throughput of optimal parallelism
            is_fsb, new_thr, _ = self.db_querier.query_db(
                QueryConfigs(len(new_locality), new_locality[0], new_gpu_type, model_name, param_num, 
                            batch_size, only_opt=True),
                force_opt=True
            )
        else:
            # Query database for new resources with throughput of profiled data
            is_fsb, new_thr, _ = self.db_querier.query_db(
                QueryConfigs(len(new_locality), new_locality[0], new_gpu_type, model_name, param_num, 
                            batch_size, only_dp=self.force_dp, search_max=(not self.force_dp))
            )

        if not is_fsb:
            # Infeasible new resources
            return INFEASIBLE_THR
        
        if not prev_gpu_type:
            # No prev resources
            return new_thr
        
        if self._sched_with_opt:
            # Query database for prev resources with throughput of optimal parallelism
            is_fsb, prev_thr, _ = self.db_querier.query_db(
                QueryConfigs(len(prev_locality), prev_locality[0], prev_gpu_type, model_name, param_num, 
                            batch_size, only_opt=True),
                force_opt=True,
            )
        else:
            # Query database for new resources with throughput of profiled data
            is_fsb, prev_thr, _ = self.db_querier.query_db(
                QueryConfigs(len(prev_locality), prev_locality[0], prev_gpu_type, model_name, param_num, 
                            batch_size, only_dp=self.force_dp, search_max=(not self.force_dp))
            )

        if not is_fsb:
            # Previously allocated resources is infeasible (e.g., intermediate results during 
            # job downgrade search)
            return (new_thr - INFEASIBLE_THR)

        # assert is_fsb, \
        #     f"Previously allocated resources should be feasible, but got infeasible with," + \
        #     f"this should not happen: Model name: {model_name} | Param num: {param_num} | " + \
        #     f"Batch size: {batch_size} | GPU type: {prev_gpu_type} | GPU num: {prev_gpu_num} | " + \
        #     f"Locality: {prev_locality}."
        
        return (new_thr - prev_thr)
    
    def _get_iter_time(self, model_name: str, batch_size: int, gpu_type: str, 
                       locality: Sequence[int], force_opt: bool = False):
        """ 
        Get the e2e iteration time of the model with specified resources based on profilng results. 
        """
        if self._dummy_test:
            # Dummy test for debug
            return 1.0
        
        # About job
        (model_name, param_num) = model_name.split("__")
        
        if self._sched_with_opt or force_opt:
            # (for debug) Query database with throughput of optimal parallelism
            is_fsb, thr, _ = self.db_querier.query_db(
                QueryConfigs(len(locality), locality[0], gpu_type, model_name, param_num, 
                            batch_size, only_opt=True),
                force_opt=force_opt,
            )
        else:
            # Query database with throughput of profiled data
            is_fsb, thr, _ = self.db_querier.query_db(
                QueryConfigs(len(locality), locality[0], gpu_type, model_name, param_num, 
                            batch_size, only_dp=self.force_dp, search_max=(not self.force_dp))
            )
        
        return round(batch_size / thr , PREC) if is_fsb else INFEASIBLE_ITER_TIME

    ######################################
    #      Timer Related Functions       #
    ######################################

    def init_timer(self, timestamp: int):
        """ Init the global timer. """
        self.global_timer = timestamp

    def _update_timer(self):
        """ Update timer to the next rescheduling event. """
        self.global_timer += SCHEDULING_INTERVAL
    
    def _is_crius_ddl_satisfied(self, target_job: Job, gpu_type: str, locality: Sequence[int]):
        """ 
        Check whether the deadline can be satisfied with the given gpu num, only used in 
        scalability analysis of crius. 
        """
        iter_time = self._get_iter_time(target_job.model_name, target_job.batch_size, 
                                        gpu_type, locality, force_opt=True)
        
        if self.global_timer - target_job.sub_time <= SCHEDULING_INTERVAL:
            # New job
            resched_overhead = min(RESCHED_OVERHEAD_WITH_PRUNE * target_job.resource_quota.gpu_num, 
                                MAX_RESCHED_OVERHEAD_WITH_PRUNE)
        else:
            resched_overhead = self.resched_time_debt_table[target_job.uuid] \
                    if target_job.uuid in self.resched_time_debt_table else 0
        
        # print(
        #     self.global_timer + resched_overhead + SCHEDULING_INTERVAL +
        #                  int(target_job.remained_iter_num * iter_time)
        # )
        # print(target_job.deadline)
        
        return True if (iter_time != INFEASIBLE_ITER_TIME and 
                        (self.global_timer + resched_overhead + SCHEDULING_INTERVAL +
                         int(target_job.remained_iter_num * iter_time)) 
                        <= target_job.deadline) else False
    
    #################################################
    #    Simulating Performance Related Functions   #
    #################################################

    def _get_cluster_perf(self, makespan_list: Sequence[int], jct_list: Sequence[int], 
                          queue_time_list: Sequence[int], new_resched_jid_list: Sequence[str]):
        """ 
        Get the cluster performance metrics in the current round and 
        consider rescheduling overhead. 
        """
        # Monitored metrics
        metrics = {"thr": 0.0, "avg_job_thr_per_gpu": 0.0, "makespan_list": makespan_list, 
                   "jct_list": jct_list, "queue_time_list": queue_time_list}

        # Rollback global timer
        self.global_timer -= SCHEDULING_INTERVAL
        
        for _job in self.running_job_queue:
            query_with_opt = os.environ.get("CRIUS_FORCE_DEPLOY_WITH_AP", "0") == "1"
            _thr = self._get_delta_thr(_job, _job.resource_quota.gpu_type, 
                                       _job.resource_quota.gpu_num, 
                                       _job.resource_quota.locality, 
                                       query_with_opt=query_with_opt)
            _thr = 0.0 if _thr == INFEASIBLE_THR else _thr
            # assert _thr != INFEASIBLE_THR, \
            #     f"Throughput of a running job '{_job.alias}' " + \
            #     f"({_job.resource_quota.gpu_type}, {_job.resource_quota.gpu_num}, " + \
            #     f"{_job.resource_quota.locality}) should not be infeasible."

            if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                # Use larger rescheduling overhead upper bound in revision mode
                # MAX_RESCHED_OVERHEAD_WITH_PRUNE = 160
                pass
            
            # Consider resched overhead
            if _job.uuid in new_resched_jid_list:
                gpu_num = _job.resource_quota.gpu_num
                per_gpu_overhead = RESCHED_OVERHEAD_WITH_PRUNE \
                    if (os.environ.get("SCHED_POLICY", "crius").split("-")[0] == "crius"
                        and not self.disable_ap_prune) \
                    else RESCHED_OVERHEAD_WITHOUT_PRUNE
                max_tune_time = MAX_RESCHED_OVERHEAD_WITH_PRUNE \
                    if (os.environ.get("SCHED_POLICY", "crius").split("-")[0] == "crius"
                        and not self.disable_ap_prune) \
                        else MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE
                tune_time = min(per_gpu_overhead * gpu_num, max_tune_time)
                # Ckpt-resume overhead
                tune_time += min(CKPT_RESUME_OVERHEAD * gpu_num, MAX_CKPT_RESUME_OVERHEAD)

                if tune_time <= SCHEDULING_INTERVAL:
                    # Can be tuned in this round
                    _thr *= (SCHEDULING_INTERVAL - tune_time) / SCHEDULING_INTERVAL
                    if _job.uuid in self.resched_time_debt_table:
                        # Tuning is not completed in the last scheduling round
                        self.resched_time_debt_table.pop(_job.uuid)
                else:
                    # Cause time debt, rewrite if already exists
                    self.resched_time_debt_table[_job.uuid] = tune_time

            if _job.uuid in self.resched_time_debt_table:
                # With time debt
                time_debt = self.resched_time_debt_table[_job.uuid]
                if time_debt <= SCHEDULING_INTERVAL:
                    # Can be tuned in this round
                    _thr *= (SCHEDULING_INTERVAL - time_debt) / SCHEDULING_INTERVAL
                    self.resched_time_debt_table.pop(_job.uuid)
                else:
                    # Cannot be tuned, reduce time debt
                    _thr = 0
                    self.resched_time_debt_table[_job.uuid] -= SCHEDULING_INTERVAL

            metrics["thr"] += _thr
            metrics["avg_job_thr_per_gpu"] += _thr / _job.resource_quota.gpu_num
        
        if len(self.running_job_queue) > 0:
            metrics["avg_job_thr_per_gpu"] = metrics["avg_job_thr_per_gpu"] / len(self.running_job_queue)
        
        # Resume global timer
        self.global_timer += SCHEDULING_INTERVAL
        
        return metrics

    def _get_end_job_metrics(self):
        """ Get the runtime metrics of ended jobs in the last scheduling round. """
        makespan_list, jct_list, queue_time_list = list(), list(), list()
        
        for _job in self.ended_job_queue:
            # Makespan
            makespan_list.append(
                self.global_timer - _job.sub_time
            )
            # Job completion time = queuing time + execution time
            jct_list.append(
                self.global_timer - _job.sub_time
            )
            # Queuing time
            assert _job.uuid in self.queue_time_table, \
                f"Job {_job.alias} should be recorded in queue time table once " + \
                f"submitted to the scheduler."
            queuing_time = self.queue_time_table[_job.uuid]
            if self.prepend_profile_overhead:
                queuing_time += self.in_profile_time_table[_job.uuid]
            queue_time_list.append(queuing_time)
        
        return makespan_list, jct_list, queue_time_list
    
    def _update_queue_time_table(self):
        """ Update queuing time of pending jobs in the last round. """
        for _job in self.pending_job_queue:
            self.queue_time_table = dict_counter_add(self.queue_time_table, _job.uuid, SCHEDULING_INTERVAL)
        for _job in self.in_profile_job_queue:
            self.in_profile_time_table = dict_counter_add(self.in_profile_time_table, _job.uuid, SCHEDULING_INTERVAL)
        for _job in self.running_job_queue:
            if _job.uuid not in self.queue_time_table:
                # Record this new job
                self.queue_time_table[_job.uuid] = 0
        
    def _update_executed_time_table(self):
        """ Update executed time of running jobs in the last round. """
        for _job in self.running_job_queue:
            self.exec_time_table = dict_counter_add(self.exec_time_table, _job.uuid, SCHEDULING_INTERVAL)

    ######################################
    #       Job Related Functions        #
    ######################################
    
    def _sort_jobs(self, jobs: Sequence[Job]):
        """ Sort job queue by the priority of each job in increasing order. """
        return sorted(jobs, key=lambda x: x.priority)

    def _get_best_locality(self, gpu_num: int, gpu_type: str, 
                           prev_locality: Sequence[int] = None, 
                           local_mgrt_num: int = None):
        """ 
        Get the best locality with the constraint of maximal migration num from previous locality. 
        If prev_locality[0] (#gpu-per-node) * (#mgrt + 1) >= #gpus, we can retrieve the optimal 
        locality with single-node or fully-occupiled on multiple nodes.
        """
        if (prev_locality is None or 
            prev_locality[0] * (local_mgrt_num + 1) >= min(gpu_num, 
                                                           NODE_CAPACITY[gpu_type])):
            # Optimal locality is retrieved
            return [gpu_num] if gpu_num <= NODE_CAPACITY[gpu_type] else \
                [NODE_CAPACITY[gpu_type] for _ in range(gpu_num // NODE_CAPACITY[gpu_type])]
        else:
            # Partial occupation on each node
            _per_node_quota = prev_locality[0] * (local_mgrt_num + 1)
            return [_per_node_quota for _ in range(gpu_num // _per_node_quota)]
    
    def submit_job(self, job: Job, sub_time: Any = None):
        """ Submit a job to the scheduler. """
        assert job.status == JOB_SUBMITTED_STATUS, \
            f"The submitted job should be in {JOB_SUBMITTED_STATUS} status, got {job.status}."
        job.sub_time = sub_time if sub_time else time.time()
        job.priority = self._get_job_priority(job)
        job.status = JOB_INIT_READY_STATUS
        self.submit_init_job_queue.append(job)
        if job.alias in self._jau_table:
            raise RuntimeWarning(f"Error: Job alias '{job.alias}' has already been used.")
        self._jau_table[job.alias] = job.uuid
        self._job_init_gpu_num_table[job.uuid] = job.resource_quota.gpu_num

    def _end_job(self, job: Job):
        """ End one running job. """
        assert job in self.running_job_queue, \
            f"Job '{job.alias}' (alias) is not in the running queue, temporarily only support end running jobs."
        job.update_status(JOB_COMPLETED_STATUS)
        self.running_job_queue.remove(job)
        assert job not in self.ended_job_queue, \
            f"A newly ended job {job.alias} should not be in ended job queue before."
        self.ended_job_queue.append(job)

    def _get_job_by_uuid_internal(self, uuid: str, job_status: str):
        """ Get job from varying job queues. """
        if job_status == JOB_INIT_READY_STATUS:
            job = search_list_with_uuid(self.submit_init_job_queue, 
                                         uuid)
            if self.prepend_profile_overhead and job is None:
                job = search_list_with_uuid(self.in_profile_job_queue, uuid)
            return job
        elif job_status == JOB_RUNNING_STATUS:
            return search_list_with_uuid(self.running_job_queue, 
                                         uuid)
        elif job_status == JOB_PENDING_STATUS:
            return search_list_with_uuid(self.pending_job_queue, 
                                         uuid)
        elif (job_status == JOB_COMPLETED_STATUS or 
              job_status == JOB_ERROR_STATUS):
            return search_list_with_uuid(self.ended_job_queue, 
                                         uuid)
        else:
            return None

    def _gen_fake_job(self, uuid: str):
        """ Construct one fake empty job instance. """
        _job = Job(
            {
                "job_id": uuid,
                "alias": "fake_job",
                # "user_id": "default_user",
                # "vc_id": "default_vc",
                "sub_time": 100,
                "iter_num": 100, 
                "resource_quota": ResourceQuota(job_id=uuid, gpu_num=1, 
                                                gpu_type="v100"),
                "model_name": "default_model",
                "batch_size": 32,
            }
        )
        _job.status = None
        return _job
    
    def get_job_by_uuid(self, uuid: str):
        """ Get job instance with the given uuid. """
        if FAKE_JOB_ID in str(uuid):
            # Fake job created in introspective migration search
            return self._gen_fake_job(uuid)
        
        if uuid in self.job_registry:
            # Registered job
            return self.job_registry[uuid]
        
        for job_status in JOB_STATUS_TABLE:
            job = self._get_job_by_uuid_internal(uuid, job_status)
            if job:
                self.job_registry[uuid] = job
                return job
        return None

    def _get_run_job_ids(self, target_job: Job, crs_table: dict):
        """
        Scan the entire crs_table to construct a job uuid list includes 
        both the target job and all running jobs (include opportunism 
        job, but not those who might be suspended by the target job). 
        """
        _list = list()
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if (crs_table[_nid][_gid]["status"] == USED_STATUS and 
                    crs_table[_nid][_gid]["used_job_id"] not in _list):
                    
                    job_id = crs_table[_nid][_gid]["used_job_id"]
                    job = self.get_job_by_uuid(job_id)
                    if self.prepend_profile_overhead and job.profile_time_budget > 0:
                        # This job is being profiled and occupy resources
                        continue
                    
                    # Check the crs_table to get the available job, filtering out the 
                    # opportunism jobs that MAYBE SUSPENDED by the target pending job 
                    # in pending jobs restart trial, which occupied resources have been 
                    # tagged as 'idle'.
                    _list.append(job_id)
        assert target_job.uuid not in _list, \
            f"Both the newly arrived and the pending jobs cannot be registered in crs_table."
        _list.append(target_job.uuid)
        
        return _list
  
    def _update_run_jobs(self, crs_table: dict):
        """ 
        Scan the entire crs_table to update the allocated resources and the 
        resource quota of running jobs based on the scheduling decision. 
        """
        for _job in self.running_job_queue:
            gpu_type = None
            locality = list()
            node_to_gpu_table = dict()          # Allocated node -> gpu mapping table
            
            for _nid in crs_table:
                per_node_quota = 0
                for _gid in crs_table[_nid]:
                    if crs_table[_nid][_gid]["used_job_id"] == _job.uuid:
                        per_node_quota += 1
                        if not gpu_type:
                            gpu_type = crs_table[_nid][_gid]["type"]
                        assert gpu_type == crs_table[_nid][_gid]["type"], \
                            f"Currently mixed GPU type of one job is not supported, got {gpu_type} " + \
                            f"and {crs_table[_nid][_gid]['type']}."
                        node_to_gpu_table = create_entry_and_append_queue(node_to_gpu_table, _nid, _gid)
                if per_node_quota > 0:
                    locality.append(per_node_quota)
            assert (FAKE_JOB_ID in str(_job.uuid)) or all([locality[i] == locality[0] for i in range(len(locality))])
            assert len(locality) > 0, f"Job {_job.alias} not found in crs_table."
            
            gpu_num = int(np.sum(locality))
            # Note that the occupied GPU num of fake job (bubbles) may be the number 
            # other than 1 or divided by 2.
            assert (FAKE_JOB_ID in str(_job.uuid)) or (gpu_num == 1) or (gpu_num % 2 == 0)
            
            # Update resource quota of the job
            _job.resource_quota = ResourceQuota(_job.uuid, gpu_num, gpu_type, locality)
            # Update the allocated resources of the job
            _job.update_resource_alloc(new_resources=Resources(node_to_gpu_table))

    def _print_modification(self, idx: int, fsb_plans: Sequence[SearchPlan], 
                                  crs_table: dict):
        """ Print a series of modifying info. """
        mdf_queue = fsb_plans[idx].mdf_queue
        if len(mdf_queue) == 0:
            print("     - Modifying Info: No modifying operation is performed.")
        
        for _info in mdf_queue:
            job_alias = self.get_job_by_uuid(uuid=_info[1][0]).alias
            if _info[0] == IS_SHRINKED:
                print(f"     - Modifying info ({_info[0]}): Job alias: {job_alias} " + \
                      f"| GPU type: {_info[1][1]} | Prev GPU num: {_info[1][2]} | " + \
                      f"Prev locality: {_info[1][3]} | New GPU num: {_info[1][4]} | " + \
                      f"New locality: {_info[1][5]}")
            else:
                print(f"     - Modifying info ({_info[0]}): Job alias: {job_alias} " + \
                      f"| Prev GPU type: {_info[1][1]} | Prev GPU num: {_info[1][2]} | " + \
                      f"Prev locality: {_info[1][3]} | New GPU type: {_info[1][4]} | " + \
                      f"New GPU num: {_info[1][5]} | New locality: {_info[1][6]}")

        if self.verbose:
            self._parse_crs_table_change(crs_table, fsb_plans[idx].crs_table)

    def _update_and_check_end(self):
        """ 
        Update the remained iteration num of all running jobs in the last 
        scheduling round (interval), checking and ending the jobs that 
        should be ended (since the iteration num has been run out).
        We approximate the _delta_iter_num = np.around(SCHEDULING_INTERVAL / time per iteration).
        """
        print("")
        print("[I] Updating the remained iteration num of running jobs and check job ending...")

        ended_job_id_list = list()
        for _job in self.running_job_queue:
            if _job.uuid in self.resched_time_debt_table:
                # Still in reshced tuing process
                continue
            
            _iter_time = self._get_iter_time(_job.model_name, _job.batch_size, 
                                             _job.resource_quota.gpu_type,
                                             _job.resource_quota.locality,
                                             force_opt=True)
            assert _iter_time != INFEASIBLE_THR, \
                f"The iteration time of job '{_job.alias}' (alias) is infeasible (probably " + \
                f"not profiled in the database)."
            
            _job.update_remained_iter_num(SCHEDULING_INTERVAL, _iter_time)
            if _job.remained_iter_num <= 0:
                # This job is completed and should be ended
                ended_job_id_list.append(_job.uuid)
                # Update ddl satisfied job num
                if self.global_timer <= _job.deadline:
                    self.ddl_satisfied_job_num += 1
        
        print("[I] The following jobs are completed and ended:", [
            self.get_job_by_uuid(uuid=_uuid).alias for _uuid in ended_job_id_list
        ])
        for _jid in ended_job_id_list:
            self._end_job(self.get_job_by_uuid(_jid))
        
        # crs_table = self.resource_interface.get_crs_table()
        # for node_id in crs_table:
        #     node = self.resource_interface.node_pool[node_id]
        #     if node.idle_gpu_num == node.capacity:
        #         continue

        #     job_names = []
        #     for gpu_id in crs_table[node_id]:
        #         if node.gpu_to_job_table[gpu_id]["used"] == EMPTY_JOB_ID:
        #             continue

        #         job = self.get_job_by_uuid(node.gpu_to_job_table[gpu_id]["used"])
        #         if job.alias not in job_names:
        #             job_names.append(job.alias)

        #     print(f"Node: {node.alias} | Jobs: {job_names}")

        if self.prepend_profile_overhead:
            to_complete_profile_jobs = []
            # Update job profile budget
            for job in self.in_profile_job_queue:
                assert job.profile_time_budget > 0, "Should have been poped in the previous round."
                if job.uuid not in self.profiled_job_to_node_ids:
                    # Not start profile
                    continue

                job.profile_time_budget -= SCHEDULING_INTERVAL
                if job.profile_time_budget <= 0:
                    # Complete
                    print(f"[I] Job {job.alias} has completed profiling, all resources are released.")
                    occuied_node_ids = self.profiled_job_to_node_ids[job.uuid]
                    crs_table = self.resource_interface.get_crs_table()
                    for gpu_type in occuied_node_ids:
                        for node_id in occuied_node_ids[gpu_type]:
                            node = self.resource_interface.node_pool[node_id]
                            # print(f"---------> Node {node.alias} would be released by job {job.alias} after profiling.")
                            for gpu_id in crs_table[node_id]:
                                crs_table[node_id][gpu_id]["status"] = IDLE_STATUS
                                crs_table[node_id][gpu_id]["used_job_id"] = EMPTY_JOB_ID
                    
                    self.resource_interface.apply_sched(crs_table)
                    del self.profiled_job_to_node_ids[job.uuid]
                    to_complete_profile_jobs.append(job)

            # print(f"\n\n(Before pop out) Jobs in profile queue: {[_job.alias for _job in self.in_profile_job_queue]}")

            # Relaunch
            self.submit_init_job_queue = to_complete_profile_jobs + self.submit_init_job_queue
            for job in to_complete_profile_jobs:
                # print(f"----> Job {job.alias} has been removed.")
                self.in_profile_job_queue.remove(job)
            
            # print(f"\n\n(After pop out) Jobs in profile queue: {[_job.alias for _job in self.in_profile_job_queue]}")

            # crs_table = self.resource_interface.get_crs_table()
            # for node_id in crs_table:
            #     node = self.resource_interface.node_pool[node_id]
            #     if node.idle_gpu_num == node.capacity:
            #         continue

            #     job_names = []
            #     for gpu_id in crs_table[node_id]:
            #         if node.gpu_to_job_table[gpu_id]["used"] == EMPTY_JOB_ID:
            #             continue

            #         job = self.get_job_by_uuid(node.gpu_to_job_table[gpu_id]["used"])
            #         if job.alias not in job_names:
            #             job_names.append(job.alias)

            #     print(f"Node: {node.alias} | Jobs: {job_names}")


    def _eliminate_unnecessary_suspend_resume(self, prev_crs_table: dict):
        """ 
        Eliminate unnecessary job suspend/resume with unchanged resource quota 
        but changed allocated resources. For instance, job 1 is allocated with 
        node 1 nad node 2 with GPU type A, then after scheduling it is reallocated
        with node 1 and node 3 with GPU type A, while node 2 is assigned to a new
        job 2. In this case, we reallocate job 2 to node 3 and eliminate suspend/resume
        of job 1 by still assigning node 1 and node 2 to it.
        """
        crs_table = self.resource_interface.get_crs_table()

        # self._parse_crs_table_change(prev_crs_table, crs_table)
        
        for _job in self.running_job_queue:
            # Node uuid -> [gpu uuid 1, ...]
            prev_resources = self._get_occ_resources(prev_crs_table, _job.uuid, 
                                                     print_option=False)
            prev_node_id_list = list(prev_resources.keys())
            new_resources = self._get_occ_resources(crs_table, _job.uuid, 
                                                     print_option=False)
            new_node_id_list = list(new_resources.keys())
            if len(prev_node_id_list) == 0:
                # New job
                continue
            if prev_node_id_list == new_node_id_list:
                # Unchanged resource allocation
                continue
            if (_job.resource_quota.gpu_num // len(prev_node_id_list) < 
                NODE_CAPACITY[_job.resource_quota.gpu_type]):
                # Not fully occupy nodes
                continue
            
            # Analyze difference
            if (self.resource_interface.node_pool[new_node_id_list[0]].gpu_type != 
                self.resource_interface.node_pool[prev_node_id_list[0]].gpu_type):
                # Different gpu type
                continue
            if (len(prev_node_id_list) != len(new_node_id_list) or 
                (len(prev_resources[prev_node_id_list[0]]) != 
                 len(new_resources[new_node_id_list[0]]))):
                # Different node num or per-node gpu num
                continue
            
            # Need eliminate 
            to_switch_node_id_list = list()
            for _node_id in new_node_id_list:
                if _node_id not in prev_node_id_list:
                    to_switch_node_id_list.append(_node_id)
            
            print([self.resource_interface.node_pool[_nid].alias for _nid in prev_node_id_list])
            print([self.resource_interface.node_pool[_nid].alias for _nid in new_node_id_list])
            
            print(f"[I] Switching nodes for job {_job.alias} (GPU type: {_job.resource_quota.gpu_type}), " + 
                  f"currently allocated nodes are: {[self.resource_interface.node_pool[_nid].alias for _nid in new_node_id_list]}, " + 
                  f"target nodes are: {[self.resource_interface.node_pool[_nid].alias for _nid in prev_node_id_list]}")

            switched_node_id_list = list()
            for _to_switch_node_id in to_switch_node_id_list:
                gpu_id_list = list(crs_table[_to_switch_node_id].keys())
                for _node_id in prev_node_id_list:
                    if _node_id in new_node_id_list or _node_id in switched_node_id_list:
                        continue
                    # Switch nodes
                    switched_node_id_list.append(_node_id)
                    for _i, _gpu_id in enumerate(crs_table[_node_id]):
                        assert (crs_table[_node_id][_gpu_id]["type"] == 
                                _job.resource_quota.gpu_type), f"Mismatched GPU type."
                        assert crs_table[_node_id][_gpu_id]["status"] == USED_STATUS, \
                            f"To-switch GPUs should be occupied by other jobs."
                        crs_table[_to_switch_node_id][gpu_id_list[_i]]["used_job_id"] = \
                            crs_table[_node_id][_gpu_id]["used_job_id"]
                        crs_table[_node_id][_gpu_id]["used_job_id"] = _job.uuid
                    break

        # for node_id in crs_table:
        #     node = self.resource_interface.node_pool[node_id]
        #     if node.idle_gpu_num == node.capacity:
        #         continue

        #     job_names = []
        #     for gpu_id in crs_table[node_id]:
        #         if node.gpu_to_job_table[gpu_id]["used"] == EMPTY_JOB_ID:
        #             continue

        #         job = self.get_job_by_uuid(node.gpu_to_job_table[gpu_id]["used"])
        #         if job.alias not in job_names:
        #             job_names.append(job.alias)

        #     print(f"Node: {node.alias} | Jobs: {job_names}")
            
        
        # Update real cluster resources
        self.resource_interface.apply_sched(crs_table)
        # Update the resource status of running jobs
        self._update_run_jobs(crs_table)

    ######################################
    #     Temporary Related Functions    #
    ######################################

    def _tmp_occ_nids(self, job_id: str, crs_table: dict):
        """ 
        Temporarily get the occupied node id list of the target job 
        (recently added) during the placement trial. 
        """
        _list = list()
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if (crs_table[_nid][_gid]["used_job_id"] == job_id and 
                    _nid not in _list):
                    _list.append(_nid)
        return _list
    
    def _tmp_occ_gpu_num(self, job_id: str, crs_table: dict):
        """ 
        Temporarily get the occupied GPU num (maybe modified by 
        shrink operation) of the job during the placement trial.
        """
        _num = 0
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]['used_job_id'] == job_id:
                    _num += 1
        return _num

    def _tmp_job_gpu_type(self, job_id: str, crs_table: dict):
        """ 
        Temporarily get the occupied GPU type (maybe modified by 
        type change operation) of the job during the placement trial.
        """
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]['used_job_id'] == job_id:
                    return self.resource_interface.node_pool[_nid].gpu_type
        return None
    
    def _get_job_rt_stat_v1(self, job_id: str, target_job: Job, crs_table: dict):
        """ 
        Temporarily get the latest job GPU num and locality statistical 
        information. 
        """
        if target_job is not None and job_id == target_job.uuid:
            # If the target job has not been registered in crs_table (e.g., new job), use the information recorded before this scheduling round.
            _tmp_job_gpu_type = self._tmp_job_gpu_type(job_id=job_id, crs_table=crs_table)
            target_job_gpu_type = _tmp_job_gpu_type if _tmp_job_gpu_type is not None else target_job.resource_quota.gpu_type
            _tmp_occ_gpu_num = self._tmp_occ_gpu_num(job_id=job_id, crs_table=crs_table)
            target_job_gpu_num = _tmp_occ_gpu_num if _tmp_occ_gpu_num > 0 else target_job.resource_quota.gpu_num
            _tmp_job_node_num = len(self._tmp_occ_nids(job_id=job_id, crs_table=crs_table))
            target_job_node_num = _tmp_job_node_num if _tmp_job_node_num > 0 else max(1, int(target_job_gpu_num / NODE_CAPACITY[target_job_gpu_type]))
            target_job_locality = [int(target_job_gpu_num / target_job_node_num) for i in range(target_job_node_num)]
            return target_job_gpu_type, target_job_gpu_num, target_job_locality, []
        else:
            # Else, use job information in crs_table since this info must be latest.
            # Note that the placement of the _job could be cleared (e.g., opportunism job in pending jobs restart).
            _job = self.get_job_by_uuid(uuid=job_id)
            _tmp_job_gpu_type = self._tmp_job_gpu_type(job_id=job_id, crs_table=crs_table)
            job_gpu_type = _tmp_job_gpu_type if _tmp_job_gpu_type is not None else _job.resource_quota.gpu_type
            _tmp_occ_gpu_num = self._tmp_occ_gpu_num(job_id=job_id, crs_table=crs_table)
            job_gpu_num = _tmp_occ_gpu_num if _tmp_occ_gpu_num > 0 else _job.resource_quota.gpu_num
            job_node_id_list = self._tmp_occ_nids(job_id=job_id, crs_table=crs_table)
            _tmp_job_node_num = len(job_node_id_list)
            job_node_num = _tmp_job_node_num if _tmp_job_node_num > 0 else max(1, int(job_gpu_num / NODE_CAPACITY[job_gpu_type]))
            job_locality = [int(job_gpu_num / job_node_num) for i in range(job_node_num)]
            assert job_locality[0] <= NODE_CAPACITY[job_gpu_type]
            return job_gpu_type, job_gpu_num, job_locality, job_node_id_list
    
    def _get_job_rt_stat(self, job_id: str, crs_table: dict, overwrite_job: Job = None):
        """ 
        Temporarily get the latest job GPU num and locality statistical 
        information. 
        Args:
            - job_id: The uuid of the job to be queried.
            - crs_table: Introduced in resource_interface.py.
            - overwrite_job: The temporary `Job` instance to store intermediate results 
                             (e.g., gpu type change). If not `None`, overwrite the 
                             querying results based on this job instance. Note that this 
                             overwrite_job instance (runtime) can be different from the
                             instance from get_job_by_uuid(overwrite_job.uuid), which is
                             the instance updated in the last round.
        """
        
        def __get_stat(job_id: str, job: Job = None):
            """ Get job stat from (1) crs_table or (2) job instance. """
            # Type
            _gpu_type = self._tmp_job_gpu_type(job_id, crs_table)
            gpu_type = _gpu_type if _gpu_type else job.resource_quota.gpu_type
            # Num
            _gpu_num = self._tmp_occ_gpu_num(job_id, crs_table)
            gpu_num = _gpu_num if _gpu_num > 0 else job.resource_quota.gpu_num
            assert is_power_of(2, gpu_num), \
                f"Job {job.alias}: GPU num should be the power of 2, got {gpu_num}."
            # Locality
            nid_list = self._tmp_occ_nids(job_id, crs_table)
            _node_num = len(nid_list) if len(nid_list) > 0 else max(1, gpu_num // NODE_CAPACITY[gpu_type])
            locality = [gpu_num // _node_num for _ in range(_node_num)]

            return (gpu_type, gpu_num, locality, nid_list)
        
        if overwrite_job:
            # Intermediate results of different searching depth 
            # during one-round scheduling.
            return __get_stat(overwrite_job.uuid, overwrite_job)

        # Normally get the job instance (updated in the last 
        # scheduling round) in case that the job is not registered
        # or cleared in crs_table.
        job = self.get_job_by_uuid(job_id)
        
        return __get_stat(job_id, job)

    ######################################
    #     Resource Related Functions     #
    ######################################

    def _get_bubble_num(self, node_table: dict):
        """ Get the bubble num of the target node. """
        _num = 0
        for _gid in node_table:
            if node_table[_gid]["status"] == IDLE_STATUS:
                _num += 1
        return _num

    def _sort_partial_occ_nodes(self, crs_table: dict, gpu_type: str, gpu_num: int):
        """ 
        Sort all nodes (given gpu type) in decreasing order (for load balancing) of 
        bubble (idle gpu slot) num, removing idle nodes and fully-occupied nodes. 
        -------------------------------------------------------------
        Return:
            - node_bubble_num_list: A nested list of sorted and partially occupied nodes.
            - idle_nids: An uuid list of idle nodes.
        """
        node_cap = NODE_CAPACITY[gpu_type]
        node_bubble_num_list = list()       # A nested list of node - bubble status
        idle_nids = list()

        for _nid in crs_table:
            if self.resource_interface.node_pool[_nid].gpu_type != gpu_type:
                # Skip different gpu type
                continue
            bubble_num = self._get_bubble_num(crs_table[_nid])
            if bubble_num > 0 and bubble_num < node_cap:
                # Partially occupied
                node_bubble_num_list.append([_nid, bubble_num])
            if bubble_num == node_cap:
                # Idle
                idle_nids.append(_nid)
        
        node_bubble_num_list = sorted(node_bubble_num_list, key=lambda x: x[1], reverse=True)
        # Cut for exactly exceeding the demanding gpu num
        _bubble_num, idx = 0, -1
        for _i, _rec in enumerate(node_bubble_num_list):
            _bubble_num += _rec[1]
            if _bubble_num >= gpu_num:
                idx = _i
                break

        return (node_bubble_num_list[:(idx + 1)], idle_nids) if idx > -1 else (node_bubble_num_list, idle_nids)
    
    def _record_and_clear_nodes(self, node_id_list: Sequence[str], crs_table: dict):
        """ Record the job info of the nodes and clear them. """
        job_to_node_num_table = dict()          # Job uuid -> occupied node num
        job_to_nids_table = dict()              # Job uuid -> occupied node uuids

        for _nid in node_id_list:
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]["status"] == USED_STATUS:
                    used_job_id = crs_table[_nid][_gid]["used_job_id"]
                    job_to_nids_table[used_job_id] = [_nid] if used_job_id not in job_to_nids_table \
                                                                else unique_append(
                                                                    job_to_nids_table[used_job_id], _nid)
                    job_to_node_num_table = dict_counter_add(job_to_node_num_table, used_job_id, 1)
                    # Only clear on the node in node_id_list, even if there exists 
                    # another node contains the workers of this job.
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID

        return (job_to_node_num_table, job_to_nids_table, crs_table)

    def _clear_placement(self, job_id: str, crs_table: dict):
        """ Clear the placement of the specified job in crs_table. """
        is_modified = False
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]["used_job_id"] == job_id:
                    is_modified = True
                    # Clear placement
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
        return is_modified

    def _sort_given_nodes(self, node_id_list: Sequence[str], crs_table: dict):
        """ 
        Sort all nodes with given uuids in increasing order (for load balancing) 
        of bubble (idle GPU slot) num. 
        """
        return sorted(node_id_list, key=lambda x: self._get_bubble_num(crs_table[x]))
 
    def _is_sufficient(self, crs_table: dict, gpu_type: str, gpu_num: int):
        """ 
        Check whether the required resource quota (gpu type, gpu num) is sufficient 
        in current runtime cluster resources. 
        """
        _num = 0
        for _nid in crs_table:
            if self.resource_interface.node_pool[_nid].gpu_type == gpu_type:
                for _gid in crs_table[_nid]:
                    if crs_table[_nid][_gid]["status"] == IDLE_STATUS:
                        _num += 1
                        if _num >= gpu_num:
                            return True
        return False

    def _is_type_existed(self, crs_table: dict, gpu_type: str):
        """ 
        Check whether the required resource type is existed in 
        current runtime cluster resources. 
        """
        for _nid in crs_table:
            if self.resource_interface.node_pool[_nid].gpu_type == gpu_type:
                return True
        return False

    def _print_idle_resources(self, crs_table: dict):
        """ Print the idle GPU num of each node. """
        for _nid in crs_table:
            _num = 0
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]['status'] == IDLE_STATUS:
                    _num += 1
            print(f" - Node alias: {self.resource_interface.node_pool[_nid].alias} " + \
                  f"| Idle GPU num: {_num}")

    def _get_occ_resources(self, crs_table: dict, job_id: str, 
                           print_option: bool = True):
        """ Get the occupied resources of the target job. """
        table, occupied_resources = dict(), dict()
        for _nid in crs_table:
            node_alias = self.resource_interface.node_pool[_nid].alias
            for _gid in crs_table[_nid]:
                gpu_alias = self.resource_interface.node_pool[_nid].get_gpu_by_uuid(_gid).alias
                if crs_table[_nid][_gid]['used_job_id'] == job_id:
                    table = create_entry_and_append_queue(table, node_alias, gpu_alias)
                    occupied_resources = create_entry_and_append_queue(occupied_resources, _nid, _gid)
        if not print_option:
            return occupied_resources
        
        print(f"[I] The part of the crs_table occpupied by job " + \
              f"'{self.get_job_by_uuid(uuid=job_id).alias}':")
        print(" - ", table)

    def _parse_crs_table_change(self, prev_crs_table: dict, new_crs_table: dict, 
                                ended_job_info_table: dict = None, is_runtime: bool = False,
                                quiet_mode: bool = False):
        """ Parse the changes of the crs_table and display. """
        mdf_part = dict()
        new_resched_jid_list = list()

        # Parse
        for _nid in prev_crs_table:
            assert _nid in new_crs_table
            for _gid in prev_crs_table[_nid]:
                assert _gid in new_crs_table[_nid]
                if ((prev_crs_table[_nid][_gid]["status"] != new_crs_table[_nid][_gid]["status"]) or
                    (prev_crs_table[_nid][_gid]["used_job_id"] != new_crs_table[_nid][_gid]["used_job_id"])):
                    if _nid not in mdf_part:
                        mdf_part[_nid] = dict()
                    mdf_part[_nid][_gid] = {
                            "prev_status": prev_crs_table[_nid][_gid]["status"],
                            "prev_used_job_id": prev_crs_table[_nid][_gid]["used_job_id"],
                            "new_status": new_crs_table[_nid][_gid]["status"],
                            "new_used_job_id": new_crs_table[_nid][_gid]["used_job_id"],
                        }
                    if (new_crs_table[_nid][_gid]["used_job_id"] not in new_resched_jid_list and 
                        new_crs_table[_nid][_gid]["used_job_id"] != FAKE_JOB_ID and 
                        new_crs_table[_nid][_gid]["used_job_id"] != EMPTY_JOB_ID):
                        new_resched_jid_list.append(new_crs_table[_nid][_gid]["used_job_id"])
        
        if quiet_mode:
            # Return for evaluate resched overhead
            return new_resched_jid_list
        
        # Traverse each modified node
        node_mdf_infos = dict()         # Format: node uuid -> prev job uuid -> [prev gpu status, prev job alias, 
                                                    # new gpu status, new job alias, gpu num]
        for _nid in mdf_part:
            _dict = dict()
            for _gid in mdf_part[_nid]:
                # Prev
                prev_status = mdf_part[_nid][_gid]["prev_status"]
                prev_used_job_id = mdf_part[_nid][_gid]["prev_used_job_id"]
                if prev_used_job_id != EMPTY_JOB_ID and not self.get_job_by_uuid(prev_used_job_id):
                    # This ended job has been completely removed
                    prev_used_job_alias = ended_job_info_table[prev_used_job_id]
                else:
                    prev_used_job_alias = self.get_job_by_uuid(prev_used_job_id).alias \
                        if prev_used_job_id != EMPTY_JOB_ID else EMPTY_JOB_ALIAS
                # New
                new_status = mdf_part[_nid][_gid]["new_status"]
                new_used_job_id = mdf_part[_nid][_gid]["new_used_job_id"]
                new_used_job_alias = self.get_job_by_uuid(uuid=new_used_job_id).alias \
                    if self.get_job_by_uuid(new_used_job_id) else EMPTY_JOB_ALIAS

                # Stat how much gpus are modified
                if prev_used_job_id not in _dict:
                    # First entry of this job
                    _dict[prev_used_job_id] = [[prev_status, prev_used_job_alias, 
                                                new_status, new_used_job_alias, 1]]
                else:
                    # Update recorded modified gpu num of this job
                    is_rec = False
                    for _rec in _dict[prev_used_job_id]:
                        if (prev_status == _rec[0] and prev_used_job_alias == _rec[1] and 
                            new_status == _rec[2] and new_used_job_alias == _rec[3]):
                            _rec[4] += 1
                            is_rec = True
                            break
                    if not is_rec:
                        # This job has other entries, but not this configuration. 
                        # E.g., consecutive transformations of gpu num scaling/type changed.
                        _dict[prev_used_job_id].append([prev_status, prev_used_job_alias, 
                                                        new_status, new_used_job_alias, 1])

            node_mdf_infos[_nid] = _dict

            node = self.resource_interface.node_pool[_nid]
            for _prev_jid in _dict:
                for _rec in _dict[_prev_jid]:
                    if ((_rec[0] == IDLE_STATUS and _rec[1] == EMPTY_JOB_ALIAS) or 
                        (_rec[0] != IDLE_STATUS and _rec[1] != EMPTY_JOB_ALIAS)):
                        if is_runtime:
                            print(f"     - Node update info (alias: {node.alias} | ip_addr: {node.ip_addr})" + 
                                  f": {_rec[4]} GPUs change from '{_rec[0]}' status occpuied by '{_rec[1]}' " + 
                                  f"to '{_rec[2]}' status occpuied by '{_rec[3]}'.")
                        else:
                            print(f"     - Node update info (alias: {node.alias}): {_rec[4]} GPUs change " + 
                                  f"from '{_rec[0]}' status occpuied by '{_rec[1]}' to '{_rec[2]}' status " + 
                                  f"occpuied by '{_rec[3]}'.")
                    else:
                        # Opportunism jobs in pending jobs restart trial, in which the resource 
                        # status occupied by the opportunism jobs is recolored as IDLE_STATUS 
                        # while the used_job_id is still set as the uuid of this opportunism job.
                        if is_runtime:
                            print(f"     - Node update info (alias: {node.alias} | ip_addr: {node.ip_addr})" + 
                                  f": {_rec[4]} GPUs change from the temporary resources occpuied by '{_rec[1]}' " + 
                                  f"(opportunism job) to '{_rec[2]}' status occpuied by '{_rec[3]}'.")
                        else:
                            print(f"     - Node update info (alias: {node.alias}): {_rec[4]} GPUs change " + 
                                  f"from the temporary resources occpuied by '{_rec[1]}' (opportunism job) " + 
                                  f"to '{_rec[2]}' status occpuied by '{_rec[3]}'.")

        return mdf_part, node_mdf_infos

    def _recolor_with_occupied_resources_as_bubbles(self, crs_table: dict, job: Job):
        """ Recolor the crs_table with only self-occupied resources as bubbles. """
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if not (_nid in job.resource_alloc.node_to_gpu_table and 
                        _gid in job.resource_alloc.node_to_gpu_table[_nid]):
                    # Not occupied by the job
                    assert crs_table[_nid][_gid]["used_job_id"] != job.uuid
                    if crs_table[_nid][_gid]["status"] == IDLE_STATUS:
                        # Virtually occupied
                        crs_table[_nid][_gid]["status"] = USED_STATUS
                        # Tag with "FAKE_JOB_ID + GPU type" to gaurantee homogenous GPU type 
                        # when applying introspective bubble migration.
                        _fake_jid = FAKE_JOB_ID + "_" + self.resource_interface.node_pool[_nid].gpu_type
                        crs_table[_nid][_gid]["used_job_id"] = _fake_jid
                else:
                    # Recolor this GPU as a bubble
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
        return crs_table

    def _restore_recolored_resources(self, crs_table: dict):
        """ 
        Restore the recolored resources by recoloring resources occupied by 
        fake job ID as bubbles. 
        """
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                # All virtual bubbles should be occupied by the target job
                if crs_table[_nid][_gid]["status"] == IDLE_STATUS:
                    print(crs_table[_nid][_gid]["used_job_id"])
                
                assert crs_table[_nid][_gid]["status"] != IDLE_STATUS
                _fake_jid = FAKE_JOB_ID + "_" + self.resource_interface.node_pool[_nid].gpu_type
                if crs_table[_nid][_gid]["used_job_id"] == _fake_jid:
                    # Recolor as bubbles
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID 
        return crs_table

    def _release_resources_and_update_decision_queue(self):
        """ Release all related resources of ended jobs and update decision queue. """
        print("")
        print("[I] Clearing occupied resources of ended jobs...")
        if self.verbose:
            print("[I] The released resources are presented as follows:")
            for _job in self.ended_job_queue:
                print(f" - Job alias: {_job.alias} | " + 
                      f"Released resources: {_job.resource_alloc.node_to_gpu_table}")
        
        ended_job_id_list = list()
        crs_table = self.resource_interface.get_crs_table()
        # Debug mode
        if self.verbose:
            print(" - Idle resource status before releasing all related resources:")
            self._print_idle_resources(crs_table=crs_table)

        for _job in self.ended_job_queue:
            assert (_job.status == JOB_COMPLETED_STATUS) or (_job.status == JOB_ERROR_STATUS), \
                f"Jobs in the ended job queue should be in either completed or error status, " + \
                f"got {_job.status}."
            # Record the uuid of the ended job
            ended_job_id_list.append(_job.uuid)
            # De-register from the job registry table
            self.job_registry.pop(_job.uuid)
            # De-allocate resources
            for _nid in _job.resource_alloc.node_to_gpu_table:
                for _gid in _job.resource_alloc.node_to_gpu_table[_nid]:
                    assert crs_table[_nid][_gid]["used_job_id"] == _job.uuid
                    # Clear
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
            # Update decision queue
            if _job.is_opportunism and _job in self.decision_queue:
                # assert job in self.decision_queue
                _job_idx = self.decision_queue.index(_job)
                # First, check whether there exists another opportunism job below
                is_existed = False
                for _idx in range(_job_idx + 1, len(self.decision_queue), 1):
                    if self.decision_queue[_idx].is_opportunism:
                        is_existed = True
                        break
                # Then, update upper blockers if needed
                if not is_existed:
                    for _idx in range(_job_idx - 1, -1, -1):
                        if self.decision_queue[_idx].is_blocker:
                            # Re-tag as vanilla pending job
                            self.decision_queue[_idx].is_blocker = False
                        elif self.decision_queue[_idx].is_opportunism:
                            # Found another upper opportunism job, break
                            break
                        else:
                            raise RuntimeWarning("Error: A vanilla pending job " +
                                                 "appears above the opportunism job.")
                # Remove from the decision queue
                self.decision_queue.remove(_job)
        # Debug mode
        if self.verbose:
            print(" - Idle resource status after releasing all related resources:")
            self._print_idle_resources(crs_table=crs_table)

        # Apply deallocation
        self.resource_interface.apply_sched(crs_table=crs_table)

        return ended_job_id_list

    ######################################
    #     Migration Related Functions    #
    ######################################

    def _get_migration_num_bound(self, job_list: Sequence[Job]):
        """ 
        Get the global and local migration times bound. 
        -------------------------------------------------------------
        - Global migration num = \eta_g / \sum_i (gpu_num_i)
        - Local migration num (i) = \min{ \eta_l / gpu_num_i, \log(gpu_num_i), \log(node_cap_i) }
        """
        assert len(job_list) > 0, "Empty job list for migration."
        local_mgrt_num_table = dict()
        sum = 0
        
        for _job in job_list:
            gpu_type, gpu_num = _job.resource_quota.gpu_type, _job.resource_quota.gpu_num
            sum += gpu_num
            # local_mgrt_num_table[_job.uuid] = int(min(np.floor(LOCAL_ETA / gpu_num), np.log2(gpu_num), 
            #                                           np.log2(NODE_CAPACITY[gpu_type])))
            local_mgrt_num_table[_job.uuid] = LOCAL_ETA
    
        # return int(np.floor(GLOBAL_ETA / sum)), local_mgrt_num_table
        return GLOBAL_ETA, local_mgrt_num_table
    
    def _place_job(self, target_job: Job, fix_job_gpu_num: int = None, 
                   node_id_list: Sequence[str] = None, crs_table: dict = None):
        """ 
        Simulate to place the target job (uniformly) on the target node(s). 
        The nodes in the node_id_list should be compatible with the current target 
        GPU type of the target_job.
        """
        assert is_power_of(2, len(node_id_list)), \
            f"Allocated node num of job {target_job.alias} is {len(node_id_list)}."

        # Only uniform topology is allowed
        if not fix_job_gpu_num:
            _, job_gpu_num, _, _ = self._get_job_rt_stat(target_job.uuid, crs_table, 
                                                         overwrite_job=target_job)
        else:
            # Used for re-placing partial cross-nodes job in RHD case, since the info 
            # of the target on the node in the node_id_list will be clear before entering 
            # this function, and _get_job_rt_stat will get only partial GPU num.
            job_gpu_num = fix_job_gpu_num
        
        # Note that the occupied GPU num of fake job (bubbles) may be the number other 
        # than 1 or divided by 2.
        assert FAKE_JOB_ID in str(target_job.uuid) or job_gpu_num % len(node_id_list) == 0
        
        per_node_quota = job_gpu_num // len(node_id_list)
        assert FAKE_JOB_ID in str(target_job.uuid) or is_power_of(2, per_node_quota), \
            f"Job ({target_job.alias})'s per-node quota should be power of 2, got {per_node_quota}."
        
        # Search bubbles
        bubble_blocks = list()
        for _nid in node_id_list:
            bubble_block = list()
            for _gid in crs_table[_nid]:
                if (crs_table[_nid][_gid]["status"] == IDLE_STATUS or 
                    crs_table[_nid][_gid]["used_job_id"] == target_job.uuid):
                    # Idle GPU slot or occupied by the target job.
                    bubble_block.append(_gid)
                # NOTE: We do not differentiate GPUs under the same node
                # else:
                #     bubble_block.clear()    # Block must be consistant
                # Check whether found
                if len(bubble_block) == per_node_quota:
                    bubble_blocks.append(bubble_block)
                    break
        
        if len(bubble_blocks) < len(node_id_list):
            # Fail to get enough bubbles to place
            return False

        # Place job
        for _i, _nid in enumerate(node_id_list):
            # Clear all used info related to the target job
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]["used_job_id"] == target_job.uuid:
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
            # Place the target job into the bubble block
            assert len(bubble_blocks[_i]) == per_node_quota, \
                f"Length of bubble block _i ({len(bubble_blocks[_i])}) not equal " + \
                f"to per-node quota {per_node_quota}"
            # Color each GPU
            for _gid in bubble_blocks[_i]:
                crs_table[_nid][_gid]["status"] = USED_STATUS
                crs_table[_nid][_gid]["used_job_id"] = target_job.uuid

        # Compatible bubble blocks are found in all requested nodes.        
        return True
    
    def _rhd_bubble_migration_internal(self, node_id_1: str, bubble_num_1: int, 
                                       node_id_2: str, bubble_num_2: int, 
                                       crs_table: dict, node_cap: int):
        """ Operate RHD Bubble Migration between two nodes. """
        assert self.resource_interface.node_pool[node_id_1].gpu_type == \
            self.resource_interface.node_pool[node_id_2].gpu_type
        # Record prev crs_table to calculate node_id_list of each job 
        # (since some nodes will be recorded and clear).
        prev_crs_table = deepcopy(crs_table)
        
        if bubble_num_1 + bubble_num_2 > node_cap:
            # After Bubble Migration, one of the nodes will be emptied, 
            # the other node is partially occupied. We can move all workers 
            # of the node (with less workers/more bubbles) onto the other directly.
            
            # Debug mode
            if self.verbose:
                print("[T][RHD] RHD Type: One partial one empty.")

            # Decide src & dst node
            (src_node_id, dst_node_id) = (node_id_1, node_id_2) \
                if bubble_num_1 >= bubble_num_2 else (node_id_2, node_id_1)
            # On src side
            (job_to_node_num_table, 
             _, crs_table) = self._record_and_clear_nodes([src_node_id], crs_table)
            # On dst side
            for _jid in job_to_node_num_table:
                job = self.get_job_by_uuid(_jid)
                assert job is not None, f"Job {_jid} is not found."
                # Occupied nodes of the job. Job must have been registered 
                # in prev_crs_table.
                job_node_id_list = self._tmp_occ_nids(_jid, prev_crs_table)
                
                if len(job_node_id_list) == 1:
                    # This job is a single-node job
                    is_placed = self._place_job(job, job_to_node_num_table[_jid], 
                                                [dst_node_id], crs_table)
                    assert is_placed
                else:
                    # This job is a cross-nodes job.
                    assert src_node_id in job_node_id_list
                    if (dst_node_id in job_node_id_list and 
                        len(job_node_id_list) > 1 and FAKE_JOB_ID not in str(_jid)):
                        # Only support 2 nodes merged in the case that cross-nodes 
                        # on both src and dst (handled by self._place_job()).
                        # Otherwise, the uniform topology rule will be violated.
                        # Recover crs_table
                        crs_table = deepcopy(prev_crs_table)

                        return False, crs_table
                    # Place
                    is_placed = self._place_job(job, job_to_node_num_table[_jid], 
                                                [dst_node_id], crs_table)
                    assert is_placed
        else:
            # After Bubble Migration, one of the nodes will be fully-occupied, 
            # the other node is partially occupied. Since need to additionally 
            # consider the reversed bubbles, we can directly sort all jobs on 
            # these two nodes in the decreasing order of the worker num and 
            # sequentially place.

            # Debug mode
            if self.verbose:
                print("[T][RHD] RHD Type: One partial one full.")

            # Record job info on the two nodes and clear them 
            (job_to_node_num_table, 
             job_to_nids_table, crs_table) = self._record_and_clear_nodes([node_id_1, node_id_2], 
                                                                          crs_table)
            # Sort
            jwn_nested_list = sorted(
                [[_jid, job_to_node_num_table[_jid]] 
                 for _jid in job_to_node_num_table], key=lambda x: x[1], reverse=True
            )

            # print(self.resource_interface.node_pool[node_id_1].alias, 
            #       self.resource_interface.node_pool[node_id_2].alias,)
            # print([[self.get_job_by_uuid(_r[0]).alias, _r[0], _r[1]] for _r in jwn_nested_list])

            # For the largest job, we place directly on its previous node, setting this node as 
            # the first_node. Then, for the rest jobs, when considering job_k, if its prev node 
            # is the other node and  the rest jobs can fullfill the first_node, we place it on 
            # its prev node; otherwise, migrate to the first_node.
            
            # Get the first_node
            _nid_list = job_to_nids_table[jwn_nested_list[0][0]]
            if len(_nid_list) > 1:
                assert len(_nid_list) == 2, \
                    f"Length of node id list should be 2, got {len(_nid_list)}"
                # We select the node with less bubbles as the first_node
                first_node_id = node_id_1 if bubble_num_1 <= bubble_num_2 else node_id_2
            else:
                first_node_id = _nid_list[0]
            second_node_id = node_id_1 if first_node_id != node_id_1 else node_id_2
            
            # Re-place
            # If you statistic the migration num, note that for partial migrated cross-nodes job 
            # (in node_id_1), if the part of the job (in node_id_1) is migrated to node_id_2, it 
            # is not counted as the migration num.
            # for _rec in jwn_nested_list:
            for _idx, _rec in enumerate(jwn_nested_list):
                job = self.get_job_by_uuid(_rec[0])
                assert job is not None
                # Job must have been registered in prev_crs_table.
                job_node_id_list = self._tmp_occ_nids(_rec[0], prev_crs_table)
        
                if (len(job_node_id_list) > 2 and node_id_1 in job_node_id_list and 
                    node_id_2 in job_node_id_list and FAKE_JOB_ID not in str(_rec[0])):
                    # Only support 2 nodes merged in the case that cross-nodes on both node_1 and 
                    # node_2 (handled by self._place_job()). Otherwise, the uniform topology rule 
                    # will be violated (except for fake jobs).
                    # Recover crs_table      
                    crs_table = deepcopy(prev_crs_table)

                    return False, crs_table
                
                if _rec[1] > node_cap:
                    # Only support 2 nodes merged in the case that cross-nodes on both node_1 and 
                    # node_2 (handled by self._place_job()). Otherwise, the uniform topology rule 
                    # will be violated (except for fake jobs).
                    assert FAKE_JOB_ID in str(_rec[0]), \
                        f"Occupied GPU num of job {job.alias} is larger than node " + \
                        f"capacity {node_cap}, this situation can only happened on " + \
                        f"fake jobs (from introspective migration), since GPU num and " + \
                        f"node capacity can only be the power of 2."
                    # Recover crs_table      
                    crs_table = deepcopy(prev_crs_table)

                    return False, crs_table

                target_node_id = None
                if job_to_nids_table[_rec[0]] == first_node_id:
                    # Directly place since the previous node of this job is first_node
                    target_node_id = first_node_id
                else:
                    # Check whether the rest nodes can fullfill the first_node
                    _bubble_num = self._get_bubble_num(crs_table[first_node_id])
                    _rest_gpu_num = 0
                    for i in range(_idx + 1, len(jwn_nested_list), 1):
                        _rest_gpu_num += jwn_nested_list[i][1]
                    # Set the target node based on whether the first_node can be 
                    # fullfilled in the future
                    target_node_id = second_node_id \
                        if _rest_gpu_num >= _bubble_num else first_node_id
                    other_node_id = first_node_id \
                        if target_node_id != first_node_id else second_node_id

                    # Place the job
                    is_placed = self._place_job(job, job_to_node_num_table[_rec[0]], 
                                                [target_node_id], crs_table)
                    if not is_placed:
                        # The first node is full, place on the second node.
                        is_placed = self._place_job(job, job_to_node_num_table[_rec[0]], 
                                                    [other_node_id], crs_table)
                    assert is_placed
        # RHD Bubble Migration is completed.   
        return True, crs_table
    
    def _rhd_bubble_migration(self, node_bubble_num_list: Sequence[list], 
                              crs_table: dict, node_cap: int):
        """ 
        Operate RHD Bubble Migration among partially-occupied nodes with 
        the same GPU type. 
        """
        for _idx in range(len(node_bubble_num_list) // 2):
            ((_id_1, _num_1), (_id_2, _num_2)) = (node_bubble_num_list[_idx], 
                                                  node_bubble_num_list[-_idx-1])
            (is_migrated, 
             crs_table) = self._rhd_bubble_migration_internal(_id_1, _num_1, _id_2, _num_2, 
                                                              crs_table, node_cap)
            # Debug mode
            if self.verbose:
                _node_1_alias = self.resource_interface.node_pool[_id_1].alias
                _node_2_alias = self.resource_interface.node_pool[_id_2].alias
                print(f"[T][RHD] RHD Status: Node 1: {_node_1_alias} | Bubble num 1: {_num_1} " + 
                      f"| Node 2: {_node_2_alias} | Bubble num 2: {_num_2} | " + 
                      f"Is migrated: {is_migrated}")
        
        return crs_table
    
    def _place_job_with_relaxted_locality(self, target_job: Job, crs_table: dict):
        """ 
        Try to place the target job on the migrated crs_table with feasible 
        relaxed locality. 
        """
        # if self.is_runtime:
        #     # Currently, cross-nodes placement is not supported in runtime scheduling scenarios
        #     return False, None, INFEASIBLE_THR
        
        job_gpu_type, job_gpu_num, _, _  = self._get_job_rt_stat(target_job.uuid, crs_table, 
                                                                 overwrite_job=target_job)

        # if job_gpu_type in forbid_cross_nodes_gpu_type:
        #     # NOTE: In runtime, we forbid the cross-nodes placement of certain 
        #     #       GPU type due to network legacy, which would cause Runtime error
        #     return False, None, INFEASIBLE_THR
        
        # Node - bubble status nested list that also contains idle nodes
        node_bubble_num_list, idle_nids = self._sort_partial_occ_nodes(crs_table, job_gpu_type, 
                                                                       job_gpu_num)
        wide_node_bubble_num_list = [[_nid, NODE_CAPACITY[job_gpu_type]] for _nid in idle_nids] \
                                        + node_bubble_num_list
        # Per-node GPU quota 
        per_node_quota = job_gpu_num
        while per_node_quota > 1:
            assert per_node_quota % 2 == 0, \
                f"Job ({target_job.alias})'s per-node quota {per_node_quota} is " + \
                f"not divisible by 2. GPU num: {job_gpu_num}."
            # Relaxed locality
            per_node_quota = per_node_quota // 2
            if per_node_quota > NODE_CAPACITY[job_gpu_type]:
                # Excceed the node capacity
                continue
            locality = [per_node_quota for _ in range(job_gpu_num // per_node_quota)]

            if not self._is_alloc_feasible(target_job, job_gpu_type, locality):
                # Current relaxed locality is not feasible.
                return (False, None, INFEASIBLE_THR)

            node_id_list = list()
            for _rec in wide_node_bubble_num_list:
                if _rec[1] >= per_node_quota:
                    node_id_list.append(_rec[0])
                    if len(node_id_list) == len(locality):
                        # Placed
                        is_placed = self._place_job(target_job, None, node_id_list, crs_table)
                        assert is_placed
                        # Get thr variation
                        _gain = self._get_delta_thr(target_job, job_gpu_type, job_gpu_num, locality)
                        return (True, locality, _gain)
        
        # Current resources is insufficient
        return (False, None, INFEASIBLE_THR)

    def _migration_search_internal(self, target_job: Job, crs_table: dict, 
                                   global_mgrt_num: int, local_mgrt_num: int):
        """ 
        The internal function of migration search.
        Notes:
            - To simulate, we only modify the content of `crs_table`.
            - Only the structured marginal gain of the target job should be considered, 
              since the Bubble Migration will not affect the locality of other running jobs.
        -------------------------------------------------------------
        Return:
            - is_feasible: Whether can find a feasible solution.
            - is_migrated: Whether there exists a running job that is migrated by 
                           RHD Bubble Migration.
            - mgrt_gain: Current migration gain in this round, which is the throughput 
                         obtained by placing the target job.
            - is_relaxed: Whether the locality of the target job is relaxed.
            - global_mgrt_num: The remained global migration num.
            - # inst_changed: Discussed in _migration_search().
        """
        job_gpu_type, job_gpu_num, _, _ = self._get_job_rt_stat(target_job.uuid, crs_table,
                                                                overwrite_job=target_job)
        node_cap = NODE_CAPACITY[job_gpu_type]
        assert job_gpu_num % node_cap == 0 or node_cap % job_gpu_num == 0 
        
        # The structured marginal gain (throughput) obtained by placing the target job 
        # with the best locality
        best_locality = self._get_best_locality(job_gpu_num, job_gpu_type, None, 
                                                local_mgrt_num)
        
        # A nested list of node - bubble status
        node_bubble_num_list, idle_nids = self._sort_partial_occ_nodes(crs_table, job_gpu_type, 
                                                                       job_gpu_num)
        
        # Check whether can run with best locality
        if not self._is_alloc_feasible(target_job, job_gpu_type, best_locality):
            # The target job cannot run even with the best locality, directly return 
            return (False, False, INFEASIBLE_THR, False, global_mgrt_num, crs_table)
        # The gain (throughput) by placing the target job with the best locality
        gain_with_best_locality = self._get_delta_thr(target_job, job_gpu_type, job_gpu_num, 
                                                      best_locality)
        
        # Step 1. Check whether the target job can be directly started with best locality
        # - First, check whether can start on idle node(s) (in a scattered manner)
        needed_idle_node_num = 1 if job_gpu_num <= node_cap else job_gpu_num // node_cap
        if needed_idle_node_num <= len(idle_nids):
            # Can be placed
            node_id_list = idle_nids[:needed_idle_node_num]
            is_placed = self._place_job(target_job, None, node_id_list, crs_table)
            assert is_placed
            # Debug mode
            if self.verbose:
                _idles_nodes = [
                    self.resource_interface.node_pool[node_id].alias for node_id in node_id_list
                ]
                print(f"[T][MGRT] Directly start job '{target_job.alias}' (alias) on " + 
                      f"idle node(s): {_idles_nodes}")
            
            return (True, False, gain_with_best_locality, False, global_mgrt_num, crs_table)
        
        # - Then, check whether can start on partially-occupied node.
        for _rec in node_bubble_num_list:
            if job_gpu_num <= _rec[1]:
                # Placed, only modify the crs_table as simulation.
                is_placed = self._place_job(target_job, None, [_rec[0]], crs_table)
                assert is_placed
                # Debug mode
                if self.verbose:
                    _node_alias = self.resource_interface.node_pool[_rec[0]].alias
                    print(f"[T][MGRT] Directly start job '{target_job.alias}' (alias) " + 
                          f"on partially occupied node '{_node_alias}' (alias).")

                return (True, False, gain_with_best_locality, False, global_mgrt_num, crs_table)
        
        # Step 2. Recursive Halving and Doubling (RHD) Migration within the limitation of 
        #         migration num bound.
        mgrt_cnt = 0
        mgrt_upper_bnd = min(global_mgrt_num, local_mgrt_num)
        while mgrt_cnt < mgrt_upper_bnd:
            # RHD Bubble Migration
            crs_table = self._rhd_bubble_migration(node_bubble_num_list, crs_table, node_cap)

            mgrt_cnt += 1
            # Re-generate node_bubble_num_list
            node_bubble_num_list, idle_nids = self._sort_partial_occ_nodes(crs_table, job_gpu_type, 
                                                                           job_gpu_num)
            # Debug mode
            if self.verbose:
                _list = [
                    [self.resource_interface.node_pool[_rec[0]].alias, _rec[1]] 
                        for _rec in node_bubble_num_list
                ]
                _list.extend(
                    [[self.resource_interface.node_pool[_nid].alias, node_cap] for _nid in idle_nids]
                )
                
                print(f"[T][MGRT] Iter {mgrt_cnt}: Node Bubble Num (NBS) Status (in alias format, " + 
                      f"including idle nodes): {_list}")
            
            # Re-check whether best topology exists
            # - First, Whether can start on idle node(s)
            needed_idle_node_num = 1 if job_gpu_num <= node_cap else int(job_gpu_num / node_cap)
            if needed_idle_node_num <= len(idle_nids):
                # Can be placed
                node_id_list = idle_nids[:needed_idle_node_num]
                is_placed = self._place_job(target_job, None, node_id_list, crs_table)
                assert is_placed
                # Debug mode
                if self.verbose:
                    _idle_nodes = [
                        self.resource_interface.node_pool[_nid].alias for _nid in node_id_list
                    ]
                    print(f"[T][MGRT] After {mgrt_cnt} round(s) of migration, start job " + 
                          f"'{target_job.alias}' (alias) on idle node(s): {_idle_nodes}")

                return (True, True, gain_with_best_locality, False, (global_mgrt_num - mgrt_cnt), crs_table)
            
            # - Then, check whether can start on partially-occupied node.
            for _rec in node_bubble_num_list:
                if job_gpu_num <= _rec[1]:
                    # Placed, only modify the crs_table as simulation.
                    is_placed = self._place_job(target_job, None, [_rec[0]], crs_table)
                    assert is_placed
                    # Debug mode
                    if self.verbose:
                        _node_alias = self.resource_interface.node_pool[_rec[0]].alias
                        print(f"[T][MGRT] After {mgrt_cnt} round(s) of migration, start job " + 
                              f"'{target_job.alias}' (alias) on partially occupied node " + 
                              f"'{_node_alias}' (alias).")

                    return (True, True, gain_with_best_locality, False, (global_mgrt_num - mgrt_cnt), crs_table)
        
        # Step 3. After the migration num is exhausted, try find a relaxed topology that is 
        #         feasible on the migrated crs_table.
        (is_succeed, _locality, 
         gain_with_relaxed_locality) = self._place_job_with_relaxted_locality(target_job, crs_table)
        # Debug mode
        if self.verbose:
            print(f"[T][MGRT] Try placing job '{target_job.alias}' (alias) with relaxed locality: " + 
                  f"Is Succeed: {is_succeed} | Locality: {_locality}")
        
        return (True, (mgrt_cnt > 0), gain_with_relaxed_locality, True, (global_mgrt_num - mgrt_upper_bnd), crs_table) \
                if is_succeed else (False, (mgrt_cnt > 0), INFEASIBLE_THR, False, global_mgrt_num, crs_table)

    def _migration_search(self, target_job: Job, crs_table: dict, 
                          global_mgrt_num: int, local_mgrt_num: int, 
                          is_clear_target_job_placement: bool = False):
        """
        Search for intra-node and inter-node migration plan with maximal structured 
        marginal gain (maybe < 0) and upper bound of migration num.
        -------------------------------------------------------------
        Return:
            - is_feasible: Whether can find a feasible solution.
            - is_migrated: Whether there exists a running job that is migrated by RHD 
                           Bubble Migration.
            - _mgrt_gain: Current migration gain in this round, which is the throughput 
                          gain obtained by re-placing the target job (prev placement 
                          -> new placement).
            - is_relaxed: Whether the locality of the target job is relaxed.
            - global_mgrt_num: The remained global migration num.
            # - inst_changed: Whether the job instance stores latest modification of 
            #                 gpu type/gpu num (e.g., in htc we directly modify gpu 
            #                 type in job instance). if enabled, overwrite job latest 
            #                 gpu num and gpu type rather than querying them from 
            #                 crs_table.
        """
        # Previous thr variation
        _tmp_job_gpu_type = self._tmp_job_gpu_type(target_job.uuid, crs_table)
        if _tmp_job_gpu_type is None:
            # New job
            _prev_gain = 0.0
        else:
            _tmp_occ_gpu_num = self._tmp_occ_gpu_num(target_job.uuid, crs_table)
            _tmp_job_node_num = len(self._tmp_occ_nids(target_job.uuid, crs_table))
            _tmp_job_locality = [
                _tmp_occ_gpu_num // _tmp_job_node_num for _ in range(_tmp_job_node_num)
            ]
            _prev_gain = self._get_delta_thr(target_job, _tmp_job_gpu_type, _tmp_occ_gpu_num, 
                                             _tmp_job_locality)
        if is_clear_target_job_placement:
            # Clear the previous placement of the target job
            is_modified = self._clear_placement(target_job.uuid, crs_table)
            # Debug mode
            if is_modified and self.verbose:
                print(f"[I][MGRT] The previous placement of the target job " + 
                      f"'{target_job.alias}' (alias) has been cleared.")

        (is_fsb, is_migrated, _cur_gain, 
         is_relaxed, global_mgrt_num, 
         crs_table) = self._migration_search_internal(target_job, crs_table, 
                                                      global_mgrt_num, local_mgrt_num)
        # Debug mode
        if self.verbose:
            print(f"[I][MGRT] Migration search status -> Is feasible: {is_fsb} | " + 
                  f"Is migrated: {is_migrated} | Is relaxed (only partially occupied " + 
                  f"case): {is_relaxed} | Migration gain (thr of the new job): " + 
                  f"{_cur_gain - _prev_gain} | Remained global migration num (if " + 
                  f"applied): {global_mgrt_num}")
            
        return (is_fsb, is_migrated, (_cur_gain - _prev_gain), 
                is_relaxed, global_mgrt_num, crs_table)

    def _introspective_migration_search(self):
        """ 
        Called in the process of running jobs optimization, search for 
        inter-node migration plan with maximal structured marginal gain 
        on cross-nodes jobs to improve their locality. 
        This function is temporarily DEPRECATED.
        """
        if self.verbose:
            print("")
        print("[I] Begin Introspective Bubble Migration search among all cross-nodes jobs...")
        
        # Step 1. Statistic cross-nodes jobs
        cross_nodes_job_queue = list()
        for _job in self.running_job_queue:
            if _job.is_cross_nodes and (_job.resource_quota.locality[0] < 
                                       NODE_CAPACITY[_job.resource_quota.gpu_type]):
                # Partially cross-nodes job
                cross_nodes_job_queue.append(_job)

        if self.verbose:
            print("")
        print("[I][IMGRT] Cross-nodes job queue:")
        for _job in cross_nodes_job_queue:
            print(f" - Job alias: {_job.alias} | GPU type: {_job.resource_quota.gpu_type} | " + 
                  f"GPU num: {_job.resource_quota.gpu_num} | Locality: {_job.resource_quota.locality}")
        if len(cross_nodes_job_queue) == 0:
            print("[I][IMGRT] No cross-nodes job exists.")
            return 

        # Step 2. Calculate global and local migration num.
        global_mgrt_num, local_mgrt_num_table = self._get_migration_num_bound(job_list=cross_nodes_job_queue)
        
        # Debug mode
        if self.verbose:
            print("")
            print(f"[I][IMGRT] Global migration num: {global_mgrt_num}")
            print("[I][IMGRT] Local migration num dict:")
            for _jid in local_mgrt_num_table:
                print(f" - Job alias {self.get_job_by_uuid(_jid).alias} | " + 
                      f"Migration num: {local_mgrt_num_table[_jid]}")
            print("")
        
        # Introspective info
        introsp_info_queue = list()
        is_introsp_plan_found = True
        # Loop until no feasible plan is found
        while is_introsp_plan_found:
            is_introsp_plan_found = False
            # Search for the best plan for the maximal thr
            _max_gain = INFEASIBLE_THR
            introspective_job_id = None
            decision_crs_table = None
            remained_global_mgrt_num = None
            
            for _job in cross_nodes_job_queue:
                # Step 3. Recolor the crs_table with only self-occupied resources as bubbles
                assert _job is not None
                crs_table = self.resource_interface.get_crs_table()
                (prev_gpu_type, prev_gpu_num, 
                 prev_locality, _) = self._get_job_rt_stat(_job.uuid, crs_table, 
                                                           overwrite_job=_job)
                if eq(prev_locality, 
                      self._get_best_locality(prev_gpu_num, prev_gpu_type, None, 
                                              local_mgrt_num_table[_job.uuid])):
                    # This cross-nodes job has been modified to the best locality
                    # when updating the previous cross-nodes job.
                    continue
                # Recolor
                crs_table = self._recolor_with_occupied_resources_as_bubbles(crs_table, _job)

                # Step 4. Perform RHD Bubble Migration Search
                # Since the previously occupied GPUs has been recolored as bubbles, 
                # we don't need to clear its placement
                # NOTE: In this search, we modify crs_table and _global_mgrt_num
                (is_feasible, _, _, is_relaxed, 
                 _global_mgrt_num, crs_table) = self._migration_search(_job, crs_table, 
                                                                       global_mgrt_num, 
                                                                       local_mgrt_num_table[_job.uuid],
                                                                       is_clear_target_job_placement=False)
                # A feasible plan can be found since the original relaxed 
                # locality is already feasible.
                assert is_feasible

                # Step 5. Restore recolored resources and update decision
                crs_table = self._restore_recolored_resources(crs_table)
                (new_gpu_type, new_gpu_num, 
                 new_locality, _) = self._get_job_rt_stat(_job.uuid, crs_table, 
                                                          overwrite_job=_job)
                # Throughput gain
                _gain = self._get_delta_thr(_job, new_gpu_type, new_gpu_num, new_locality, 
                                            prev_gpu_type, prev_gpu_num, prev_locality)
                # Update
                if _gain > 0 and _gain > _max_gain:
                    # Avoid updating with the same configuration as the prev 
                    # by requiring that _gain > 0
                    _max_gain = _gain
                    introspective_job_id = _job.uuid
                    decision_crs_table = crs_table
                    remained_global_mgrt_num = _global_mgrt_num

            # Apply introspective decision
            if introspective_job_id:
                is_introsp_plan_found = True
                # Get prev job info
                _introspective_job = self.get_job_by_uuid(introspective_job_id)
                _crs_table = self.resource_interface.get_crs_table()
                (prev_gpu_type, prev_gpu_num, 
                 prev_locality, _) = self._get_job_rt_stat(introspective_job_id, _crs_table)      
                # Update real cluster resources
                self.resource_interface.apply_sched(decision_crs_table)
                # Update the resource status of running jobs
                self._update_run_jobs(decision_crs_table)
                # Remove from cross-nodes job queue
                cross_nodes_job_queue.remove(_introspective_job)
                # Record
                _introspective_job = self.get_job_by_uuid(introspective_job_id)
                (new_gpu_type, new_gpu_num, 
                 new_locality, _) = self._get_job_rt_stat(introspective_job_id, decision_crs_table)
                introsp_info_queue.append(
                    [introspective_job_id, prev_gpu_type, prev_gpu_num, 
                     prev_locality, new_gpu_type, new_gpu_num, new_locality]
                )
                # Update global migration num
                global_mgrt_num = remained_global_mgrt_num
        
        if len(introsp_info_queue) == 0:
            print("")
            print("[I][ITSPT] No introspective migrating plan is found among cross-nodes jobs.")
        else:
            print("")
            print(f"[I][ITSPT] The introspective migrating plans are sequentially presented " + 
                  f"as follows (also the applying order):")
            for _idx, _info in enumerate(introsp_info_queue):
                _job_alias = self.get_job_by_uuid(_info[0]).alias
                print(f" - Plan {_idx + 1}: Job alias: {_job_alias} | Prev GPU type: {_info[1]} | " + 
                      f"Prev GPU num: {_info[2]} | Prev locality: {_info[3]} | " + 
                      f"New GPU type: {_info[4]} | New GPU num: {_info[5]} | New locality: {_info[6]}")

    ######################################
    #     Downgrade Related Functions    #
    ######################################
    
    def _update_crs_table_after_job_shrink(self, crs_table: dict, prev_jni_list: Sequence[str], 
                                           prev_locality: Sequence[int], shrinked_job_id: str, 
                                           new_gpu_num: int):
        """ 
        Update crs_table after a job is shrinked. 
        Args:
            - prev_jni_list: Previous node uuid list of the shrinked job.
        """
        if len(prev_locality) == 1:
            # Just shrink the num
            _cnt = 0
            assert len(prev_jni_list) == 1
            for _gid in crs_table[prev_jni_list[0]]:
                if crs_table[prev_jni_list[0]][_gid]["used_job_id"] == shrinked_job_id:
                    _cnt += 1
                    if _cnt > new_gpu_num:
                        # Clear
                        crs_table[prev_jni_list[0]][_gid]["status"] = IDLE_STATUS
                        crs_table[prev_jni_list[0]][_gid]["used_job_id"] = EMPTY_JOB_ID
            assert _cnt == new_gpu_num * 2, \
                f"Shrinked GPU num ({new_gpu_num}) should be half of prev GPU num ({_cnt})"
        else:
            # Just halve the locality (e.g., [4, 4] -> [4]).
            prev_jni_list = self._sort_given_nodes(prev_jni_list, crs_table)
            # Half 
            _len = len(prev_jni_list)
            assert _len % 2 == 0
            prev_jni_list = prev_jni_list[: _len // 2]
            # Traverse prev nodes
            for _nid in prev_jni_list:
                for _gid in crs_table[_nid]:
                    if crs_table[_nid][_gid]["used_job_id"] == shrinked_job_id:
                        # Clear
                        crs_table[_nid][_gid]["status"] = IDLE_STATUS
                        crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
    
    def _shrink_one_job_and_migration_search(self, _target_job: Job, cur_gpu_type: str, 
                                             cand_job_id_list: Sequence[str], crs_table: dict, gain: float, 
                                             global_mgrt_num: int, local_mgrt_num: int):
        """
        Choose one job with specified GPU type among candidate job id list based on maximizing 
        structured marginal gain to shrink at a time, then call migration search for the target job.
        Note:
            - If the cur_gpu_type is the same as the required GPU type of the target job, then we 
              perform migration search of the target job.
            - If the cur_gpu_type is different from the reqiuired GPU type of the target job, which 
              is to be prepared for the htc search in the next iteration, we do not perform migration 
              search of the target job in this iteration.
        -------------------------------------------------------------
        Return:
            - is_feasible: Whether can find a feasible solution.
            - _crs_table: Tmp crs_table that contains the result of the migration search after one 
                          job shrink.
            - is_shrinked: Whether sucessfully find a shrinkable job and shrink.
            - shrinked_job_id: The job ID of the shrinked job.
            - shrink_info: The shrink info (prev_gpu_num, prev_locality, new_gpu_num, new_locality) of 
                           this operation.
            - gain: Current total shrink and htc gain (for prev k operations)
            - mgrt_gain: Current migration gain in this round.
            - is_relaxed: Whether the locality of the target job is relaxed.
            - global_mgrt_num: The remained global migration num.
        """
        is_shrinked = False
        # Note that we allow the infeasible shrink operation, but it is still unacceptable. 
        # e.g., 2-GPU 1080ti -> 1-GPU 1080ti -> (future) 1-GPU a100
        max_gain = INFEASIBLE_THR - 1
        shrinked_job_id = EMPTY_JOB_ID
        new_gpu_num = None
        prev_locality = None
        expected_new_locality = None
        prev_jni_list = None  
        # GPU type of the target job
        target_gpu_type, _, _, _ = self._get_job_rt_stat(_target_job.uuid, crs_table, 
                                                         overwrite_job=_target_job) 

        # Get the max allocated gpu num of all candiate jobs
        max_alloc_gpu_num = -1
        for _jid in cand_job_id_list:
            # Prev GPU num & locality
            _overwrite_job = _target_job if _jid == _target_job.uuid else None
            _gpu_type, _gpu_num, _, _ = self._get_job_rt_stat(_jid, crs_table, _overwrite_job) 
            if _gpu_type == cur_gpu_type:
                max_alloc_gpu_num = max(_gpu_num, max_alloc_gpu_num)
        
        # is_found = False
        # cur_gpu_num = max_alloc_gpu_num
        # while cur_gpu_num > 1 and not is_found:
        for _idx, _jid in enumerate(cand_job_id_list):
            # Prev GPU num & locality
            _overwrite_job = _target_job if _jid == _target_job.uuid else None
            _gpu_type, _prev_gpu_num, _prev_locality, job_node_id_list = self._get_job_rt_stat(_jid, crs_table, 
                                                                                            _overwrite_job)   
            
            if _prev_gpu_num < max_alloc_gpu_num // 2:
            # if _prev_gpu_num < cur_gpu_num:
                # We prefer to shrink jobs with large sallocated gpu num first.
                # Then, we select the one with the best throughput variation. 
                # The inherent reason is that in some cases, shrinking the job with
                # the best throughput variaion (e.g., wres with low thr) can release
                # less idle gpus, which is bad for allocating other new jobs.
                continue
            
            if _jid != _target_job.uuid and len(job_node_id_list) == 0:
                # The placement of the job has been cleared in this crs_table (e.g., opportunism 
                # job in pending jobs restart case). Regain the crs_table of the real cluster 
                # resources, locally used to get the placement of the job.
                _crs_table = self.resource_interface.get_crs_table()
                _gpu_type, _prev_gpu_num, _prev_locality, job_node_id_list = self._get_job_rt_stat(_jid, _crs_table)   
                assert len(job_node_id_list) > 0
            
            if _gpu_type != cur_gpu_type:
                # In this function, we only consider the jobs with the same GPU type as the target job.
                continue
            
            if _prev_gpu_num == 1:
                # Cannot be further shrinked
                continue
            
            # New GPU num & locality
            _new_gpu_num = _prev_gpu_num // 2
            # If target job is cross-nodes, half the locality (e.g., [4, 4] -> [4]).
            _expected_new_locality = [
                _prev_locality[0] for _ in range(len(_prev_locality) // 2)
            ] if len(_prev_locality) > 2 else [_new_gpu_num]
            # Get thr variation
            _job = self.get_job_by_uuid(_jid)
            _gain = self._get_delta_thr(_job, _gpu_type, _new_gpu_num, _expected_new_locality, 
                                        _gpu_type, _prev_gpu_num, _prev_locality)

            if _gain > max_gain:
                # Record best
                is_found = True
                max_gain = _gain
                shrinked_job_id = _jid
                prev_jni_list = job_node_id_list
                (prev_locality, expected_new_locality, 
                new_gpu_num) = (_prev_locality, _expected_new_locality, _new_gpu_num)
                # else:
                #     # No shrinkable job with current gpu num
                #     cur_gpu_num = cur_gpu_num // 2
        
        if shrinked_job_id != EMPTY_JOB_ID:
            is_shrinked = True
            # Update total gain
            gain += max_gain
            # Decide how to shrink
            if shrinked_job_id != _target_job.uuid:
                # Other running job is shrinked, need to update crs_table
                self._update_crs_table_after_job_shrink(crs_table, prev_jni_list, prev_locality, 
                                                        shrinked_job_id, new_gpu_num)
            else:
                # The target job is shrinked
                _target_job.resource_quota.gpu_num = new_gpu_num
                _target_job.resource_quota.locality = expected_new_locality
                if len(prev_jni_list) > 0:
                    # Although is target job but has registered in crs_table 
                    # (maybe in job grow search).
                    self._update_crs_table_after_job_shrink(crs_table, prev_jni_list, prev_locality, 
                                                            shrinked_job_id, new_gpu_num)
        else:
            # Can not find a job to shrink
            return (False, crs_table, is_shrinked, EMPTY_JOB_ID, None, gain, 
                    INFEASIBLE_THR, False, global_mgrt_num)
        
        if cur_gpu_type == target_gpu_type:
            # Apply migration search for the target job.
            # In this search, we modify _crs_table and global_mgrt_num.
            _crs_table = deepcopy(crs_table)
            (is_feasible, is_migrated, mgrt_gain, 
             is_relaxed, global_mgrt_num, _crs_table) = self._migration_search(_target_job, _crs_table, 
                                                                   global_mgrt_num, local_mgrt_num, 
                                                                   is_clear_target_job_placement=True)
            # Real locality after migration search (maybe further relaxed than the expected new locality)
            _overwrite_job = _target_job if shrinked_job_id == _target_job.uuid else None
            _, _job_gpu_num, _job_locality, _ = self._get_job_rt_stat(shrinked_job_id, _crs_table, 
                                                                      _overwrite_job)
            assert _job_gpu_num == new_gpu_num, f"Mismatched job GPU num of job " + \
                f"{self.get_job_by_uuid(shrinked_job_id).alias}: {_job_gpu_num}, {new_gpu_num}"
            # Shrink info
            shrink_info = (shrinked_job_id, cur_gpu_type, new_gpu_num * 2, 
                           prev_locality, new_gpu_num, _job_locality) if is_shrinked else None
            # Return
            return (is_feasible, _crs_table, is_shrinked, shrinked_job_id, shrink_info, gain, 
                    mgrt_gain, is_relaxed, global_mgrt_num)
        else:
            # Current GPU type is not compatible with the required GPU type of the target job, 
            # just shrink for future htc operation of other jobs.
            # Shrink info without further modifying by a migration search.
            shrink_info = (shrinked_job_id, cur_gpu_type, new_gpu_num * 2, 
                           prev_locality, new_gpu_num, expected_new_locality) if is_shrinked else None
            # Return
            return (False, crs_table, is_shrinked, shrinked_job_id, shrink_info, gain, 
                    INFEASIBLE_THR, False, global_mgrt_num)
        
    def _update_crs_table_after_htc(self, crs_table: dict, prev_jni_list: Sequence[str], 
                                    idle_resources: dict, htc_job_id: str):
        """ Update crs_table after the GPU type of a job is changed. """
        # First, clear prev info
        for _nid in prev_jni_list:
            for _gid in crs_table[_nid]:
                if crs_table[_nid][_gid]["used_job_id"] == htc_job_id:
                    # Clear
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
        # Then, update new info
        for _nid in idle_resources:
            for _gid in idle_resources[_nid]:
                assert crs_table[_nid][_gid]["status"] == IDLE_STATUS
                # Color
                crs_table[_nid][_gid]["status"] = USED_STATUS
                crs_table[_nid][_gid]["used_job_id"] = htc_job_id
    
    def _htc_one_job_and_migration_search(self, _target_job: Job, cand_job_id_list: Sequence[str], 
                                          crs_table: dict, gain: float, global_mgrt_num: int, 
                                          local_mgrt_num: int):
        """
        Choose one job among candidate job id list based on maximizing structured marginal gain 
        to downgrade GPU type at a time, then call migration search for the target job.
        -------------------------------------------------------------
        Return:
            - is_feasible: Whether can find a feasible solution.
            - _crs_table: Tmp crs_table that contains the result of the migration search after 
                          the GPU type of a job is changed.
            - is_htc: Whether sucessfully find a job to change its GPU type.
            - htc_job_id: The job ID of the htc job.
            - htc_info: The htc info (prev_gpu_type, new_gpu_type) of this operation.
            - gain: Current total shrink and htc gain (for prev k operations)
            - mgrt_gain: Current migration gain in this round.
            - is_relaxed: Whether the locality of the target job is relaxed.
            - global_mgrt_num: The remained global migration num.
        """
        is_htc = False
        max_gain = INFEASIBLE_THR
        max_gain_with_mgrt = INFEASIBLE_THR
        htc_job_id = EMPTY_JOB_ID
        new_gpu_type = None
        new_locality = None
        new_global_mgrt_num = None
        prev_gpu_type = None
        prev_gpu_num = None
        prev_locality = None
        prev_jni_list = None        # Prev job node id list
        # GPU type of the target job
        target_gpu_type, _, _, _ = self._get_job_rt_stat(_target_job.uuid, crs_table, 
                                                         overwrite_job=_target_job)

        for _jid in cand_job_id_list:
            # _job = self.get_job_by_uuid(_jid).deepcopy()
            # _job = deepcopy(self.get_job_by_uuid(_jid))
            # _job = pickle.loads(pickle.dumps(self.get_job_by_uuid(_jid)))
            _job = deepcopy_pickle(self.get_job_by_uuid(_jid))
            (_prev_gpu_type, _gpu_num, 
             _locality, job_node_id_list) = self._get_job_rt_stat(_jid, crs_table)

            if _prev_gpu_type != target_gpu_type:
                # In this function, we only consider the jobs with the same GPU 
                # type as the target job. 
                continue
            
            # Traverse GPU type
            for _new_gpu_type in self.supported_gpu_types:
                if _new_gpu_type == target_gpu_type:
                    # Skip if the new GPU type is the same as the prev GPU type
                    continue
                
                if not self._is_sufficient(crs_table, _new_gpu_type, _gpu_num):
                    # Debug mode
                    if self.verbose:
                        print(f"[T][HTC] Job alias '{_job.alias}' requires {_gpu_num} " + 
                              f"GPUs and is insufficient with GPU type '{_new_gpu_type}'.")
                    # Skip if the remained resources is not sufficient
                    continue
                
                # Debug mode
                if self.verbose:
                    print(f"[T][HTC] Search for a htc with job alias '{_job.alias}' " + 
                          f"and GPU type '{_new_gpu_type}'")

                # New locality (default to be the best locality)
                _expected_new_locality = self._get_best_locality(_gpu_num, _new_gpu_type, 
                                                                 _locality, local_mgrt_num)
                # Get structured marginal gain within best new locality
                _gain = self._get_delta_thr(_job, _new_gpu_type, _gpu_num, _expected_new_locality, 
                                            _prev_gpu_type, _gpu_num, _locality)
                # Migration trial of the tmp htc job
                # In this search, we modify _crs_table _job and _global_mgrt_num.
                _crs_table = deepcopy(crs_table)
                # Update GPU type and GPU num (in case that the GPU num of this job is 
                # modified in previous iteration)
                _job.resource_quota.gpu_type = _new_gpu_type
                _job.resource_quota.gpu_num = _gpu_num
                # Try to migrate this tmp htc job as a new job.
                (_is_feasible, is_migrated, _mgrt_gain, 
                 is_relaxed, _global_mgrt_num, _crs_table) = self._migration_search(_job, _crs_table, 
                                                                        global_mgrt_num, local_mgrt_num, 
                                                                        is_clear_target_job_placement=True)
                _gain_with_mgrt = _gain + _mgrt_gain
                
                # Update best choice
                if ((_gain > max_gain and _jid == _target_job.uuid) or 
                    (_gain_with_mgrt > max_gain_with_mgrt and 
                     _jid != _target_job.uuid)) and _is_feasible:
                    if _jid == _target_job.uuid:
                        max_gain = _gain
                    else:
                        max_gain_with_mgrt = _gain_with_mgrt
                    htc_job_id = _jid
                    prev_gpu_type, prev_gpu_num, prev_locality = _prev_gpu_type, _gpu_num, _locality
                    new_gpu_type, new_locality = _new_gpu_type, _expected_new_locality
                    prev_jni_list = job_node_id_list
                    new_global_mgrt_num = _global_mgrt_num
        
        if htc_job_id != EMPTY_JOB_ID:
            is_htc = True
            # Decide how to htc
            if htc_job_id != _target_job.uuid:
                # Other running job is htc
                # _htc_job = self.get_job_by_uuid(htc_job_id).deepcopy()
                # _htc_job = copy.deepcopy(self.get_job_by_uuid(htc_job_id))
                # _htc_job = pickle.loads(pickle.dumps(self.get_job_by_uuid(htc_job_id)))
                _htc_job = deepcopy_pickle(self.get_job_by_uuid(htc_job_id))
                # Operate migration search of the selected htc job, only to update crs_table
                # In this search, we modify crs_table and _htc_job.
                # Update GPU type
                _htc_job.resource_quota.gpu_type = new_gpu_type
                _htc_job.resource_quota.gpu_num = prev_gpu_num
                # Migrate as a new job
                # In this search, we only modify crs_table.
                is_feasible, _, _, _, _, crs_table = self._migration_search(_htc_job, crs_table, 
                                                                 global_mgrt_num, local_mgrt_num, 
                                                                 is_clear_target_job_placement=True)
                assert is_feasible
                # Both htc gain and the migration gain of the htc job
                gain += max_gain_with_mgrt
                global_mgrt_num = new_global_mgrt_num
            else:
                # The target job is htc
                _target_job.resource_quota.gpu_type = new_gpu_type
                _target_job.resource_quota.gpu_num = prev_gpu_num
                
                # To be migrated later
                if len(prev_jni_list) > 0:
                    # Clear job placement of the target job if previously registered in crs_table
                    is_modified = self._clear_placement(htc_job_id, crs_table)
                    assert is_modified

                # Only htc gain, since the migration of the target job is performed below
                gain += max_gain
        else:
            # Can not find a job to htc
            return (False, crs_table, is_htc, EMPTY_JOB_ID, None, gain, 
                    INFEASIBLE_THR, False, global_mgrt_num, global_mgrt_num)

        # Apply migration search for the target job
        # In this search, we modify _crs_table and global_mgrt_num_with_mgrt 
        # (_target_job is modified above).
        _crs_table = deepcopy(crs_table)
        (is_feasible, _, mgrt_gain, 
         is_relaxed, global_mgrt_num_with_mgrt, 
         _crs_table) = self._migration_search(_target_job, _crs_table, 
                                              global_mgrt_num, local_mgrt_num, 
                                              is_clear_target_job_placement=True)
        # HTC info, locate here in case that the htc job is the target job.
        _overwrite_job = _target_job if htc_job_id == _target_job.uuid else None
        job_gpu_type, new_gpu_num, new_locality, _ = self._get_job_rt_stat(htc_job_id, _crs_table, 
                                                                           _overwrite_job)
        assert job_gpu_type == new_gpu_type, \
            f"Mismatched gpu type in htc from {job_gpu_type} to {new_gpu_type}."
        htc_info = (htc_job_id, prev_gpu_type, prev_gpu_num, prev_locality, 
                    new_gpu_type, new_gpu_num, new_locality) if is_htc else None

        return (is_feasible, _crs_table, is_htc, htc_job_id, htc_info, gain, 
                mgrt_gain, is_relaxed, global_mgrt_num, global_mgrt_num_with_mgrt)

    def _is_no_infeasible_job(self, mdf_queue: Sequence[list]):
        """ 
        Check whether a series of modifying operations lead to the infeasibility of certain job. 
        """
        if self._dummy_test:
            return True
        # Check        
        infeasible_job_id_list = list()
        for _info in mdf_queue:
            if _info[0] == IS_SHRINKED:
                _job_id = _info[1][0]
                _gpu_type = _info[1][1]
                _new_locality = _info[1][5]
            else:
                assert _info[0] == IS_HTC
                _job_id = _info[1][0]
                _gpu_type = _info[1][4]
                _new_locality = _info[1][6]

            _is_feasible = self._is_alloc_feasible(self.get_job_by_uuid(_job_id), _gpu_type, 
                                                   _new_locality)
            if not _is_feasible and _job_id not in infeasible_job_id_list:
                infeasible_job_id_list.append(_job_id)
            else:
                infeasible_job_id_list = remove_if_exist(infeasible_job_id_list, [_job_id])

        return (len(infeasible_job_id_list) == 0)

    def _job_downgrade_search(self, target_job: Job, crs_table: dict, iter_num: int, 
                              global_mgrt_num: int, local_mgrt_num: int):
        """
        Search for job downgrade plan (shrink and Hardware Type Change (HTC)) among all 
        accepted jobs for at most iter_num rounds based on structured marginal gain.
        Notes:
            - The search result (crs_table) should be consistantly reserved for job 
              shrink and hardware type change, but restored for inline migration 
              search (_crs_table).
        -------------------------------------------------------------
        Return:
            - is_feasible: Whether can find a feasible solution.
            - crs_table_queue: A list of candidate plans.
        """
        fsb_plans = list()                          # A list of completed and feaisible plans
        mdf_queue = list()                          # A list of modifying infos (job shrink or GPU type change)
        # Variables related to BFS process
        bfs_input_queue = list()                    # Input tmp plan for the current round
        bfs_output_queue = list()                   # Output tmp plan of the current round
        # Init input queue  
        _crs_table = deepcopy(crs_table)
        # _target_job = target_job.deepcopy()         # Modified during the search
        _target_job = copy.deepcopy(target_job)
        _mdf_queue = deepcopy(mdf_queue)
        bfs_input_queue.append(
            SearchPlan(_crs_table, _target_job, 0.0, global_mgrt_num, 
                       modified_flag=IS_NO_MODIFIED, mdf_queue=_mdf_queue)
        )

        # BFS process
        _iter_cnt = 0
        while (len(bfs_input_queue) > 0 or len(bfs_output_queue) > 0) and _iter_cnt < iter_num:
            _iter_cnt += 1
            # Traverse the search plan in the input queue
            while len(bfs_input_queue) > 0:
                # Step 0. Get a tmp plan
                tmp_plan = bfs_input_queue.pop()

                # Step 1. Try job shrink
                # Debug mode
                if self.verbose:
                    print("")
                    print(f"[I] Iter {_iter_cnt}: Try job shrink...")
                    print(" - Idle resource status before entering this shrink search:")
                    self._print_idle_resources(tmp_plan.crs_table)

                for cur_gpu_type in self.supported_gpu_types:
                    # Check existance
                    if not self._is_type_existed(tmp_plan.crs_table, cur_gpu_type):
                        # The GPU type is not existed
                        continue
                    # Debug mode
                    if self.verbose:
                        print(f"[I] Search job shrink for GPU type '{cur_gpu_type}'")
                    
                    _crs_table = deepcopy(tmp_plan.crs_table)
                    # _target_job = tmp_plan.target_job.deepcopy()
                    # _target_job = copy.deepcopy(tmp_plan.target_job)
                    # _target_job = pickle.loads(pickle.dumps(tmp_plan.target_job))
                    _target_job = deepcopy_pickle(tmp_plan.target_job)
                    _gain = tmp_plan.gain
                    _global_mgrt_num = tmp_plan.global_mgrt_num
                    _mdf_queue = deepcopy(tmp_plan.mdf_queue)
                    # Candidate job id list, since target_job won't be modified, 
                    # we directly use target_job.
                    cand_job_id_list = self._get_run_job_ids(target_job, _crs_table)
                    
                    # Attempt one job shrink
                    # In this search, we modify _crs_table, _target_job, _gain and _global_mgrt_num.
                    if not self.disable_scaling:
                        (is_feasible, _tmp_crs_table, is_shrinked, shrinked_job_id, shrink_info, 
                        _gain, mgrt_gain, is_relaxed, 
                        _global_mgrt_num_with_mgrt) = self._shrink_one_job_and_migration_search(_target_job, cur_gpu_type, 
                                                                                                cand_job_id_list, _crs_table,
                                                                                                _gain, _global_mgrt_num, 
                                                                                                local_mgrt_num)
                    else:
                        is_feasible, is_shrinked, is_relaxed = False, False, False
                        (_tmp_crs_table, shrinked_job_id, shrink_info, 
                         _gain, mgrt_gain, _global_mgrt_num_with_mgrt) = None, None, None, None, None, None

                    # Debug mode
                    if shrinked_job_id != EMPTY_JOB_ID and self.verbose:
                        _job_alias = self.get_job_by_uuid(shrinked_job_id).alias \
                            if shrinked_job_id != EMPTY_JOB_ID else 'none'
                        print(f"[I] Iter {_iter_cnt}: Shinked job alias: {_job_alias} | " + 
                              f"GPU type: {cur_gpu_type} Is feasible: {is_feasible}")
                        if shrinked_job_id != EMPTY_JOB_ID:
                            print(f" - Prev GPU num: {shrink_info[2]} | Prev locality: {shrink_info[3]}")
                            print(f" - New GPU num: {shrink_info[4]} | New locality: {shrink_info[5]}")
                    
                    if is_shrinked:
                        # Record
                        _mdf_queue.append([IS_SHRINKED, shrink_info])

                    if is_feasible and self._is_no_infeasible_job(_mdf_queue):
                        # Record both shrink/htc and migration result
                        fsb_plans.append(
                            SearchPlan(_tmp_crs_table, _target_job, (_gain + mgrt_gain), 
                                       _global_mgrt_num_with_mgrt, 
                                       modified_flag=IS_SHRINKED, mdf_queue=_mdf_queue)
                        )
                    
                    if (not is_feasible or is_relaxed) and is_shrinked:
                        # Only record shrink/htc result
                        bfs_output_queue.append(
                            SearchPlan(_crs_table, _target_job, _gain, _global_mgrt_num, 
                                       modified_flag=IS_SHRINKED, mdf_queue=_mdf_queue)
                        )
                
                # Step 2. Try hardware (GPU) type change (HTC)
                # Debug mode
                if self.verbose:
                    print("")
                    print(f"[I] Iter {_iter_cnt}: Try job HTC...")
                    print(" - Idle resource status before entering this htc search:")
                    self._print_idle_resources(tmp_plan.crs_table)

                _crs_table = deepcopy(tmp_plan.crs_table)
                # _target_job = tmp_plan.target_job.deepcopy()
                # _target_job = copy.deepcopy(tmp_plan.target_job)
                # _target_job = pickle.loads(pickle.dumps(tmp_plan.target_job))
                _target_job = deepcopy_pickle(tmp_plan.target_job)
                _gain = tmp_plan.gain
                _global_mgrt_num = tmp_plan.global_mgrt_num
                _mdf_queue = deepcopy(tmp_plan.mdf_queue)
                # Candidate job id list, since target_job won't be modified, 
                # we directly use target_job
                cand_job_id_list = self._get_run_job_ids(target_job, _crs_table)
                
                # Attempt one GPU type change of a job
                # In this search, we modify _crs_table, _target_job, _gain and _global_mgrt_num.
                if not self.disable_htc:
                    (is_feasible, _tmp_crs_table, is_htc, htc_job_id, htc_info, 
                    _gain, mgrt_gain, is_relaxed, _global_mgrt_num, 
                    _global_mgrt_num_with_mgrt) = self._htc_one_job_and_migration_search(_target_job, cand_job_id_list, 
                                                                                        _crs_table, _gain, 
                                                                                        _global_mgrt_num, local_mgrt_num)
                else:
                    is_feasible, is_htc, is_relaxed = False, False, False
                    (_tmp_crs_table, htc_job_id, htc_info, 
                     _gain, mgrt_gain, _global_mgrt_num_with_mgrt) = None, None, None, None, None, None

                # Debug mode
                if self.verbose:
                    _job_alias = self.get_job_by_uuid(htc_job_id).alias if htc_job_id != EMPTY_JOB_ID else "none"
                    print(f"[I] Iter {_iter_cnt}: HTC job alias: {_job_alias} | Is feasible: {is_feasible}")
                    if htc_job_id != EMPTY_JOB_ID:
                        print(f" - Prev GPU type: {htc_info[1]} | Prev GPU num: {htc_info[2]} | " + 
                              f"Prev locality: {htc_info[3]}")
                        print(f" - New GPU type: {htc_info[4]} | New GPU num: {htc_info[5]} | " + 
                              f"New locality: {htc_info[6]}")

                if is_htc:
                    # Record
                    _mdf_queue.append([IS_HTC, htc_info])
                
                if is_feasible and self._is_no_infeasible_job(_mdf_queue):
                    # Record both shrink/htc and migration result
                    fsb_plans.append(
                        SearchPlan(_tmp_crs_table, _target_job, (_gain + mgrt_gain), _global_mgrt_num_with_mgrt, 
                                   modified_flag=IS_HTC, mdf_queue=_mdf_queue)
                    )
                
                if (not is_feasible or is_relaxed) and is_htc:
                    # Only record shrink/htc result
                    bfs_output_queue.append(
                        SearchPlan(_crs_table, _target_job, _gain, _global_mgrt_num, 
                                   modified_flag=IS_HTC, mdf_queue=_mdf_queue)
                    )
        
            # Update input queue
            for tmp_plan in bfs_output_queue:
                bfs_input_queue.append(tmp_plan)
        
        return (len(fsb_plans) > 0), fsb_plans

    ######################################
    #    Reservation Related Functions   #
    ######################################
    
    def _get_reserved_resources(self, job: Job):
        """ Get the available resources of the pending job (blocker or vanilla pending job). """
        assert not job.is_opportunism
        # Cluster resource status table
        crs_table = self.resource_interface.get_crs_table()
        if not job.is_blocker:
            # Vanilla pending job, only idle resources are available
            return crs_table
        
        # Blocker, add resources occupied by the opportunism jobs with lower priority
        _job_idx = self.decision_queue.index(job)
        for _idx in range(_job_idx + 1, len(self.decision_queue), 1):
            job = self.decision_queue[_idx]
            if not job.is_blocker and not job.is_opportunism:
                # Vanilla pending job, there can't exist opportunism job below
                break
            if job.is_opportunism and job.resource_alloc:
                # The opportunism job may be resubmitted (in runtime), in which case 
                # the resource alloc would be None. We ignore the suspend of this job.
                # Recolor occupied resources as available
                for _nid in job.resource_alloc.node_to_gpu_table:
                    for _gid in job.resource_alloc.node_to_gpu_table[_nid]:
                        assert crs_table[_nid][_gid]["used_job_id"] == job.uuid
                        # Recolor as bubbles, only modify status, use used_job_id as the 
                        # identifier to recover.
                        crs_table[_nid][_gid]["status"] = IDLE_STATUS
                        # crs_table[node_id][_gid]['used_job_id'] = EMPTY_JOB_ID
        
        return crs_table
    
    def _analyze_resource_contention(self, prev_crs_table: dict, decision_crs_table: dict):
        """ 
        Analyze the resource contention between the pending job (to be restarted) and 
        the related opportunism jobs. 
        """
        conflict_job_id_list = list()   # List of opportunism jobs with resource contention to the pending job
        prev_jgm_table = dict()         # Previous job id -> occupied GPU num table
        cur_jgm_table = dict()          # Current job id -> occupied GPU num table

        for _nid in prev_crs_table:
            for _gid in prev_crs_table[_nid]:
                # Statistic the previous running job id, including conflict and non-conflict opportunism jobs
                prev_jgm_table = dict_counter_add(prev_jgm_table, prev_crs_table[_nid][_gid]['used_job_id'], 1)
                # Statistic the current running job id, the conflict opportunism jobs are completedly 
                # or partially overwrited
                cur_jgm_table = dict_counter_add(cur_jgm_table, decision_crs_table[_nid][_gid]['used_job_id'], 1)
        
        # Get conflict job id list
        for _job_id in prev_jgm_table:
            if _job_id == EMPTY_JOB_ID or not self.get_job_by_uuid(_job_id).is_opportunism:
                # Bubbles or not an opportunism job
                continue
            if _job_id not in cur_jgm_table and _job_id not in conflict_job_id_list:
                # Completedly overwrited
                conflict_job_id_list.append(_job_id)
            if (_job_id in cur_jgm_table and prev_jgm_table[_job_id] != cur_jgm_table[_job_id] and 
                _job_id not in conflict_job_id_list):
                # Partially overwrited, also tagged as conflict job
                assert prev_jgm_table[_job_id] > cur_jgm_table[_job_id]
                conflict_job_id_list.append(_job_id)

        return conflict_job_id_list

    def _restart_one_pending_job(self, pending_job: Job):
        """ Restart one pending job (blocker or vanilla pending job) as a normal running job. """
        # Remove from the pending queue
        assert pending_job in self.pending_job_queue
        self.pending_job_queue.remove(pending_job)
        # Add to the running queue
        assert pending_job not in self.running_job_queue
        self.running_job_queue.append(pending_job)
        # Remove a blocker or a vanilla pending queue, just remove from the decision_queue
        assert pending_job in self.decision_queue
        self.decision_queue.remove(pending_job)
        if pending_job.is_blocker:
            pending_job.is_blocker = False
        # Update job status
        assert pending_job.status == JOB_PENDING_STATUS
        pending_job.update_status(JOB_RUNNING_STATUS)

    def _suspend_one_job(self, job: Job):
        """ To suspend one job (normally an opportunism job). """
        # Update the decision_queue if needed
        if job in self.decision_queue:
            assert job.is_opportunism
            # Update tag
            job.is_opportunism = False
            # Update decision queue
            _job_idx = self.decision_queue.index(job)
            # First, check whether there exists another opportunism job below
            is_existed = False
            for _idx in range(_job_idx + 1, len(self.decision_queue), 1):
                if self.decision_queue[_idx].is_opportunism:
                    is_existed = True
                    break
            # Then, update upper blockers if needed
            if not is_existed:
                # Tag this job as a vanilla pending job
                job.is_blocker = False
                # Update
                for _idx in range(_job_idx - 1, -1, -1):
                    if self.decision_queue[_idx].is_blocker:
                        # Re-tag as vanilla pending job
                        self.decision_queue[_idx].is_blocker = False
                    elif self.decision_queue[_idx].is_opportunism:
                        # Found another upper opportunism job, break
                        break
                    else:
                        raise RuntimeWarning("Error: A vanilla pending job appears " + 
                                             "above the previous opportunism job.")
            else:
                # Exist another opportunism job below, just need to modify the tag of this job
                job.is_blocker = True

        # Update job stautus
        assert job.status == JOB_RUNNING_STATUS
        job.update_status(JOB_PENDING_STATUS)
        # Update job iter num
        job.iter_num = job.remained_iter_num
        # Clear the allocated resources of the job
        job.resource_alloc = None

        # Remove from the running queue
        assert job in self.running_job_queue
        self.running_job_queue.remove(job)
        # Add to the pending queue
        assert job not in self.pending_job_queue
        self.pending_job_queue.append(job)

    def _untag_one_opportunism_job(self, job: Job):
        """ Untag as a normal running job. """
        assert (job in self.running_job_queue) and (job not in self.pending_job_queue)
        # Remove from the decision queue
        self.decision_queue.remove(job)
        # Untag the identity of opportunism job
        job.is_opportunism = False

    def _pending_jobs_restart_trial(self, overwrite_restart_trial_num: int = 0):
        """ 
        Sequentially try to restart pending jobs (blocker or vanilla pending job) in 
        the decision queue. Since the order in the pending queue also follows the submission 
        priority order (init job priority & submitted round)
        """
        if self.verbose:
            print("")
        print("[I] Begin Pending Jobs Restart Trial...")
        if len(self.pending_job_queue) == 0:
            print("[I][REST] No blocker or vanilla pending job exists.")
            return 
    
        is_restart_plan_found = True
        restarted_job_idx = 0
        restarted_info_queue = list()
        restart_cnt = 0

        restart_trial_num = overwrite_restart_trial_num if overwrite_restart_trial_num > 0 else MAX_RESTART_TRIAL_NUM
        while is_restart_plan_found and restart_cnt < restart_trial_num:
            restart_cnt += 1
            if len(self.pending_job_queue) == 0:
                break

            is_restart_plan_found = False
            # Calculate global and local migration num
            global_mgrt_num, local_mgrt_num_table = self._get_migration_num_bound(self.pending_job_queue)

            # For ablation study
            _blocked_job_gpu_type = list()
    
            infeasible_job_infos = list()
            for _i, job in enumerate(self.decision_queue):
                if _i < restarted_job_idx:
                    # Already tried to restart and failed with even more idle resources
                    continue

                if self.enable_ddl and job.uuid in self.timeout_job_id_list:
                    # Timeout job
                    continue

                # if (
                #     self.prepend_profile_overhead and
                #     # (self.global_timer - job.sub_time) < self.per_job_profile_overhead
                #     job.profile_time_budget > 0
                # ):
                #     # This job is still not profiled or in profiling
                #     continue
                
                info_key = f"{job.model_name}_{job.batch_size}"
                if info_key in infeasible_job_infos:
                    # Already been searched
                    print(f"[TMP] Job info (model name: {job.model_name}, " + 
                          f"batch size: {job.batch_size}) has been profiled, loading cache...")
                    continue
                
                if job.is_opportunism:
                    # The opportunism job does not need to be restarted
                    continue
                assert job in self.pending_job_queue

                if self.disable_opportunistic and job.resource_quota.gpu_type in _blocked_job_gpu_type:
                    # Disable opportunistic 
                    continue

                # Count available resources
                crs_table = self._get_reserved_resources(job)
                # Try to fit the pending job
                (is_feasible, decision_crs_table, 
                 remained_global_mgrt_num) = self._fit_one_job(crs_table, job, global_mgrt_num, 
                                                               local_mgrt_num_table[job.uuid])
                # Check whether to restart the pending job
                if not is_feasible:
                    if self.disable_opportunistic:
                        # Block this GPU type
                        assert job.resource_quota.gpu_type not in _blocked_job_gpu_type
                        _blocked_job_gpu_type.append(job.resource_quota.gpu_type)
                    # Record key
                    assert info_key not in infeasible_job_infos, \
                        "Infeasible model configuration should not be searched twice."
                    infeasible_job_infos.append(info_key)
                    # Not able to be restarted
                    continue
                else:
                    is_restart_plan_found = True
                    restarted_job_idx = _i
                    # Analyze the resource contention with opportunism jobs
                    prev_crs_table = self.resource_interface.get_crs_table()
                    conflict_job_id_list = self._analyze_resource_contention(prev_crs_table, 
                                                                             decision_crs_table)
                    # Construct scheduling decision
                    for _nid in decision_crs_table:
                        for _gid in decision_crs_table[_nid]:
                            # Lead to wrong result when in decision_crs_table, the restarting job performs 
                            # downgrade search and occupies the GPUs that is occupied by other vanilla 
                            # running jobs in prev_crs_table.

                            # if decision_crs_table[_nid][_gid]["used_job_id"] != job.uuid and \
                            #    decision_crs_table[_nid][_gid]["used_job_id"] != prev_crs_table[_nid][_gid]["used_job_id"]:
                            if (decision_crs_table[_nid][_gid]["status"] == IDLE_STATUS and 
                                decision_crs_table[_nid][_gid]["used_job_id"] != EMPTY_JOB_ID):
                                assert self.get_job_by_uuid(decision_crs_table[_nid][_gid]["used_job_id"]).is_opportunism
                                # Record the occupation of the conflict-free opportunism jobs
                                decision_crs_table[_nid][_gid]["status"] = USED_STATUS
                    
                    # Clear all resources occupied by partially conflicted opportunism jobs.
                    for _job_id in conflict_job_id_list:
                        _ = self._clear_placement(_job_id, decision_crs_table)
                    
                    # Restart this pending job
                    self._restart_one_pending_job(job)
                    # Suspend the opportunism jobs that lead to resource contention
                    for _job_id in conflict_job_id_list:
                        self._suspend_one_job(self.get_job_by_uuid(_job_id))
                    
                    # Untag all conflict-free opportunism jobs if no longer below a blocker
                    untagged_job_id_list = list()
                    for _job in self.decision_queue:
                        if _job.is_opportunism and _job.resource_alloc:
                            # The opportunism job may be resubmitted (in runtime), in which case 
                            # the resource alloc would be None. We ignore the suspend of this job.
                            # Untag as a normal running job
                            untagged_job_id_list.append(_job.uuid)
                        else:
                            # Blocker or vanilla pending job
                            break
                    for _job_id in untagged_job_id_list:
                        self._untag_one_opportunism_job(self.get_job_by_uuid(_job_id))
                    
                    # Apply the scheduling decision
                    self.resource_interface.apply_sched(decision_crs_table)
                    # Update the resource status of running jobs
                    self._update_run_jobs(decision_crs_table)
                    
                    # Record
                    (job_gpu_type, job_gpu_num, 
                     locality, _) = self._get_job_rt_stat(job.uuid, decision_crs_table, 
                                                          overwrite_job=job)
                    restarted_info = [job.uuid, job_gpu_type, job_gpu_num, locality, 
                                      conflict_job_id_list, untagged_job_id_list]
                    restarted_info_queue.append(restarted_info)
                    # Update global migration num
                    global_mgrt_num = remained_global_mgrt_num 
                    # Debug mode
                    if self.verbose:
                        self._get_occ_resources(decision_crs_table, job.uuid)
                    # Continue searching from the queue head
                    break
        
        if self.verbose:
            print("")
        if len(restarted_info_queue) == 0:
            print("[I][REST] No Restarting plan is found among pending jobs.")
        else:
            print("[I][REST] The Restarting plans are sequentially presented as follows:")
            for restarted_info in restarted_info_queue:
                _idx = restarted_info_queue.index(restarted_info)
                _job_alias = self.get_job_by_uuid(restarted_info[0]).alias
                _suspended_job_alias_list = [
                    self.get_job_by_uuid(_uuid).alias for _uuid in restarted_info[4]
                ]
                _untagged_job_alias_list = [
                    self.get_job_by_uuid(_uuid).alias for _uuid in restarted_info[5]
                ]
                print(f" - Plan {_idx + 1}: Job alias: {_job_alias} | GPU type: {restarted_info[1]} | " + 
                      f"GPU num: {restarted_info[2]} | Locality: {restarted_info[3]} | " + 
                      f"Suspended jobs: {_suspended_job_alias_list} | Untagged jobs: {_untagged_job_alias_list}")

    def _is_exist_pend_job_same_gpu_type(self, gpu_type: str):
        """ Check whether there exists a pending job with the same GPU type. """
        for _job in self.pending_job_queue:
            if _job.resource_quota.gpu_type == gpu_type:
                return True
        return False
    
    def _drop_all_timeout_jobs_in_pending_queue(self):
        """ Drop all timeout jobs in pending queue. """
        for _job in self.pending_job_queue:
            if (_job.deadline <= self.global_timer and 
                _job.uuid not in self.timeout_job_id_list):
                print(f"[I][DDL] Job '{_job.alias}' has been dropped due to timeout.")
                self.timeout_job_id_list.append(_job.uuid)

        # _dropped_job_list = list()
        # for _job in self.pending_job_queue:
        #     if not _job.deadline <= self.global_timer:
        #         _dropped_job_list.append(_job)
        # # Drop
        # for _job in _dropped_job_list:
        #     print(f"[I][DDL] Job '{_job.alias}' has been dropped due to timeout.")
        #     self.pending_job_queue.remove(_job)
    
    ######################################
    #      Upgrade Related Functions     #
    ######################################

    def _job_upgrade_search(self, target_job: Job, global_mgrt_num: int, local_mgrt_num: int):
        """ 
        Search for an one-step job upgrade plan (grow and Hardware Type Change (HTC)) 
        for the target running job. 
        -------------------------------------------------------------
        Return:
            - is_feasible: Whether can find a feasible (and better than current 
                           configuration) upgrade plan.
            - decision_crs_table: Decision crs_table that contains the result of 
                                  the one-step job upgrade.
            - gain: The structured marginal gain obtained from this job upgrade.
            - global_mgrt_num: The remained global migration num.
        """
        crs_table = self.resource_interface.get_crs_table()
        # Clear the placement of the target job (to be reallocated)
        is_modified = self._clear_placement(job_id=target_job.uuid, crs_table=crs_table)
        assert is_modified
        # Best plan for the maximal structured marginal gain
        max_gain = INFEASIBLE_THR
        decision_crs_table = None
        remained_global_mgrt_num = None
        # About job
        target_gpu_type = target_job.resource_quota.gpu_type
        target_gpu_num = target_job.resource_quota.gpu_num
        target_locality = target_job.resource_quota.locality

        # First, Try htc
        if not self.disable_htc:
            for _gpu_type in self.supported_gpu_types:
                if _gpu_type == target_gpu_type:
                    # Same type as the prev
                    continue
                if not self._is_sufficient(crs_table, _gpu_type, 
                                        target_gpu_num):
                    # The resources is insufficient
                    continue
                
                # Apply migration search for the target job
                # In this search, we modify _job, _crs_table and _global_mgrt_num
                # _job = target_job.deepcopy()
                # _job = copy.deepcopy(target_job)
                # _job = pickle.loads(pickle.dumps(target_job))
                _job = deepcopy_pickle(target_job)
                _job.resource_quota.gpu_type = _gpu_type
                _crs_table = deepcopy(crs_table)
                # Try migration search
                (is_feasible, _, _, _, 
                _global_mgrt_num, _crs_table) = self._migration_search(_job, _crs_table, 
                                                                        global_mgrt_num, local_mgrt_num,
                                                                        is_clear_target_job_placement=True)
                _, _, new_locality, _ = self._get_job_rt_stat(target_job.uuid, _crs_table, overwrite_job=_job)
                
                # print(new_locality)
                
                # Structured marginal gain
                _gain = self._get_delta_thr(_job, _gpu_type, target_gpu_num, new_locality, 
                                            target_gpu_type, target_gpu_num, target_locality)
                # Update 
                if is_feasible and _gain > max_gain:
                    max_gain = _gain
                    decision_crs_table = _crs_table
                    remained_global_mgrt_num = _global_mgrt_num
        
        # Then, try grow GPU num
        if not self.disable_scaling:
            # max_gpu_num = MAX_SUPPORTED_GPU_NUM if not self.is_runtime else NODE_CAPACITY[target_gpu_type]
            max_gpu_num = MAX_SUPPORTED_GPU_NUM
            if target_gpu_num < max_gpu_num:
                new_gpu_num = target_gpu_num * 2
                if self._is_sufficient(crs_table, target_gpu_type, new_gpu_num):
                    # Apply migration search for the target job
                    # In this search, we modify _job, _crs_table and _global_mgrt_num
                    # _job = target_job.deepcopy()
                    # _job = copy.deepcopy(target_job)
                    # _job = pickle.loads(pickle.dumps(target_job))
                    _job = deepcopy_pickle(target_job)
                    _job.resource_quota.gpu_num = new_gpu_num
                    _crs_table = deepcopy(crs_table)
                    # Try migration search
                    (is_feasible, _, _, _, 
                    _global_mgrt_num, _crs_table) = self._migration_search(_job, _crs_table, 
                                                                            global_mgrt_num, local_mgrt_num, 
                                                                            is_clear_target_job_placement=True)
                    _, _, new_locality, _ = self._get_job_rt_stat(target_job.uuid, _crs_table, 
                                                                overwrite_job=target_job)
                    # Structured marginal gain
                    _gain = self._get_delta_thr(_job, target_gpu_type, new_gpu_num, new_locality, 
                                                target_gpu_type, target_gpu_num, target_locality)
                    # Update
                    if is_feasible and _gain > max_gain:
                        max_gain = _gain
                        decision_crs_table = _crs_table
                        remained_global_mgrt_num = _global_mgrt_num
        
        if max_gain > 0:
            # Find a feasible plan and the cluster throughput better than the current allocation
            return True, decision_crs_table, max_gain, remained_global_mgrt_num
        else:
            return False, None, INFEASIBLE_THR, global_mgrt_num

    def _running_jobs_upgrade_trial(self):
        """ 
        Upgrade running jobs step by step with maximizing structured marginal gain.
        In each iteration, we find one running job to upgrade, then apply the scheduling 
        decision and update job entries.
        """
        if self.verbose:
            print("")
        print("[I] Begin Running Jobs Upgrade Trial...")

        # Check whether the running jobs queue is empty
        if len(self.running_job_queue) == 0:
            print("[I][UPGD] Currently no running job exists.")
            # No running job, return
            return
        
        is_upgrade_plan_found = True
        # Calculate global and local migration num
        global_mgrt_num, local_mgrt_num_table = self._get_migration_num_bound(self.running_job_queue)
        # Upgraded info
        upgraded_info_queue = list()
        _upgraded_cnt = 0
        # Loop until no feasible plan is found
        while is_upgrade_plan_found and _upgraded_cnt < MAX_UPGRADE_STEP_NUM:
            _upgraded_cnt += 1
            is_upgrade_plan_found = False
            # Search for the best plan with the maximized structured marginal gain
            _max_gain = INFEASIBLE_THR
            upgraded_job_id = None
            decision_crs_table = None
            remained_global_mgrt_num = None

            for _job in self.running_job_queue:
                # Try upgrade this job
                (is_feasible, _crs_table, 
                 _gain, _global_mgrt_num) = self._job_upgrade_search(_job, global_mgrt_num, 
                                                                     local_mgrt_num_table[_job.uuid])
                
                if is_feasible and _gain > _max_gain:
                    _gpu_type, _, _locality, _ = self._get_job_rt_stat(_job.uuid, _crs_table, 
                                                                       overwrite_job=_job)
                    # Check the profiling database whether this new resource quota 
                    # is valid and profiled.
                    is_feasible = self._is_alloc_feasible(_job, _gpu_type, _locality)

                    # # NOTE: In runtime, we forbid the cross-nodes placement of certain GPU type due to network legacy, which would cause Runtime error
                    # # Get current job info
                    # if self.is_runtime and _gpu_type in forbid_cross_nodes_gpu_type and len(_locality) > 1:
                    #     is_feasible = False

                if is_feasible and _gain > _max_gain:
                    _max_gain = _gain
                    upgraded_job_id = _job.uuid   
                    decision_crs_table = _crs_table
                    remained_global_mgrt_num = _global_mgrt_num
            
            # Apply upgrade decision for one step
            if upgraded_job_id:
                is_upgrade_plan_found = True
                # Get prev job info
                _upgraded_job = self.get_job_by_uuid(upgraded_job_id)
                _crs_table = self.resource_interface.get_crs_table()
                (prev_gpu_type, prev_gpu_num, 
                 prev_locality, _) = self._get_job_rt_stat(upgraded_job_id, _crs_table)
                # Update real cluster resources
                self.resource_interface.apply_sched(decision_crs_table)
                # Update the resource status of running jobs
                self._update_run_jobs(decision_crs_table)
                # Record
                _upgraded_job = self.get_job_by_uuid(upgraded_job_id)
                new_gpu_type, new_gpu_num, new_locality, _ = self._get_job_rt_stat(upgraded_job_id, 
                                                                                   decision_crs_table)
                _upgraded_info = [upgraded_job_id, prev_gpu_type, prev_gpu_num, prev_locality, 
                                  new_gpu_type, new_gpu_num, new_locality]
                upgraded_info_queue = unique_append(upgraded_info_queue, _upgraded_info)
                # Update global migration num
                global_mgrt_num = remained_global_mgrt_num
                # Debug mode
                if self.verbose:
                    print(f"[I][UPGD] Decide to upgrade job '{_upgraded_job.alias}' (alias)...")
                    # Print occupied resources
                    self._get_occ_resources(decision_crs_table, upgraded_job_id)

        if self.verbose:
            print("")
        if len(upgraded_info_queue) == 0:
            print("[I][UPGD] No upgrading plan is found among running jobs.")
        else:
            print("[I][UPGD] The upgrading plans are sequentially presented as follows:")
            for _info in upgraded_info_queue:
                _idx = upgraded_info_queue.index(_info)
                _job_alias = self.get_job_by_uuid(uuid=_info[0]).alias
                print(f" - Plan {_idx + 1}: Job alias: {_job_alias} | Prev GPU type: {_info[1]} | " + 
                      f"Prev GPU num: {_info[2]} | Prev locality: {_info[3]} | New GPU type: {_info[4]} " + 
                      f"| New GPU num: {_info[5]} | New locality: {_info[6]}")
                
    # def _launch_profiled_jobs(self):
        # """ Launch jobs that complete profile. """


    ######################################
    #        Top-level Functions         #
    ######################################

    def _fit_one_job(self, crs_table: dict, job: Job, global_mgrt_num: int, local_mgrt_num: int):
        """ Perform bubble migration search and job downgrade search of one job. """
        fsb_plans = list()                  # Feasible plan queue
        decision_crs_table = None           # Scheduling decision
        remained_global_mgrt_num = None     # Remained global migration num
        _time_marker = time.time()
        
        print("")
        print(f"[I] Scheduling Job alias '{job.alias}' | Model name: {job.model_name} | " + 
              f"Batch size: {job.batch_size} " + 
              f"| Job GPU type: {job.resource_quota.gpu_type} " + 
              f"| Job GPU num: {job.resource_quota.gpu_num}")
        # Debug mode
        if self.verbose:
            print("[I] Idle resource status before entering this iteration:")
            # Print idle GPU num
            self._print_idle_resources(crs_table)
        
        # Step 1. Perform RHD Bubble Migration Search
        if self.verbose:
            print("")
        print("[I] Begin vanilla bubble migration search...")
        # In this search, we modify _crs_table and _global_mgrt_num
        _crs_table = deepcopy(crs_table)
        (is_feasible, _, mgrt_gain, 
         is_relaxed, _global_mgrt_num, _crs_table) = self._migration_search(job, _crs_table, 
                                                                global_mgrt_num, local_mgrt_num,
                                                                is_clear_target_job_placement=True)
        # Record the result as a candidate plan
        if is_feasible:
            fsb_plans.append(
                SearchPlan(_crs_table, job, mgrt_gain, _global_mgrt_num,
                           modified_flag=IS_NO_MODIFIED, mdf_queue=list())
            )

        # Step 2. Perform Job Downgrade Search
        if (not is_feasible or is_relaxed):
            # If the plan is not feasible or the result locality of the job is relaxed.
            if self.verbose:
                print("")
            print("[I] Begin iterative job shrink and HTC search...")
            # In this search, we maintain different versions of crs_table, target_job, 
            # gain and global_mgrt_num in different plan branch.
            # The orginal crs_table is not modified.
            if not self.is_runtime:
                _downgrade_max_depth = 3 if len(self.running_job_queue) <= 150 else 2
            else:
                _downgrade_max_depth = 4
            
            if os.environ.get("DOWNGRADE_MAX_DEPTH", "none") != "none":
                _downgrade_max_depth = int(os.environ.get("DOWNGRADE_MAX_DEPTH", "none"))
            
            is_feasible, _fsb_plans = self._job_downgrade_search(job, crs_table, 
                                                                 _downgrade_max_depth, 
                                                                 global_mgrt_num, local_mgrt_num)
            # Update candidate plans
            fsb_plans.extend(_fsb_plans)
            
            # Search for the best plan with the maximized structured marginal gain
            _max_gain = INFEASIBLE_THR
            best_idx = -1

            # Debug mode
            if self.verbose:
                print("[I] Candidate plans retrieved from migration search and job shrink search:")
                print("----------------------------------------")

            for _idx in range(len(fsb_plans)):
                # Debug mode
                if self.verbose:
                    print(f" - Plan {_idx + 1}: Structured Marginal Gain: {fsb_plans[_idx].gain} " + 
                          f"| Remained Global migration num: {fsb_plans[_idx].global_mgrt_num} | " + 
                          f"Is modified: {fsb_plans[_idx].modified_flag}")
                    self._print_modification(_idx, fsb_plans, crs_table)
                    
                # print(f" - Plan {_idx + 1}: Structured Marginal Gain: {fsb_plans[_idx].gain} " + 
                #       f"| Remained Global migration num: {fsb_plans[_idx].global_mgrt_num} | " + 
                #       f"Is modified: {fsb_plans[_idx].modified_flag}")
                
                # Update 
                if fsb_plans[_idx].gain > _max_gain:
                    _max_gain = fsb_plans[_idx].gain
                    best_idx = _idx

            if best_idx == -1:
                # No feasible plan found
                if not self.verbose:
                    print("----------------------------------------")
                print(" - No feasible plan found.")
                print("----------------------------------------")
                _sched_overhead = time.time() - _time_marker
                print(f"[I] Scheduling process takes {_sched_overhead} s.")
                if job.uuid not in self.job_sched_overhead_table:
                    self.job_sched_overhead_table[job.uuid] = [_sched_overhead]
                else:
                    self.job_sched_overhead_table[job.uuid].append(_sched_overhead)
                # Early return
                return False, None, global_mgrt_num
            else:
                if self.verbose:
                    print("----------------------------------------")
                    print(f" - Index of the best plan: {best_idx + 1}")
                    print("----------------------------------------")
                else:
                    print(f"[I] The optimized placing plan of job '{job.alias}' (alias) is:")
                    self._print_modification(best_idx, fsb_plans, crs_table)

            # Record as decision
            decision_crs_table = fsb_plans[best_idx].crs_table
            remained_global_mgrt_num = fsb_plans[best_idx].global_mgrt_num
        else:
            # if not self.disable_scaling:
            # Directly record the result of migration search
            assert len(fsb_plans) == 1
            decision_crs_table = fsb_plans[0].crs_table
            remained_global_mgrt_num = fsb_plans[0].global_mgrt_num
            print(f"[I] The optimized placing plan of job '{job.alias}' (alias) is:")
            self._print_modification(0, fsb_plans, crs_table)
            # else:
            #     if len(fsb_plans) > 0:
            #         # Migration search can find a feasible plan
            #         print(f"[I] The optimized placing plan of job '{job.alias}' (alias) is:")
            #         self._print_modification(0, fsb_plans, crs_table)
            #         print(f"[I] Scheduling process takes {time.time() - _time_marker} s.")
            #         return True, fsb_plans[0].crs_table, fsb_plans[0].global_mgrt_num
            #     # Migration serach can not find a feasible plan while the job scaling is disabled
            #     print(f"[I] Scheduling process takes {time.time() - _time_marker} s.")
            #     return False, None, None
        
        _sched_overhead = time.time() - _time_marker
        print(f"[I] Scheduling process takes {_sched_overhead} s.")
        if job.uuid not in self.job_sched_overhead_table:
            self.job_sched_overhead_table[job.uuid] = [_sched_overhead]
        else:
            self.job_sched_overhead_table[job.uuid].append(_sched_overhead)

        return (len(fsb_plans) > 0), decision_crs_table, remained_global_mgrt_num

    def _fit_new_jobs(self):
        """ 
        Fit new jobs, triggered by job arrival, enter if new job exists in self.submit_init_job_queue 
        during periodically schedule. 
        """
        print("")
        print("#################################################")
        print("#        Handling with New Job(s) Arrival       #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.submit_init_job_queue)} new jobs to be fitted...")
        # For eliminating unnecessary job suspend/resume
        prev_crs_table = self.resource_interface.get_crs_table()
        
        # Calculate global and local migration num
        global_mgrt_num, local_mgrt_num_table = self._get_migration_num_bound(self.submit_init_job_queue)
        # Debug mode
        if self.verbose:
            print("")
            print(f"[I] Global migration num: {global_mgrt_num}")
            print("[I] Local migration num dict:")
            for _jid in local_mgrt_num_table:
                print(f" - Job alias: {self.get_job_by_uuid(_jid).alias} " + 
                      f"| Migration num: {local_mgrt_num_table[_jid]}")
        
        # Sort jobs by init priority
        self.submit_init_job_queue = self._sort_jobs(self.submit_init_job_queue)
        
        # Step 1. Fit each new job
        for job in self.submit_init_job_queue:
            
            if self.prepend_profile_overhead and job.profile_time_budget is None:
                print(f"[I] Job '{job.alias}' (alias) has entered profiling.")
                job.profile_time_budget = self.per_job_profile_overhead
                self.in_profile_job_queue.append(job)
                continue
            
            # Push into decision queue
            self.decision_queue.append(job)

            # print(f"-------> Job {job.alias} | Decision queue: {[_job.alias for _job in self.decision_queue]}")
            
            # Cluster resource status table
            crs_table = self.resource_interface.get_crs_table()

            # Fit one job
            (is_feasible, decision_crs_table, 
             remained_global_mgrt_num) = self._fit_one_job(crs_table, job, global_mgrt_num, 
                                                           local_mgrt_num_table[job.uuid])
            
            if self.disable_opportunistic and is_feasible:
                # Disable opportunistic for ablation study
                is_feasible = not self._is_exist_pend_job_same_gpu_type(job.resource_quota.gpu_type)

            # print(f"-------> Job {job.alias} | Is fit feasible: {is_feasible}")
            
            # Check whether to perform: (1) apply scheduling decision; (2) job pending and resource reservation
            if is_feasible:
                # No need to pend this job, directly apply the scheduling decision 
                # and deploy on real resources.
                self.resource_interface.apply_sched(decision_crs_table)
                global_mgrt_num = remained_global_mgrt_num

                if len(self.decision_queue) == 1:
                    # Not opportunism job
                    _ = self.decision_queue.pop()
                else:
                    # Is opportunism job
                    job.is_opportunism = True
                    print(f"[I] Job '{job.alias}' (alias) has been running as an " + 
                          f"opportunism job, try update the decision queue...")
                    # Update previous jobs in decision queue
                    for _job in self.decision_queue[:-1]:
                        if not _job.is_blocker and not _job.is_opportunism:
                            # Vanilla pending job, set as blocker
                            _job.is_blocker = True
                            print(f" - Job '{_job.alias}' (alias) has been retagged as a blocker.")
                
                # Update job status
                job.update_status(JOB_RUNNING_STATUS)
                # Add to the running job queue in the scheduler
                self.running_job_queue.append(job)
                # Update the allocated resources of all running jobs.
                self._update_run_jobs(decision_crs_table)
                # Debug mode
                if self.verbose:
                    self._get_occ_resources(decision_crs_table, job.uuid)
            else:
                print(f"[I] Job '{job.alias}' (alias) has been pending as a vanilla pending job.")
                # No feasible plan found, need to pend this job
                job.update_status(JOB_PENDING_STATUS)
                # Add to the pending job queue in the scheduler
                self.pending_job_queue.append(job)
        
        # if self.prepend_profile_overhead:
        #     # Try launch jobs that complete profiling
        #     self._launch_profiled_jobs()

        # Step 2. Try to exploit any upgrade trials of all running jobs (mainly for new jobs)
        #         This can fix some downscale errors due to greedy-based search.
        self._running_jobs_upgrade_trial()
        
        # Step 3. Eliminate unnecessray suspend/resume by analyzing crs table changes
        self._eliminate_unnecessary_suspend_resume(prev_crs_table)
        
        print(f"[I] There are {len(self.running_job_queue)} jobs in running status...")
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")

    def _optimize_running_jobs(self, is_need_upgrade: bool):
        """ 
        Optimize running jobs, triggered by job departure, enter if ended job exists 
        in self.ended_job_queue during periodically schedule. 
        """
        print("")
        print("#################################################")
        print("#    Handling with Running Jobs Optimization    #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.ended_job_queue)} jobs are ended...")
        # For eliminating unnecessary job suspend/resume
        prev_crs_table = self.resource_interface.get_crs_table()
        
        # Step 1. Release all related resources of ended jobs and update decision queue
        ended_job_id_list = self._release_resources_and_update_decision_queue()

        # Step 2. Introspective Bubble Migration
        self._introspective_migration_search()

        # Step 3. Pending Jobs Restart Trial
        self._pending_jobs_restart_trial()

        # Step 4. Running Jobs Upgrade Trial (if will not upgrade in the 
        #         next new jobs arrival pass)
        if is_need_upgrade:
            self._running_jobs_upgrade_trial()
        
        # Step 5. Eliminate unnecessray suspend/resume by analyzing crs table changes
        self._eliminate_unnecessary_suspend_resume(prev_crs_table)

        print(f"[I] There are {len(self.running_job_queue)} jobs in running status...")
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")
        
        return ended_job_id_list
    
    def _attempt_profile_and_stop_profile_pending_jobs(self):
        """ Attempt to profile all in-profile jobs that have not been profiled, and stop profile-completed jobs. """
        
        for job in self.in_profile_job_queue:
            print(f"------> Job alias: {job.alias} | Profile time budget: {job.profile_time_budget}")
            
            if job.uuid in self.profiled_job_to_node_ids or (job.profile_time_budget is not None and job.profile_time_budget <= 0):
                # Is being profiled
                continue
            
            can_be_profiled = True
            # assert self.disable_single_device_profiler, f"Currently only supports disabling single-device profiler."

            to_profile_node_ids = {}
            # Select a feasible resource quota of all GPU types for profiling
            for gpu_type in self.resource_interface.get_gtype_num_table().keys():
                if self.required_gpu_num_for_profile < NODE_CAPACITY[gpu_type]:
                    print(f"[WARN] Current only supports profiling with GPU num >= {NODE_CAPACITY[gpu_type]} for {gpu_type}, " + 
                          f"will not occupy resources for profiling...")

                required_node_num = self.required_gpu_num_for_profile // NODE_CAPACITY[gpu_type]
                to_profile_node_ids[gpu_type] = []

                for node_id in self.resource_interface.node_pool:
                    node = self.resource_interface.node_pool[node_id]
                    if node.idle_gpu_num == node.capacity and node.gpu_type == gpu_type:
                        # Idle node, maybe allocated for profiling
                        assert node_id not in to_profile_node_ids[gpu_type]
                        to_profile_node_ids[gpu_type].append(node_id)

                    if len(to_profile_node_ids[gpu_type]) == required_node_num:
                        break
                
                if len(to_profile_node_ids[gpu_type]) < required_node_num:
                    # This node cannot be profiled now
                    can_be_profiled = False
                    break
            
            if not can_be_profiled:
                print(f"[I] Pending job '{job.alias}' (alias) cannot be profiled now due to " + 
                      "insufficient idle resources.")
                continue

            # print(f"\n\nJob {job.alias} can be profiled, occupied resources:")
            # for gpu_type in self.profiled_job_to_node_ids[job.uuid]:
            #     print(f"GPU type: {gpu_type} | Node num: {len(self.profiled_job_to_node_ids[job.uuid][gpu_type])}")
            # print("")

            # Occupy resources for profiling
            print(f"[I] Pending job '{job.alias}' (alias) is being profiled now...")
            self.profiled_job_to_node_ids[job.uuid] = to_profile_node_ids
            crs_table = self.resource_interface.get_crs_table()

            for gpu_type in self.profiled_job_to_node_ids[job.uuid]:
                for node_id in self.profiled_job_to_node_ids[job.uuid][gpu_type]:
                    # print(self.resource_interface.node_pool[node_id].alias)
                    for gpu_id in crs_table[node_id]:
                        # print(crs_table[node_id][gpu_id]["status"])
                        assert crs_table[node_id][gpu_id]["status"] == IDLE_STATUS or crs_table[node_id][gpu_id]["status"] is None
                        crs_table[node_id][gpu_id]["status"] = USED_STATUS
                        crs_table[node_id][gpu_id]["used_job_id"] = job.uuid
            
            self.resource_interface.apply_sched(crs_table)

    def schedule(self):
        """
        Periodically schedule to decide resource allocation and job placement. 
        Scheduling events: (1) Job arrival; (2) Job departure.
        Intuitively, We first perform Running Jobs Optimization driven by job departure, 
        then perform New Job(s) Arrival driven by job arrival. This can also avoid considering 
        upgrading GPU type in job downgrade search (in the process of New Jobs Arrival, since 
        intuitively we need to traverse all potential GPU type), which violates the greedy-based 
        search method inspired by the observation of the marginal gain.  
        """

        print(f"[I] Idle GPU num in cluster:", self.resource_interface.get_gtype_num_table(only_idle_gpus=True))
        
        # Update the remained iteration num of the running jobs and end
        self._update_and_check_end()
        makespan_list, jct_list, queue_time_list = self._get_end_job_metrics()
        before_resched_crs_table = self.resource_interface.get_crs_table()
        
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization
            _ = self._optimize_running_jobs(
                is_need_upgrade=(len(self.submit_init_job_queue) == 0 and len(self.in_profile_job_queue) == 0))
            # _ = self._optimize_running_jobs(is_need_upgrade=True)
            self.ended_job_queue.clear()
        
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._fit_new_jobs()
            self.submit_init_job_queue.clear()

        if self.prepend_profile_overhead:
            # Attempt to profile pending jobs (with resources accupied) that have not been profiled
            self._attempt_profile_and_stop_profile_pending_jobs()
        
        self._update_timer()

        self._update_queue_time_table()
        self._update_executed_time_table()
        if self.enable_ddl:
            self._drop_all_timeout_jobs_in_pending_queue()

        after_resched_crs_table = self.resource_interface.get_crs_table()

        # print(f"\n\nCluster resource change:")
        # self._parse_crs_table_change(before_resched_crs_table, after_resched_crs_table)

        new_resched_jid_list = self._parse_crs_table_change(before_resched_crs_table, 
                                                            after_resched_crs_table,
                                                            quiet_mode=True)
        
        for _job_id in new_resched_jid_list:
            self.resched_num_table = dict_counter_add(self.resched_num_table, _job_id, 1)
        
        # crs_table = self.resource_interface.get_crs_table()
        # for node_id in crs_table:
        #     node = self.resource_interface.node_pool[node_id]
        #     if node.idle_gpu_num == node.capacity:
        #         continue

        #     job_names = []
        #     for gpu_id in crs_table[node_id]:
        #         if node.gpu_to_job_table[gpu_id]["used"] == EMPTY_JOB_ID:
        #             continue

        #         job = self.get_job_by_uuid(node.gpu_to_job_table[gpu_id]["used"])
        #         if job.alias not in job_names:
        #             job_names.append(job.alias)

        #     print(f"Node: {node.alias} | Jobs: {job_names}")
        
        return self._get_cluster_perf(makespan_list, jct_list, queue_time_list,
                                      new_resched_jid_list)

    def runtime_schedule(self):
        """ Schedule function for Crius runtime. """
        ended_job_info_table = dict()
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization (we record the ended job id list to update the 
            # crius runtime after the this job is removed)
            ended_job_id_list = self._optimize_running_jobs(is_need_upgrade=
                                                            (len(self.submit_init_job_queue) == 0))
            for _job_id in ended_job_id_list:
                ended_job_info_table[_job_id] = self.get_job_by_uuid(_job_id).alias
            self.ended_job_queue.clear()
    
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        return ended_job_info_table
