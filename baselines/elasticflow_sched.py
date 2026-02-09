#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
The implementation of ElasticFlow: An Elastic Serverless Training Platform 
for Distributed Deep Learning. 
Referring to the "Improvement in CE" of Section 6.4 in the paper, we set 
the deadline to be long enough.
Ref: https://dl.acm.org/doi/10.1145/3575693.3575721
"""

from typing import Sequence, List

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resources.hardware_specs import NODE_CAPACITY
from job.job import Job
from db_querier import QueryConfigs
from scheduler import (
    Scheduler, AblationOptions)
from utils import dict_counter_add, deepcopy
from macro.macro_def import (
    INFEASIBLE_THR, MAX_SUPPORTED_GPU_NUM, JOB_RUNNING_STATUS, 
    JOB_PENDING_STATUS, PREC, INFEASIBLE_ITER_TIME, RESCHED_OVERHEAD_WITHOUT_PRUNE,
    MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE, SCHEDULING_INTERVAL)


# Job num to be rescheduled one job-allocating round
MAX_RESCHED_JOB_NUM = 1000
# Max job num to remained in pending status until disabling preemption.
MAX_PEND_JOB_NUM_PREEMPTION = 100


class ElasticFlowSched(Scheduler):
    """ 
    The class of the ElasticFlow scheduler, in which we implement admission 
    control and greedy resource allocation. 
    """
    def __init__(self, node_pool: dict, supported_gpu_types: List[str], enable_ddl: bool = False, enable_alpa: bool = False, 
                 is_runtime: bool = False, verbose: bool = False, 
                 dummy_test: bool = False, sched_with_opt: bool = False):
        # Init base class
        super().__init__(node_pool, supported_gpu_types, AblationOptions(force_dp=True, enable_ddl=enable_ddl), 
                         is_runtime, verbose, dummy_test, sched_with_opt)
        # # Job gpu num in the last scheduling round
        # self.prev_gpu_num_table = None
        # Global timer of the crius runtime
        self.runtime_global_timer = None
        self.all_profile_overhead = 600     # Referred to Elasticflow's paper, 10min
    
    ######################################
    #     Profiling Related Functions    #
    ######################################
    
    def _is_ddl_satisfied(self, target_job: Job, gpu_num: int):
        """ Whether the deadline can be satisfied with the given GPU num. """
        # Query database
        gpu_type = target_job.resource_quota.gpu_type
        best_locality = self._get_best_locality(gpu_num, target_job.resource_quota.gpu_type)
        iter_time = self._get_iter_time(target_job.model_name, target_job.batch_size, gpu_type,
                                        best_locality)
        is_fsb = (iter_time < INFEASIBLE_ITER_TIME)
        resched_overhead = min(RESCHED_OVERHEAD_WITHOUT_PRUNE * target_job.resource_quota.gpu_num, 
                               MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE)

        if not self.is_runtime:
            # Check whether the deadline can be satisfied with the 
            # profiled throughput and the remained iteration num.
            return True if (is_fsb and (self.global_timer + resched_overhead + 
                                        + SCHEDULING_INTERVAL +
                         int(target_job.remained_iter_num * iter_time)
                         ) <= target_job.deadline) else False
        else:
            # Use the global timer of the Crius runtime to check 
            # whether the deadline can be satisfied.
            raise NotImplementedError()
            # return True if is_fsb and (self.runtime_global_timer 
            #                            + int(target_job.remained_iter_num * iter_time) 
            #                            <= target_job.deadline) else False
    
    def _get_min_satisfactory_share(self, target_job: Job):
        """ Get the minimum satisfactory share of the target job. """
        gpu_num = 1
        is_fsb = self._is_ddl_satisfied(target_job, gpu_num)
        if is_fsb:
            return gpu_num
        
        while not is_fsb and gpu_num < MAX_SUPPORTED_GPU_NUM:
            # Recursively doubling the GPU num
            gpu_num = gpu_num * 2
            is_fsb = self._is_ddl_satisfied(target_job, gpu_num)
        
        return gpu_num if is_fsb else None

    ######################################
    #    Job Arrival Related Functions   #
    ######################################

    def _sort_jobs_by_deadline(self, queue: Sequence[Job], gpu_type: str):
        """ Sort all jobs by the deadline. """
        # Filter with the same GPU type
        _queue = list()
        for job in queue:
            if job.resource_quota.gpu_type == gpu_type:
                _queue.append(job)
        # Sort by deadline in increasing order
        return [_job.uuid for _job in sorted(_queue, key=lambda x: x.deadline)]
    
    def _ef_progressive_filling_pass(self, target_job: Job, avail_gpu_num: int):
        """ The elasticflow-style progressive filling pass. """
        # Get the minimum satisfactory share of the job
        min_gpu_num = self._get_min_satisfactory_share(target_job)
        if min_gpu_num is None:
            # The ddl can not be satisfied even with the max allowed GPU num
            return False, None
        # Return
        return (True, avail_gpu_num - min_gpu_num) \
            if avail_gpu_num >= min_gpu_num else (False, None)

    def _ef_admission_control_pass(self, target_job: Job):
        """ 
        The elasticflow-style admission control pass. Since ElasticFlow only 
        considers the homogenous GPUs, the pass only happends among jobs with 
        the same GPU type.
        """
        gpu_type = target_job.resource_quota.gpu_type

        # Step 1. Filter and sort all jobs with the same GPU type
        job_id_list = self._sort_jobs_by_deadline(queue=self.running_job_queue + [target_job], 
                                                  gpu_type=gpu_type)

        # Step 2. Count the total GPU num of the target type
        total_gpu_num = 0
        for _nid in self.resource_interface.node_pool:
            if (self.resource_interface.node_pool[_nid].gpu_type 
                == target_job.resource_quota.gpu_type):
                total_gpu_num += self.resource_interface.node_pool[_nid].capacity
        
        # Step 3. Traverse each job to perform elasticflow-style progressive filling
        for _job_id in job_id_list:
            _job = self.get_job_by_uuid(_job_id)
            # Must be the same GPU type with the target job since filtered above
            is_fsb, total_gpu_num = self._ef_progressive_filling_pass(_job, total_gpu_num)
            # Check whether the minimum satisfactory share of this job can be satisfied
            if not is_fsb:
                # There exists at least a job that can not be satisfied
                return False
        # All jobs are satisfied
        return True

    def _ef_resource_allocation_pass(self, target_job: Job, crs_table: dict):
        """ Elasticflow-style resource allocation. """
        # Job list
        job_list = [target_job]
        for _job in self.running_job_queue:
            if _job.resource_quota.gpu_type == target_job.resource_quota.gpu_type:
                job_list.append(_job)
        
        # Total GPU num & node uuid list
        total_gpu_num = 0
        node_id_list = list()
        for _nid in self.resource_interface.node_pool:
            if (self.resource_interface.node_pool[_nid].gpu_type 
                == target_job.resource_quota.gpu_type):
                total_gpu_num += self.resource_interface.node_pool[_nid].capacity
                node_id_list.append(_nid)

        # Step 1. Allocate the minimum satisfactory share for the candidate jobs
        gpu_num_table = dict()
        for _job in job_list:
            assert _job.uuid not in gpu_num_table
            gpu_num_table[_job.uuid] = self._get_min_satisfactory_share(_job)
            total_gpu_num -= gpu_num_table[_job.uuid]
            assert total_gpu_num >= 0
        
        # Step 2. Allocate the rest GPUs based on maximizing cluster throughput 
        #         (which is the target of elasticflow)
        is_fsb_plan_found = True
        while is_fsb_plan_found and total_gpu_num > 0:
            is_fsb_plan_found = False
            # Search for the best plan with the maximized throughput
            _max_gain = INFEASIBLE_THR
            alloc_job_id = None

            # Traverse the candidate jobs
            for _job in job_list:
                if (gpu_num_table[_job.uuid] > total_gpu_num or
                     gpu_num_table[_job.uuid] >= MAX_SUPPORTED_GPU_NUM):
                    # The remained GPU num cannot satisfy doubling the GPU num of this job
                    continue
                _new_best_locality = self._get_best_locality((gpu_num_table[_job.uuid] * 2), 
                                                             _job.resource_quota.gpu_type)
                _gain = self._get_delta_thr(_job, _job.resource_quota.gpu_type, 
                                            (gpu_num_table[_job.uuid] * 2),
                                            _new_best_locality, 
                                            _job.resource_quota.gpu_type,
                                            gpu_num_table[_job.uuid],
                                            self._get_best_locality(gpu_num_table[_job.uuid], 
                                                                    _job.resource_quota.gpu_type))
                if _gain > _max_gain:
                    _max_gain = _gain
                    alloc_job_id = _job.uuid
                
                # if _gain > _max_gain and len(_new_best_locality) == 1:
                #     # FIXME: In runtime, due to network legacy, we forbidden cross-nodes placement in baselines
                #     _max_gain = _gain
                #     alloc_job_id = _job.uuid

            # Apply allocating decision for one step
            if alloc_job_id:
                is_fsb_plan_found = True
                # Doubling the GPU num
                total_gpu_num -= gpu_num_table[alloc_job_id] 
                gpu_num_table[alloc_job_id] *= 2
        
        # Sort the job list based on their allocated GPU num in decreasing order
        job_list = sorted(job_list, key=lambda x: gpu_num_table[x.uuid], reverse=True)
        
        # if False:
            # # Check whether too many jobs should be rescheduled
            # changed_job_num = 0
            # for _jid in job_list:
            #     if (self.prev_gpu_num_table is not None and 
            #         _jid in self.prev_gpu_num_table and 
            #         self.prev_gpu_num_table[_jid] != gpu_num_table[_jid]):
            #         changed_job_num += 1

            # if changed_job_num > MAX_RESCHED_JOB_NUM:
            #     # Exceed max reschedulable job num
            #     return False
            
            # # Update prev gpu num table
            # self.prev_gpu_num_table = gpu_num_table

        # Step 3. Clear previous placement
        for _job in self.running_job_queue:
            if _job.resource_quota.gpu_type == target_job.resource_quota.gpu_type:
                # Clear its placement 
                is_modified = self._clear_placement(_job.uuid, crs_table)
                assert is_modified
        
        # Step 4. Dispatch GPUs for the candidate jobs based on their allocated GPU num 
        #         with buddy allocation and migration
        _placed_job_id_list = list()
        
        # - First, handle with gpu num > node capacity case
        for _job in job_list:
            gpu_type = _job.resource_quota.gpu_type
            if gpu_num_table[_job.uuid] <= NODE_CAPACITY[gpu_type]:
                break
            
            # Get fully idle nodes to place
            _needed_node_num = gpu_num_table[_job.uuid] // NODE_CAPACITY[gpu_type]
            _node_id_list = node_id_list[:_needed_node_num]
            # Place
            print(f"[I][RALC] Job '{_job.alias}' (alias) has been " + 
                  f"allocated {gpu_num_table[_job.uuid]} {gpu_type} GPUs on the following nodes:")
            print(" - Allocated node list:", [
                self.resource_interface.node_pool[_uuid].alias for _uuid in _node_id_list
            ])
            is_placed = self._place_job(_job, gpu_num_table[_job.uuid], _node_id_list, crs_table)
            assert is_placed
            # Record
            _placed_job_id_list.append(_job.uuid)
            # Remove the nodes
            for _node_id in _node_id_list:
                node_id_list.remove(_node_id)
        
        # - Then, place the rest jobs
        for _node_id in node_id_list:
            node_alias = self.resource_interface.node_pool[_node_id].alias
            # Traverse the job id list to place with in a packing manner
            for _job in job_list:
                if self._get_bubble_num(crs_table[_node_id]) == 0:
                    # Fully occupied
                    break
                
                if _job.uuid in _placed_job_id_list:
                    # This job is already placed
                    continue
                
                if gpu_num_table[_job.uuid] <= self._get_bubble_num(crs_table[_node_id]):
                    # Place
                    print(f"[I][RALC] Job '{_job.alias}' (alias) has been " + 
                          f"allocated {gpu_num_table[_job.uuid]} {gpu_type} GPUs " + 
                          f"on node '{node_alias}':")
                    is_placed = self._place_job(_job, gpu_num_table[_job.uuid], 
                                                [_node_id], crs_table)
                    assert is_placed
                    _placed_job_id_list.append(_job.uuid)
        assert len(_placed_job_id_list) == len(job_list)

        return True
    
    ######################################
    #      Restart Related Functions     #
    ###################################### 

    def _ef_pending_jobs_restart_trial(self):
        """ Try restart the pending jobs. """
        if self.verbose:
            print("")
        print("[I] Begin Pending Jobs Restart Trial...")
        if len(self.pending_job_queue) == 0:
            print("[I][REST] No pending job exists.")
            return
        
        _restarted_job_queue, _blocked_job_gpu_type = list(), list()
        for job in self.pending_job_queue:
            if job.uuid in self.timeout_job_id_list:
                # Timeout job
                continue
            
            if job.resource_quota.gpu_type in _blocked_job_gpu_type:
                # This gpu type have been blocked without opportunistic
                continue
            
            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            is_fsb = self._ef_fit_one_job(job, crs_table)

            if is_fsb:
                print(f"[I] Pending job '{job.alias}' (alias) has been restarted...")
                # Apply the scheduling decision
                self.resource_interface.apply_sched(crs_table)
                # Update job status
                job.update_status(JOB_RUNNING_STATUS)
                # Record
                _restarted_job_queue.append(job)
                # Add to the running job queue in the scheduler
                self.running_job_queue.append(job)
                # Update the allocated resources of all running jobs.
                self._update_run_jobs(crs_table)
            else:
                print(f"[I] Pending job '{job.alias}' (alias) cannot be restarted...")
                # Block this GPU type
                assert job.resource_quota.gpu_type not in _blocked_job_gpu_type
                if not os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                    # We allow ElasticFlow to restart in random order
                    _blocked_job_gpu_type.append(job.resource_quota.gpu_type)

        # Remove all restarted jobs from the pending queue
        for _job in _restarted_job_queue:
            self.pending_job_queue.remove(_job)

    ######################################
    #        Top-level Functions         #
    ######################################

    def _ef_fit_one_job(self, job: Job, crs_table: dict):
        """ Perform admission control pass and resource allocation pass. """
        # Admission control pass
        if not self._ef_admission_control_pass(job):
            # Can not be admitted, drop
            print(f"[I] Job '{job.alias}' (alias) can not be admitted, " + 
                  f"drop or (keep) pending.")
            return False
        
        print(f"[I] Job '{job.alias}' (alias) has been admitted.")
        # Resource allocation pass
        is_fsb = self._ef_resource_allocation_pass(job, crs_table)

        return is_fsb
    
    def _ef_fit_new_jobs(self):
        """ In a ElasticFlow style. """
        print("")
        print("#################################################")
        print("#     ElasticFlow-Style New Job(s) Arrival      #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.submit_init_job_queue)} new jobs to be fitted...")

        # Sort jobs by init priority
        self.submit_init_job_queue = self._sort_jobs(self.submit_init_job_queue)
        
        for job in self.submit_init_job_queue:       
            # # If this job cannot be satisfied if with the best locality, 
            # # drop it since cannot be scaling
            # _best_locality = self._get_best_locality(job.resource_quota.gpu_num, 
            #                                          job.resource_quota.gpu_type)
            # if not self._is_alloc_feasible(job, job.resource_quota.gpu_type, 
            #                                _best_locality):
            #     # Drop this job
            #     continue

            crs_table = self.resource_interface.get_crs_table()

            # Add job profiling overhead only in revision mode
            if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                self.queue_time_table[job.uuid] = self.all_profile_overhead  
            
            # Fit one job
            is_fsb = self._ef_fit_one_job(job, crs_table)
            
            pend_job_num = 0
            for _job in self.pending_job_queue:
                if _job.resource_quota.gpu_type == job.resource_quota.gpu_type:
                    pend_job_num += 1

            if is_fsb:
            # if is_fsb and pend_job_num < MAX_PEND_JOB_NUM_PREEMPTION:
                # Apply the scheduling decision
                self.resource_interface.apply_sched(crs_table)
                # Update job status
                job.update_status(JOB_RUNNING_STATUS)
                # Add to the running job queue in the scheduler
                self.running_job_queue.append(job)
                # Update the allocated resources of all running jobs
                self._update_run_jobs(crs_table)
            else:
                print(f"[I] Job '{job.alias}' (alias) has been pending as a vanilla pending job.")
                # No feasible plan found, need to pend this job
                job.update_status(JOB_PENDING_STATUS)
                # Add to the pending job queue in the scheduler
                self.pending_job_queue.append(job)
        
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")
    
    def _ef_optimize_running_jobs(self):
        """ Pending jobs restart trial. """
        print("")
        print("#################################################")
        print("#  ElasticFlow-Style Running Jobs Optimization  #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.ended_job_queue)} jobs are ended...")
        
        # Step 1. Release all related resources of ended jobs and update decision queue
        ended_job_id_list = self._release_resources_and_update_decision_queue()

        # Step 2. ElasticFlow Pending Jobs Restart Trial
        self._ef_pending_jobs_restart_trial()

        return ended_job_id_list
    
    def schedule(self):
        """
        Periodically schedule to decide resource allocation and job placement. 
        Scheduling events: (1) Job arrival; (2) Job departure.
        """

        print(f"[I] Idle GPU num in cluster:", self.resource_interface.get_gtype_num_table(only_idle_gpus=True))

        # Update the remained iteration num of the running jobs and end
        self._update_and_check_end()
        makespan_list, jct_list, queue_time_list = self._get_end_job_metrics()
        before_resched_crs_table = self.resource_interface.get_crs_table()

        # Drop all timeout jobs
        self._drop_all_timeout_jobs_in_pending_queue()
        
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization
            self._ef_optimize_running_jobs()
            self.ended_job_queue.clear()

        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._ef_fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        self._update_timer()

        self._update_queue_time_table()
        self._update_executed_time_table()

        after_resched_crs_table = self.resource_interface.get_crs_table()
        new_resched_jid_list = self._parse_crs_table_change(before_resched_crs_table, 
                                                            after_resched_crs_table,
                                                            quiet_mode=True)
        
        for _job_id in new_resched_jid_list:
            self.resched_num_table = dict_counter_add(self.resched_num_table, _job_id, 1)

        return self._get_cluster_perf(makespan_list, jct_list, queue_time_list, 
                                      new_resched_jid_list)

    def runtime_schedule(self, runtime_global_timer: int = None):
        """ Schedule function for Crius runtime. """
        # Update runtime global timer
        self.runtime_global_timer = runtime_global_timer
        
        ended_job_info_table = dict()
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization (we record the ended job id list to 
            # update the crius runtime after the this job is removed)
            ended_job_id_list = self._ef_optimize_running_jobs()
            for _job_id in ended_job_id_list:
                ended_job_info_table[_job_id] = self.get_job_by_uuid(_job_id).alias
            self.ended_job_queue.clear()
        
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._ef_fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        return ended_job_info_table
