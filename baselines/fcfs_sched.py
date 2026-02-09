#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" The implementation of vanilla First-Come-First-Served (FCFS) policy. """

from typing import Sequence, Any, List

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resources.hardware_specs import NODE_CAPACITY
from job.job import Job
from scheduler import (
    Scheduler, AblationOptions
)
from utils import dict_counter_add, deepcopy
from macro.macro_def import (
    JOB_RUNNING_STATUS, JOB_PENDING_STATUS, IDLE_STATUS, USED_STATUS, 
    EMPTY_JOB_ID, INFEASIBLE_THR)


class FCFSSched(Scheduler):
    """ 
    The class of the FCFS scheduler, in which we implement new jobs arrival 
    and pending jobs restart. 
    """
    def __init__(self, node_pool: dict, supported_gpu_types: List[str], is_allow_relaxed: bool = False,
                 enable_alpa: bool = False, is_runtime: bool = False, 
                 verbose: bool = False, dummy_test: bool = False,
                 sched_with_opt: bool = False):
        # Init base class
        super().__init__(node_pool, supported_gpu_types, AblationOptions(force_dp=True), 
                         is_runtime, verbose, dummy_test, sched_with_opt)
        # Whether to allow relaxing locality
        self.is_allow_relaxed = is_allow_relaxed
        self.prepend_profile_overhead = os.environ.get("CRIUS_PREPEND_PROFILE_OVERHEAD", "0") == "1"
        self.per_job_profile_overhead = 120   # Assume each job can be allocated 16 GPUs at most
    
    ######################################
    #    Job Arrival Related Functions   #
    ######################################

    def _place_with_best_locality(self, target_job: Job, crs_table: dict, 
                                  force_opt: bool = False):
        """ Place the new job with the best locality. """
        (gpu_type, gpu_num, 
         best_locality, _) = self._get_job_rt_stat(target_job.uuid, crs_table, 
                                                   overwrite_job=target_job)
        
        if not self._is_alloc_feasible(target_job, gpu_type, 
                                       best_locality, force_opt):
            # Cannot placed with best locality
            return False
        
        node_id_list =  list()
        for _nid in crs_table:
            if self.resource_interface.node_pool[_nid].gpu_type != gpu_type:
                # Skip since can only place with the same GPU type
                continue
            if self._get_bubble_num(crs_table[_nid]) >= best_locality[0]:
                # This node can contain a per-node quota
                node_id_list.append(_nid)
        
        if len(node_id_list) >= len(best_locality):
            # Sufficient for placement
            print(f"[I] Job '{target_job.alias}' (alias) has been running with best locality...")
            # Sort node id list to place in a scatter style
            node_id_list = sorted(node_id_list, 
                                  key=lambda x: self._get_bubble_num(crs_table[x]), 
                                  reverse=True)
            print(" - The nodes list is:", [
                self.resource_interface.node_pool[_nid].alias for _nid in node_id_list[:len(best_locality)]
            ])
            # Place on these nodes
            is_modified = self._clear_placement(target_job.uuid, crs_table)
            assert not is_modified, \
                f"In FCFS baseline, the to-allocated job should not appear in crs_table."
            is_placed = self._place_job(target_job, gpu_num, 
                                        node_id_list[:len(best_locality)], crs_table)
            assert is_placed
            # Placed
            return True
        # Cannot place with best locality
        return False
    
    def _place_with_relaxed_locality(self, target_job: Job, crs_table: dict,
                                     force_opt: bool = False):
        """ Place the new job with the relaxed locality. """
        (gpu_type, gpu_num, 
         locality, _) = self._get_job_rt_stat(target_job.uuid, crs_table, 
                                              overwrite_job=target_job)
        
        if locality[0] == 1:
            # Cannot further relaxed
            return False
        
        while locality[0] > 1:
            # Relax the locality
            per_node_quota = locality[0] // 2
            locality = [per_node_quota for _ in range(gpu_num // per_node_quota)]
            
            if not self._is_alloc_feasible(target_job, gpu_type, 
                                           locality, force_opt):
                # Cannot placed with locality
                return False

            node_id_list =  list()
            for _nid in crs_table:
                if self.resource_interface.node_pool[_nid].gpu_type != gpu_type:
                    # Can only place with the same GPU type
                    continue
                if self._get_bubble_num(crs_table[_nid]) >= locality[0]:
                    # This node can contain a per-node quota
                    node_id_list.append(_nid)

            if len(node_id_list) >= len(locality):
                # Sufficient for placement
                print(f"[I] Job '{target_job.alias}' (alias) has been running with relaxed locality...")
                # Sort node id list to place in a scatter style
                node_id_list = sorted(node_id_list, 
                                      key=lambda x: self._get_bubble_num(crs_table[x]), 
                                      reverse=True)
                # Place on these nodes
                is_modified = self._clear_placement(target_job.uuid, crs_table)
                assert not is_modified, \
                    f"In FCFS baseline, the to-allocated job should not appear in crs_table."
                # Place the entire job on the dst nodes
                is_placed = self._place_job(target_job, gpu_num, 
                                            node_id_list[:len(locality)], crs_table)
                assert is_placed
                # Placed
                return True
        # Cannot place with relaxed locality
        return False

    ######################################
    #      Restart Related Functions     #
    ###################################### 

    def _fcfs_pending_jobs_restart_trial(self):
        """ Try restart the pending jobs. """
        if self.verbose:
            print("")
        print("[I] Begin Pending Jobs Restart Trial...")
        if len(self.pending_job_queue) == 0:
            print("[I][REST] No pending job exists.")
            return 
        
        restarted_job_queue, _blocked_job_gpu_type = list(), list()
        for _job in self.pending_job_queue:
            if _job.resource_quota.gpu_type in _blocked_job_gpu_type:
                # If there exists one blocked job with the same gpu type, 
                # skip restarting trial of this job since this baseline
                # disables opportunistic execution.
                continue

            if (
                    self.prepend_profile_overhead and
                    (self.global_timer - _job.sub_time) < self.per_job_profile_overhead
                ):
                    # This job is still in profiling
                    continue

            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            # For pending job restart in FCFS, we exploit optimal parallelism
            # since it cannot modify gpu num or type.
            is_fsb = self._fcfs_fit_one_job(_job, crs_table, force_opt=True)

            if is_fsb:
                print(f"[I] Pending job '{_job.alias}' (alias) has been restarted...")
                # Apply the scheduling decision
                self.resource_interface.apply_sched(crs_table)
                # Restart job
                _job.update_status(JOB_RUNNING_STATUS)
                restarted_job_queue.append(_job)
                self.running_job_queue.append(_job)
                # Update the resource status of running jobs
                self._update_run_jobs(crs_table)
            else:
                print(f"[I] Pending job '{_job.alias}' (alias) cannot be restarted...")
                assert _job.resource_quota.gpu_type not in _blocked_job_gpu_type
                # Block this job
                _blocked_job_gpu_type.append(_job.resource_quota.gpu_type)

        for _job in restarted_job_queue:
            # Remove restarted jobs from the pending queue
            self.pending_job_queue.remove(_job)
    
    ######################################
    #        Top-level Functions         #
    ######################################
    
    def _fcfs_fit_one_job(self, job: Job, crs_table: dict, force_opt: bool = False):
        """ 
        Support placing jobs with best locality or relaxed locality (if allowed). 
        """
        is_fsb = self._place_with_best_locality(job, crs_table, force_opt)
        
        if not is_fsb and self.is_allow_relaxed:
            # Try to place with relaxed locality
            is_fsb = self._place_with_relaxed_locality(job, crs_table, force_opt)
        
        return is_fsb
    
    def _fcfs_fit_new_jobs(self):
        """ In a First-Come-First-Served (FCFS) style. """
        print("")
        print("#################################################")
        print("#         FCFS-Style New Job(s) Arrival         #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.submit_init_job_queue)} new " + 
              f"jobs to be fitted...")

        # Sort jobs by init priority
        self.submit_init_job_queue = self._sort_jobs(self.submit_init_job_queue)

        for _job in self.submit_init_job_queue:
            if (
                self.prepend_profile_overhead and
                len(self.running_job_queue) > 0   # Must be some jobs in running status
            ):
                print(f"[I] Job '{_job.alias}' (alias) has entered profiling.")
                _job.update_status(JOB_PENDING_STATUS)
                self.pending_job_queue.append(_job)
                continue
            
            # If this job cannot be satisfied if with optimal parallelism and the best locality, 
            # drop it since cannot be scaling
            _best_locality = self._get_best_locality(_job.resource_quota.gpu_num, 
                                                     _job.resource_quota.gpu_type)
            _force_opt = not self.is_runtime
            if not self._is_alloc_feasible(_job, _job.resource_quota.gpu_type, 
                                           _best_locality, _force_opt):
                # Drop this job
                print(f"[I] Since job {_job.alias} cannot be placed with best " + 
                      f"locality {_best_locality} and cannot be scaled, drop it...")
                continue
            
            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            # For new jobs arrival in FCFS, we exploit optimal parallelism
            # since it cannot modify gpu num or type.
            is_fsb = self._fcfs_fit_one_job(_job, crs_table, force_opt=True)
            
            if (is_fsb and not self._is_exist_pend_job_same_gpu_type(_job.resource_quota.gpu_type)):
                # No need to pend this job, directly apply the scheduling decision 
                # and deploy on real resources.
                self.resource_interface.apply_sched(crs_table)
                # Update job status
                _job.update_status(JOB_RUNNING_STATUS)
                # Add to the running job queue in the scheduler
                self.running_job_queue.append(_job)
                # Update the allocated resources of all running jobs.
                self._update_run_jobs(crs_table)
            else:
                print(f"[I] Job '{_job.alias}' (alias) has been pending as a vanilla pending job.")
                # No feasible plan found, need to pend this job
                _job.update_status(JOB_PENDING_STATUS)
                # Add to the pending job queue in the scheduler
                self.pending_job_queue.append(_job)
        
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")
    
    def _fcfs_optimize_running_jobs(self):
        """ Pending jobs restart trial. """
        print("")
        print("#################################################")
        print("#      FCFS-Style Running Jobs Optimization     #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.ended_job_queue)} jobs are ended...")
        
        # Step 1. Release all related resources of ended jobs and update decision queue
        ended_job_id_list = self._release_resources_and_update_decision_queue()

        # Step 2. FCFS Pending Jobs Restart Trial
        self._fcfs_pending_jobs_restart_trial()

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

        if len(self.ended_job_queue) > 0:
            # Running jobs optimization
            self._fcfs_optimize_running_jobs()
            self.ended_job_queue.clear()

        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._fcfs_fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        self._update_timer()

        self._update_queue_time_table()
        self._update_executed_time_table()

        after_resched_crs_table = self.resource_interface.get_crs_table()
        new_resched_jid_list = self._parse_crs_table_change(before_resched_crs_table, 
                                                            after_resched_crs_table,
                                                            quiet_mode=True)
        
        for _job_id in new_resched_jid_list:
            assert _job_id not in self.resched_num_table, \
                f"{self.get_job_by_uuid(_job_id).alias} should not be rescheduled."

            self.resched_num_table = dict_counter_add(self.resched_num_table, _job_id, 1)

        return self._get_cluster_perf(makespan_list, jct_list, queue_time_list, 
                                      new_resched_jid_list)

    def runtime_schedule(self):
        """ Schedule function for Crius runtime. """
        ended_job_info_table = dict()
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization (we record the ended job id list to update the 
            # crius runtime after the this job is removed)
            ended_job_id_list = self._fcfs_optimize_running_jobs()
            for _job_id in ended_job_id_list:
                ended_job_info_table[_job_id] = self.get_job_by_uuid(_job_id).alias
            self.ended_job_queue.clear()
        
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._fcfs_fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        return ended_job_info_table
