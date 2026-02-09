#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
The implementation of Gavel (Heterogeneity-Aware Cluster Scheduling Policies for 
Deep Learning Workloads). We focus on maximize throughput policy in Gavel, which is 
not detailed described in paper, but published in the code repo: 
https://github.com/stanford-futuredata/gavel/blob/master/scheduler/policies/max_sum_throughput.py

Referring to this repo, we implement Gavel baseline with the objective function of:
                                max( sum_i( sum_j ( thr_{i, j}) ) )

where i denotes the index of job uuid and j denotes the index of GPU type.
It should be noted that Gavel allows GPU type change among varying GPU types for 
training jobs, but fixs GPU num of each job. Also, job colocation within one GPU 
is disabled in our scenarios as the GPU utilization of LM is high enough.
"""

from typing import (Sequence, Any, List)
import traceback
import numpy as np
import cvxpy as cp

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
    JOB_RUNNING_STATUS, JOB_PENDING_STATUS, MAX_SUPPORTED_GPU_NUM, 
    INFEASIBLE_THR, MAX_THR)


# Job num to be migrated one job-allocating round
MAX_MGRT_JOB_NUM = 3
# MAX_MGRT_JOB_NUM = 10000

# TODO(chunyu): Add type switching penalty in Gavel.


class GavelSched(Scheduler):
    """ 
    The class of the Gavel scheduler, in which we implement new jobs 
    arrival and pending jobs restart. 
    """
    def __init__(self, node_pool: dict, supported_gpu_types: List[str], enable_alpa: bool = False, 
                 is_runtime: bool = False, verbose: bool = False, 
                 dummy_test: bool = False, sched_with_opt: bool = False):
        # Init base class
        super().__init__(node_pool, supported_gpu_types, AblationOptions(force_dp=True), 
                         is_runtime, verbose, dummy_test, sched_with_opt)
        self.admitted_job_gn_table = dict()     # Job uuid -> gpu num
        self.prev_allocation_matrix = None      # Allocation matrix last job-alloc event
        self.gtype_num_table = self.resource_interface.get_gtype_num_table()
        self.gpu_type_list = list(self.gtype_num_table.keys())
        self.type_num = len(self.gpu_type_list)

        if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
            # Further tuning Sia for better performance in revision mode
            global MAX_MGRT_JOB_NUM
            MAX_MGRT_JOB_NUM = 10
    
    ######################################
    #   Optimization Related Functions   #
    ######################################

    def _optimize_global_allocation(self, thr_matrix: Sequence[Sequence[float]], 
                                    job_num_table: dict, 
                                    prev_alloc_matrix: Any = None):
        """ Optimize the global allocation of all admitted jobs. 
        
        The structure of the global allocation matrix is: 

        |  GPU type A  |  GPU type B  |  GPU type C  |
        |       1      |       0      |       0      |   Job A (2 GPUs)
        |       1      |       0      |       0      |   Job B (4 GPUs)
        |       0      |       1      |       0      |   Job C (4 GPUs)
        |       0      |       0      |       1      |   Job D (8 GPUs)

        In the example, job A and B are allocated with 2 and 4 A GPUs,
        job C is allocated with 4 B GPUs and job D is allocated with 
        8 C GPUs. Job placement follows the packing style.
        ----------------------------------------------------
        The imeplementation refers to ThroughputNormalizedByCostSumWithPerfSLOs 
        class in `[OFFICIAL REPO]/scheduler/policies/max_sum_throughput.py`.
        """
        # Job gpu num matrix
        _vec = [job_num_table[_key] for _key in job_num_table]
        job_gn_matrix = np.tile(_vec, (self.type_num, 1)).T
        # Total gpu type num matrix
        total_gn_vector = np.array([
            self.gtype_num_table[_key] for _key in self.gtype_num_table])

        thr_matrix = np.array(thr_matrix)
        
        # Allocation matrix
        # `x[i, j]` denotes whether job j(axis=1) is allocated with gpu type i (axis=0).
        x = cp.Variable(thr_matrix.shape, boolean=True)
        # Objective function
        objective = cp.Maximize(cp.sum(cp.sum(cp.multiply(thr_matrix, x), 
                                                axis=1)))
        # Constraints
        constraints = [ x >= 0 ]
        # - Each admitted job must only be allocated on one gpu type
        constraints.append(cp.sum(x, axis=1) == 1)
        # - The allocated num of a gpu type cannot exceed total amount
        for _i in range(total_gn_vector.shape[0]):
            constraints.append(cp.sum(cp.multiply(x, job_gn_matrix), axis=0)[_i]
                                <= total_gn_vector[_i])
        # - The type-changed job amount should be no more than a threshold
        if prev_alloc_matrix is not None:
            constraints.append(cp.sum(cp.sum(cp.multiply(x, prev_alloc_matrix), axis=1)) 
                            >= len(list(job_num_table.keys())) - MAX_MGRT_JOB_NUM)

        # Problem
        cvxprob = cp.Problem(objective, constraints)
        # Solve
        _ = cvxprob.solve(solver=cp.CPLEX, verbose=False)

        print(f"[TMP] Optimization status: {cvxprob.status}")

        if cvxprob.status != "optimal":
            return None, None
        
        allocation = np.where(np.array(x.value) >= 0.5, 1, 0)

        # print(f"[TMP] Size of the allocation matrix (job_num, gpu_type_num): {allocation.shape}")
        
        cluster_thr = np.sum(np.sum(
            np.multiply(thr_matrix, allocation), axis=1
        ))

        return cluster_thr, allocation 

    ######################################
    #    Job Arrival Related Functions   #
    ######################################

    def _place_with_best_locality(self, target_job: Job, gpu_type: str, 
                                  crs_table: dict, force_opt: bool = False):
        """ Place the new job with the best locality. """
        gpu_num = target_job.resource_quota.gpu_num
        best_locality = self._get_best_locality(gpu_num, gpu_type, None, None)
        
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
                self.resource_interface.node_pool[_nid].alias 
                    for _nid in node_id_list[:len(best_locality)]
            ])
            # Place on these nodes
            is_modified = self._clear_placement(target_job.uuid, crs_table)
            assert not is_modified, \
                f"In Gavel baseline, the to-allocated job should not appear in crs_table."
            is_placed = self._place_job(target_job, gpu_num, 
                                        node_id_list[:len(best_locality)], crs_table)
            assert is_placed
            # Placed
            return True
        
        # Cannot place with best locality
        print(f"[I] Job '{target_job.alias}' (alias) cannot run with best locality...")

        return False
    
    def _place_with_relaxed_locality(self, target_job: Job, gpu_type: str, 
                                     crs_table: dict, force_opt: bool = False):
        """ Place the new job with the relaxed locality. """
        gpu_num = target_job.resource_quota.gpu_num
        locality = self._get_best_locality(gpu_num, gpu_type, None, None)
        
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
                print(" - The nodes list is:", [
                    self.resource_interface.node_pool[_nid].alias 
                        for _nid in node_id_list[:len(locality)]
                ])
                # Place on these nodes
                is_modified = self._clear_placement(target_job.uuid, crs_table)
                assert not is_modified, \
                    f"In Gavel baseline, the to-allocated job should not appear in crs_table."
                # Place the entire job on the dst nodes
                is_placed = self._place_job(target_job, gpu_num, 
                                            node_id_list[:len(locality)], crs_table)
                assert is_placed
                # Placed
                return True
        
        # Cannot place with relaxed locality
        print(f"[I] Job '{target_job.alias}' (alias) cannot run with relaxed locality...")

        return False

    def _gen_thr_matrix(self):
        """ Generate throughput matrix. """
        job_id_list = list(self.admitted_job_gn_table.keys())

        thr_matrix = []
        for _jid in job_id_list:
            job = self.get_job_by_uuid(_jid)
            gpu_num = job.resource_quota.gpu_num
            (model_name, param_num) = job.model_name.split("__")
            batch_size = job.batch_size

            thrs = []
            for _gtype in self.gtype_num_table:
                # For each GPU type, estimate throughput of each job with DP profiled data and the best locality
                best_locality = self._get_best_locality(gpu_num, _gtype)     
                num_hosts = len(best_locality)
                num_devices_per_host = best_locality[0]                          
                (_, thr, _) = self.db_querier.query_db(
                    QueryConfigs(
                        num_hosts, num_devices_per_host, _gtype, model_name, param_num, batch_size, 
                        only_opt=(self._sched_with_opt), only_dp=(not self._sched_with_opt),
                    )
                )
                thrs.append(thr)

            thr_matrix.append(thrs)

        return thr_matrix

    def _gavel_admission_control(self, target_job: Job):
        """ 
        The gavel-style admission control. We decide whether to admit the target
        job based on whether the cluster throughput will be degraded after adimtting 
        this job.
        """
        # Temporarily add target job
        self.admitted_job_gn_table[target_job.uuid] = target_job.resource_quota.gpu_num
        
        _prev_allocation_matrix = np.append(
            self.prev_allocation_matrix, [np.ones((self.type_num, ))], axis=0
        ) if self.prev_allocation_matrix is not None else None
        thr_matrix = self._gen_thr_matrix()

        # Optimize allocation
        try:
            cluster_thr, allocation = self._optimize_global_allocation(thr_matrix, self.admitted_job_gn_table,
                                                                    _prev_allocation_matrix)
        except Exception as e:
            print(f"[E] Meet optimization error, reject this job: {e}")
            del self.admitted_job_gn_table[target_job.uuid]
            traceback.print_exc()
            return False, None
        
        # print(np.array(thr_matrix))
        # print(list(self.admitted_job_gn_table.values()))
        # print(self.prev_allocation_matrix)
        # print(allocation)
        
        if not cluster_thr:
            # Currently this job cannot be admitted
            del self.admitted_job_gn_table[target_job.uuid]
            return False, None
        
        # This job can be admitted, parse gpu type modification
        mdf_gtype_table = dict()
        # Other jobs
        for _i in range(allocation.shape[0] - 1):
            prev_gtype_idx = list(self.prev_allocation_matrix[_i]).index(1) \
                                if self.prev_allocation_matrix is not None else None
            cur_gtype_idx = list(allocation[_i]).index(1)
            if prev_gtype_idx is None or prev_gtype_idx != cur_gtype_idx:
                job_id = list(self.admitted_job_gn_table.keys())[_i]
                prev_gtype = self.gpu_type_list[prev_gtype_idx] if prev_gtype_idx is not None else None
                cur_gtype = self.gpu_type_list[cur_gtype_idx]
                mdf_gtype_table[job_id] = (prev_gtype, cur_gtype)
        # Target job
        mdf_gtype_table[target_job.uuid] = (None, self.gpu_type_list[list(allocation[-1]).index(1)])

        # Update allocation matrix
        self.prev_allocation_matrix = allocation

        return True, mdf_gtype_table

    def _gavel_resource_allocation(self, target_job: Job, crs_table: dict, 
                                   mdf_gtype_table: dict, force_opt: bool = False):
        """ 
        Gavel-style resource allocation. Given the modifying info after admitting the new 
        job, gavel allocates resources in a packing style.
        """

        # Get all modified jobs and clear their previous placement
        job_id_list = list(mdf_gtype_table.keys())
        for _jid in job_id_list:
            is_modified = self._clear_placement(_jid, crs_table)
            assert is_modified or _jid == target_job.uuid
        
        # Traverse and place each job
        for _jid in job_id_list:
            job = self.get_job_by_uuid(_jid)
            new_gpu_type = mdf_gtype_table[_jid][1]
            # Try place with best locality
            is_fsb = self._place_with_best_locality(job, new_gpu_type, crs_table,
                                                    force_opt)
            
            if not is_fsb:
                # Try place with relaxed locality
                is_fsb = self._place_with_relaxed_locality(job, new_gpu_type, crs_table,
                                                           force_opt)
            
            if not is_fsb:
                # Cannot be placed
                return False, crs_table
        
        return True, crs_table

    ######################################
    #      Restart Related Functions     #
    ###################################### 

    def _gavel_pending_jobs_restart_trial(self):
        """ Try restart the pending jobs. """
        if self.verbose:
            print("")
        print("[I] Begin Pending Jobs Restart Trial...")
        if len(self.pending_job_queue) == 0:
            print("[I][REST] No pending job exists.")
            return

        _restarted_job_queue, _blocked_job_gpu_type = list(), list()
        for job in self.pending_job_queue:
            if job.resource_quota.gpu_type in _blocked_job_gpu_type:
                # This gpu type have been blocked without opportunistic
                continue
            
            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            is_feasible, crs_table = self._gavel_fit_one_job(job, crs_table)

            if is_feasible:
                print(f"[I] Pending job '{job.alias}' has been restarted...")
                # Apply the scheduling decision
                self.resource_interface.apply_sched(crs_table)
                job.update_status(JOB_RUNNING_STATUS)
                _restarted_job_queue.append(job)
                self.running_job_queue.append(job)
                self._update_run_jobs(crs_table)   # Update the allocated resources of all running jobs
            else:
                # Block this GPU type
                print(f"[I] Pending job '{job.alias}' (alias) cannot be restarted...")
                assert job.resource_quota.gpu_type not in _blocked_job_gpu_type
                if (not os.environ.get("CRIUS_REVISION_MODE", "false") == "true" and not self._sched_with_opt) or os.environ.get("CRIUS_HOMOGENEOUS_CLUSTER", "false") == "true":
                    # We allow Gavel to restart in random order
                    _blocked_job_gpu_type.append(job.resource_quota.gpu_type)

        # Remove all restarted jobs from the pending queue
        for _job in _restarted_job_queue:
            self.pending_job_queue.remove(_job)
    
    ######################################
    #        Top-level Functions         #
    ######################################

    def _gavel_fit_one_job(self, job: Job, crs_table: dict):
        """ Perform admission control and resource allocation. """
        # Admission control
        # Use dp result for optimization process
        is_admit, mdf_gtype_table = self._gavel_admission_control(job)

        if not is_admit:
            # Can not be admitted, need pending
            return False, crs_table

        print(f"[I] Job '{job.alias}' has been admitted.")
        
        # Resource allocation
        _crs_table = deepcopy(crs_table)
        # We modify crs_table in this function
        # Use optimal profiled data to allocate resources
        is_fsb, _crs_table = self._gavel_resource_allocation(job, _crs_table, 
                                                             mdf_gtype_table, force_opt=(self._sched_with_opt))

        if not is_fsb:
            # Since we use throughput of the best locality to solve the optimization
            # problem, in runtime resource allocation it may be downgraded to relaxed
            # locality, which leads to lower bandwidth. In this case, the allocation
            # is failed. We rejoin this target job into pending queue.
            # Another reason might be that although there are enough gpu num for the 
            # target gpu type, the fragmentation leads to uneven bubbles distribution.
            # In this case, the job cannot be allocated.
            print(f"[I] During resource allocation, job '{job.alias}' cannot be " + 
                  f"allocated, pending...")
            
            self.prev_allocation_matrix = np.delete(self.prev_allocation_matrix, -1, axis=0) \
                                            if self.prev_allocation_matrix is not None else None
            assert job.uuid in self.admitted_job_gn_table
            del self.admitted_job_gn_table[job.uuid]
            
            return False, crs_table
        
        # Print modification info
        print("[I] Modifying info:")
        for _jid in mdf_gtype_table:
            print(f"    - Job {self.get_job_by_uuid(_jid).alias}: {mdf_gtype_table[_jid][0]} " + 
                  f"-> {mdf_gtype_table[_jid][1]}")

        return True, _crs_table
    
    def _gavel_fit_new_jobs(self):
        """ 
        In a Maximizing Throughput (Gavel) style. 
        """
        print("")
        print("#################################################")
        print("#        Gavel-Style New Job(s) Arrival         #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.submit_init_job_queue)} new jobs to be fitted...")

        # Sort jobs by init priority
        self.submit_init_job_queue = self._sort_jobs(self.submit_init_job_queue)

        for job in self.submit_init_job_queue:
            # Check
            is_fsb = False
            for _gpu_type in self.gpu_type_list:
                _best_locality = self._get_best_locality(job.resource_quota.gpu_num, _gpu_type)
                _force_opt = not self.is_runtime
                if self._is_alloc_feasible(job, _gpu_type, _best_locality, _force_opt):
                    is_fsb = True
                    break
            
            if not is_fsb:
                # This job cannot be executed on any gpu type
                print(f"[I] Job '{job.alias}' cannot be executed on any gpu type, drop...")
                continue

            # if len(self.pending_job_queue) > 0:
            #     # Pending jobs with higher priority exist (any gpu type since type change is enabled)
            #     print(f"[I] Job '{job.alias}' has been pending as a vanilla pending job.")
            #     job.update_status(JOB_PENDING_STATUS)
            #     self.pending_job_queue.append(job)
            #     continue
            
            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            is_feasible, crs_table = self._gavel_fit_one_job(job, crs_table)

            if is_feasible:
                # Apply the scheduling decision
                self.resource_interface.apply_sched(crs_table)
                job.update_status(JOB_RUNNING_STATUS)
                self.running_job_queue.append(job)
                self._update_run_jobs(crs_table)   # Update the allocated resources of all running jobs
            else:
                # No feasible plan found, need to pend this job
                print(f"[I] Job '{job.alias}' has been pending as a vanilla pending job.")
                job.update_status(JOB_PENDING_STATUS)
                self.pending_job_queue.append(job)
        
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")
    
    def _gavel_optimize_running_jobs(self):
        """ Optimize running jobs. """
        print("")
        print("#################################################")
        print("#      Gavel-Style Running Jobs Optimization     #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.ended_job_queue)} jobs are ended...")
        
        # Release all related resources of ended jobs and update decision queue
        ended_job_id_list = self._release_resources_and_update_decision_queue()
        # Update admitted job table and prev allocation matrix
        for _jid in ended_job_id_list:
            # assert _jid in self.admitted_job_gn_table (Job can be deprecated before)
            if _jid in self.admitted_job_gn_table:
                idx = list(self.admitted_job_gn_table.keys()).index(_jid)
                del self.admitted_job_gn_table[_jid]
                self.prev_allocation_matrix = np.delete(self.prev_allocation_matrix, idx, axis=0)
        
        # Attempt restarting pending jobs
        self._gavel_pending_jobs_restart_trial()

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

        # Recheck that all admitted jobs are in running status in case some runtime errors
        to_del_job_ids = list()
        for _job_id in self.admitted_job_gn_table:
            job = self.get_job_by_uuid(_job_id)
            if not job or job.status != JOB_RUNNING_STATUS or job not in self.running_job_queue:
                to_del_job_ids.append(_job_id)
        for _job_id in to_del_job_ids:
            idx = list(self.admitted_job_gn_table.keys()).index(_job_id)
            del self.admitted_job_gn_table[_job_id]
            
            print(self.prev_allocation_matrix.shape)
            print(idx)
            
            self.prev_allocation_matrix = np.delete(self.prev_allocation_matrix, idx, axis=0)

        if len(self.ended_job_queue) > 0:
            # Running jobs optimization
            self._gavel_optimize_running_jobs()
            self.ended_job_queue.clear()

        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._gavel_fit_new_jobs()
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

    def runtime_schedule(self):
        """ Schedule function for Crius runtime. """
        ended_job_info_table = dict()
        
        # Recheck that all admitted jobs are in running status in case some runtime errors
        to_del_job_ids = list()
        for _job_id in self.admitted_job_gn_table:
            job = self.get_job_by_uuid(_job_id)
            if not job or job.status != JOB_RUNNING_STATUS or job not in self.running_job_queue:
                to_del_job_ids.append(_job_id)
        for _job_id in to_del_job_ids:
            idx = list(self.admitted_job_gn_table.keys()).index(_job_id)
            del self.admitted_job_gn_table[_job_id]
            self.prev_allocation_matrix = np.delete(self.prev_allocation_matrix, idx, axis=0)
        
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization (we record the ended job id list to update the 
            # crius runtime after the this job is removed)
            ended_job_id_list = self._gavel_optimize_running_jobs()
            for _job_id in ended_job_id_list:
                ended_job_info_table[_job_id] = self.get_job_by_uuid(_job_id).alias
            self.ended_job_queue.clear()
        
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._gavel_fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        return ended_job_info_table

# def _optimize_global_allocation(thr_matrix: Sequence[Sequence[float]], 
#                                 job_num_table: dict, gtype_num_table: dict, 
#                                 prev_alloc_matrix: Any = None, 
#                                 tc_thrsd: int = None):
#     """ 
#     Optimize the global allocation of all admitted jobs. 
#     The structure of the global allocation matrix is: 

#     |  GPU type A  |  GPU type B  |  GPU type C  |
#     |       1      |       0      |       0      |   Job A (2 GPUs)
#     |       1      |       0      |       0      |   Job B (4 GPUs)
#     |       0      |       1      |       0      |   Job C (4 GPUs)
#     |       0      |       0      |       1      |   Job D (8 GPUs)

#     In the example, job A and B are allocated with 2 and 4 A GPUs,
#     job C is allocated with 4 B GPUs and job D is allocated with 
#     8 C GPUs.
#     The job placement follows the packing style.
#     ----------------------------------------------------
#     The imeplementation refers to ThroughputNormalizedByCostSumWithPerfSLOs 
#     class in `max_sum_throughput.py`.
#     """
#     # Job gpu num matrix
#     _vec = [job_num_table[_key] for _key in job_num_table]
#     job_gn_matrix = np.tile(_vec, (len(_vec), 1)).T
#     # Total gpu type num matrix
#     total_gn_vector = np.array([gtype_num_table[_key] for _key in gtype_num_table])

#     thr_matrix = np.array(thr_matrix)
#     # Allocation matrix
#     # `x[i, j]` denotes whether job j(axis=1) is allocated with gpu type i (axis=0).
#     x = cp.Variable(thr_matrix.shape, boolean=True)
#     # Objective function
#     objective = cp.Maximize(cp.sum(cp.sum(cp.multiply(thr_matrix, x), 
#                                             axis=1)))
#     # Constraints
#     constraints = [ x >= 0 ]
#     # - Each admitted job must only be allocated on one gpu type
#     constraints.append(cp.sum(x, axis=1) == 1)
#     # - The allocated num of a gpu type cannot exceed total amount
#     constraints.append(cp.sum(cp.multiply(x, job_gn_matrix), axis=0)
#                         <= total_gn_vector)
#     # - The type-changed job amount should be no less than a threshold
#     if tc_thrsd:
#         constraints.append(cp.sum(cp.sum(cp.multiply(x, prev_alloc_matrix), axis=1)) 
#                         >= len(list(job_num_table.keys())) - tc_thrsd)

#     # Problem
#     cvxprob = cp.Problem(objective, constraints)
#     # Solve
#     _ = cvxprob.solve(solver=cp.ECOS_BB, verbose=False)

#     print(f"[I] Optimization status: {cvxprob.status}")

#     if cvxprob.status != "optimal":
#         return None, None
    
#     allocation = np.where(np.array(x.value) >= 0.5, 1, 0)
    
#     cluster_thr = np.sum(np.sum(
#         np.multiply(thr_matrix, allocation), axis=1
#     ))

#     return cluster_thr, allocation 
    

# thr_matrix = [
#     [0.1, 0.1, 0.5],
#     [0.1, 0.7, 0.3],
#     [0.1, 0.1, 0.5],
# ]

# job_num_table = {
#     "job_1": 2,
#     "job_2": 4,
#     "job_3": 2,
# }

# gtype_num_table = {
#     "g_a": 2,
#     "g_b": 4,
#     "g_c": 4,
# }

# cluster_thr, allocation = _optimize_global_allocation(thr_matrix, job_num_table, gtype_num_table)

# print(cluster_thr)

# print(allocation)
