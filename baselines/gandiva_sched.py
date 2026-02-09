#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
A more flexible implementation of Gandiva: Introspective Cluster Scheduling for Deep Learning. 
Since in large model scenarios, ignore the affinity restriction since not considering 
GPU co-location and over-subscription.
Ref: https://www.usenix.org/system/files/osdi18-xiao.pdf
"""

from typing import Sequence

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resources.hardware_specs import NODE_CAPACITY
from job.job import Job
from scheduler import (
    Scheduler, AblationOptions)
from utils import dict_counter_add, deepcopy
from macro.macro_def import (
    JOB_RUNNING_STATUS, JOB_PENDING_STATUS, IDLE_STATUS, USED_STATUS, 
    EMPTY_JOB_ID, INFEASIBLE_THR)


class GandivaSched(Scheduler):
    """ 
    The class of the Gandiva scheduler, in which we implement job migration & shrink / grow. 
    """
    def __init__(self, node_pool: dict, enable_alpa: bool = False, 
                 is_runtime: bool = False, verbose: bool = False, 
                 dummy_test: bool = False, sched_with_opt: bool = False):
        # Init base class
        super().__init__(node_pool, AblationOptions(force_dp=True), 
                         is_runtime, verbose, dummy_test, sched_with_opt)
        # Job grow / shrink
        self.tmp_resources = dict()                 # Temporarily occupied resources
        self.grown_job_init_resources = dict()      # The init resources of the grown jobs
    
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
            # Sort node id list to place in a packing style
            node_id_list = sorted(node_id_list, 
                                  key=lambda x: self._get_bubble_num(crs_table[x]), 
                                  reverse=True)
            print(" - The nodes list is:", [
                self.resource_interface.node_pool[_nid].alias for _nid in node_id_list[:len(best_locality)]
            ])
            # Place on these nodes
            is_modified = self._clear_placement(target_job.uuid, crs_table)
            assert not is_modified, \
                f"In Gandiva baseline, the to-allocated job should not appear in crs_table."
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
                # Sort node id list to place in a packing style
                node_id_list = sorted(node_id_list, 
                                      key=lambda x: self._get_bubble_num(crs_table[x]), 
                                      reverse=True)
                # Place on these nodes
                is_modified = self._clear_placement(target_job.uuid, crs_table)
                assert not is_modified, \
                    f"In Gandiva baseline, the to-allocated job should not appear in crs_table."
                # Place the entire job on the dst nodes
                is_placed = self._place_job(target_job, gpu_num, 
                                            node_id_list[:len(locality)], crs_table)
                assert is_placed
                # Placed
                return True
        # Cannot place with relaxed locality
        return False

    def _release_tmp_resources(self, crs_table: dict):
        """ 
        Update the crs table with releasing all temporarily occupied 
        resources by job grow.
        """
        for _nid in crs_table:
            for _gid in crs_table[_nid]:
                if _nid in self.tmp_resources and _gid in self.tmp_resources[_nid]:
                    crs_table[_nid][_gid]["status"] = IDLE_STATUS
                    crs_table[_nid][_gid]["used_job_id"] = EMPTY_JOB_ID
        return crs_table
    
    def _gandiva_analyze_resource_contention(self, target_job_id: str, prev_crs_table: dict, 
                                             decision_crs_table: dict):
        """ Analyze the resource contention between the new job and the grown jobs. """
        # The uuid list of the opportunism jobs that lead to 
        # resource contention with the pending job.
        conflict_jid_list = list()
        for _nid in prev_crs_table:
            for _gid in prev_crs_table[_nid]:
                if (prev_crs_table[_nid][_gid]["status"] != IDLE_STATUS and 
                    decision_crs_table[_nid][_gid]["used_job_id"] == target_job_id and 
                    prev_crs_table[_nid][_gid]["used_job_id"] not in conflict_jid_list):
                    # The conflicted gpu should: (1) Occupied by opportunisim job
                    conflict_jid_list.append(prev_crs_table[_nid][_gid]["used_job_id"])
        return conflict_jid_list
    
    ######################################
    #     Migration Related Functions    #
    ######################################

    def _is_migration_end(self, gpu_type: str):
        """ 
        Check whether the bubble num of each partially occupied nodes is less than 
        the half of the node capacity. 
        """
        for _nid in self.resource_interface.node_pool:
            node = self.resource_interface.node_pool[_nid]
            if node.gpu_type != gpu_type:
                # Skip since only consider homogeneous gpus
                continue
            if node.idle_gpu_num >= node.capacity // 2 and node.idle_gpu_num < node.capacity:
                # There exists a node with bubbles more than half of its capacity
                return False
        return True
    
    def _filter_cand_node_id_list(self, node_id_list: Sequence[str], 
                                  per_node_quota: int, crs_table: dict,
                                  ignored_node_id_list: Sequence[str] = None,
                                  gpu_type: str = None):
        """ 
        Filter the node id list to get the nodes that can contain the per-node quota. 
        """
        _list = list()
        for _nid in node_id_list:
            if ignored_node_id_list is not None and _nid in ignored_node_id_list:
                # Ignore this node
                continue
            if self.resource_interface.node_pool[_nid].gpu_type != gpu_type:
                # Skip nodes with different gpu type
                continue
            if self._get_bubble_num(crs_table[_nid]) >= per_node_quota:
                # Got enough idle gpu slots
                _list.append(_nid)
        return _list
    
    def _migrate_one_job(self, target_job: Job, crs_table: dict):
        """
        Migrate the target job to idle GPU slots to improve locality, 
        only consider migration among occupied nodes. 
        """
        (gpu_type, gpu_num, 
         cur_locality, node_id_list) = self._get_job_rt_stat(target_job.uuid, crs_table)

        locality = self._get_best_locality(gpu_num, gpu_type)
        while len(locality) < len(cur_locality):
            _per_node_quota = locality[0] - cur_locality[0]     # Quota -= slots occupied by target job
            cand_node_id_list = self._filter_cand_node_id_list(node_id_list, _per_node_quota, 
                                                               crs_table, gpu_type=gpu_type)
            if len(cand_node_id_list) >= len(locality):
                # Sort to place in a packing style
                cand_node_id_list = sorted(cand_node_id_list, 
                                           key=lambda x: self._get_bubble_num(crs_table[x]))
                # Place on these nodes
                is_modified = self._clear_placement(target_job.uuid, crs_table)
                assert is_modified
                # Place the entire job on the dst nodes
                is_placed = self._place_job(target_job, gpu_num, 
                                            cand_node_id_list[:len(locality)], crs_table)
                assert is_placed
                # Migration plan is found
                return True, crs_table
            else:
                # Downgrade the locality
                per_node_quota = locality[0] // 2
                assert per_node_quota > 0
                locality = [per_node_quota for _ in range(gpu_num // per_node_quota)]
        # Cannot find a migration plan
        return False, crs_table
    
    def _migrate_one_job_to_new_nodes(self, target_job: Job, crs_table: dict):
        """ 
        Migrate the entire target job to idle GPU slots in new nodes. 
        """
        (gpu_type, gpu_num, locality, 
         node_id_list) = self._get_job_rt_stat(target_job.uuid, crs_table)
        assert len(node_id_list) > 0, \
            f"In Gandiva, a migrated job must be allocated with resources before."

        best_locality = self._get_best_locality(gpu_num, gpu_type, None, None)
        per_node_quota = best_locality[0]
        while per_node_quota > locality[0]:
            _locality = [per_node_quota for _ in range(gpu_num // per_node_quota)]
            # Traverse all nodes to decide dst nodes in a packing style
            _all_nid_list = list(self.resource_interface.node_pool.keys())
            cand_node_id_list = self._filter_cand_node_id_list(_all_nid_list, per_node_quota,
                                                               crs_table, node_id_list, 
                                                               gpu_type=gpu_type)
            if len(cand_node_id_list) >= len(_locality):
                # Sort to place in a packing style
                cand_node_id_list = sorted(cand_node_id_list, 
                                           key=lambda x: self._get_bubble_num(crs_table[x]))
                # Place on these nodes
                is_modified = self._clear_placement(target_job.uuid, crs_table)
                assert is_modified
                # Place the entire job on the dst nodes
                is_placed = self._place_job(target_job, gpu_num, 
                                            cand_node_id_list[:len(_locality)], crs_table)
                assert is_placed
                # Migration plan is found
                return True, crs_table
            else:
                # Downgrade the locality
                per_node_quota = _locality[0] // 2
                assert per_node_quota > 0
        # Cannot find a migration plan
        return False, crs_table
    
    def _gandiva_migration_search(self):
        """
        Treat all idle GPU slots as the candidate desitination of the migration, 
        repeat until idle GPU num of each partially occupied node is less than a 
        threshold, or no feasible plan.
        In heterogenous scenario, we set the threshold to half of the node capacity.
        """
        if self.verbose:
            print("")
        print("[I] Begin Gandiva Migration search among all cross-nodes jobs...")

        # Stat cross-nodes jobs
        cross_nodes_job_queue = list()
        for _job in self.running_job_queue:
            if (_job.is_cross_nodes and 
                _job.resource_quota.locality[0] < NODE_CAPACITY[_job.resource_quota.gpu_type]):
                # Partially cross-nodes job
                cross_nodes_job_queue.append(_job)

        if len(cross_nodes_job_queue) == 0:
            print("[I][MGRT] No cross-nodes job exists.")
            return 
        
        # Calculate global and local migration num
        global_mgrt_num, _ = self._get_migration_num_bound(cross_nodes_job_queue)
        
        if self.verbose:
            print("")
        print("[I][MGRT] Cross-nodes job queue:")
        for _job in cross_nodes_job_queue:
            print(f" - Job alias: {_job.alias} | " + 
                  f"GPU type: {_job.resource_quota.gpu_type} | " + 
                  f"GPU num: {_job.resource_quota.gpu_num} | " + 
                  f"Locality: {_job.resource_quota.locality}")

        is_mgrt_plan_found = True
        while is_mgrt_plan_found and global_mgrt_num > 0:
            # Loop until no feasible plan is found
            is_mgrt_plan_found = False
            # Always perform the migration for the first migratable 
            # job in the cross-nodes job queue.
            for _job in cross_nodes_job_queue:
                if self._is_migration_end(_job.resource_quota.gpu_type):
                    # Reach end condition
                    print(f"[I][MGRT] The bubble num of all partially occupied " + 
                          f"nodes (GPU type: {_job.resource_quota.gpu_type}) is " + 
                          f"lower than the threshold.")
                    continue
                
                crs_table = self.resource_interface.get_crs_table()
                
                # Step 1. Try migration within the allocated resources of the job
                is_fsb, crs_table, = self._migrate_one_job(_job, crs_table)

                # Step 2. (if Step 1 failed) Try migration by migrating the 
                #         entire job to new nodes.
                if not is_fsb:
                    is_fsb, crs_table = self._migrate_one_job_to_new_nodes(_job, crs_table)
                
                if is_fsb:
                    # Migration plan is found
                    print(f"[I][MGRT] Job '{_job.alias}' (alias) has been migrated and " + 
                          f"its locality has been improved.")
                    is_mgrt_plan_found = True
                    # Apply the scheduling decision
                    self.resource_interface.apply_sched(crs_table)
                    # Update the resource status of running jobs
                    self._update_run_jobs(crs_table)
                    # Check locality
                    (_gpu_type, _gpu_num, 
                     _locality, _) = self._get_job_rt_stat(_job.uuid, crs_table)
                    _best_locality = self._get_best_locality(_gpu_num, _gpu_type, None, None)
                    if _locality[0] == _best_locality[0]:
                        # Placed with best locality
                        cross_nodes_job_queue.remove(_job)
                    # Update global migration num
                    global_mgrt_num -= 1
                    break

    ######################################
    #    Grow/Shrink Related Functions   #
    ######################################

    def _shrink_one_job(self, job_id: str, crs_table: dict):
        """ Shrink the target job back to the init size. """
        if job_id not in self.grown_job_init_resources:
            # Not grown yet
            return
        
        job = self.get_job_by_uuid(job_id)
        print(f"[I][SHK] Job '{job.alias}' (alias) has been shrinked " + 
              f"from {job.resource_quota.gpu_num} GPUs to init GPU num.")

        # Clear all occupied resources of the job
        is_modified = self._clear_placement(job_id, crs_table)
        assert is_modified
        
        # Allocate the init resources of the job        
        for _nid in self.grown_job_init_resources[job_id]:
            for _gid in self.grown_job_init_resources[job_id][_nid]:
                assert (_nid in crs_table) and (_gid in crs_table[_nid])
                # Color
                crs_table[_nid][_gid]["status"] = USED_STATUS
                crs_table[_nid][_gid]["used_job_id"] = job_id
        
        # Update tmp resources
        _removed_nid_list = list()          # Node uuids to be removed from tmp resources
        _removed_gid_list_table = dict()    # Gpu uuids to be removed from tmp resources
        for _nid in self.tmp_resources:
            _removed_gid_list_table[_nid] = list()
            for _gid in self.tmp_resources[_nid]:
                # Remove GPU record if needed
                if self.tmp_resources[_nid][_gid] == job_id:
                    _removed_gid_list_table[_nid].append(_gid)
            # Remove node record if needed
            if (len(list(self.tmp_resources[_nid])) == 
                len(_removed_gid_list_table[_nid])):
                _removed_nid_list.append(_nid)
        # Remove
        for _nid in _removed_gid_list_table:
            if _nid in _removed_nid_list:
                _ = self.tmp_resources.pop(_nid)
            else:
                for _gid in _removed_gid_list_table[_nid]:
                    _ = self.tmp_resources[_nid].pop(_gid)
        
        # Remove job record
        _ = self.grown_job_init_resources.pop(job_id)
    
    def _gandiva_job_grow_search(self):
        """ 
        Grow running jobs until fully occupying the occupied nodes. 
        We only need to grow the single-node jobs, the cross-nodes 
        jobs have already been trasferred into single-node jobs in 
        the previous gandiva migration search.
        """
        print("[I] Begin gandiva-style job grow search...")
        # TODO(chunyu): Can further set a thrshold to identify the improvement 
        # of growing and deccide whether to perform

        for job in self.running_job_queue:
            if (job.is_cross_nodes or 
                job.resource_quota.gpu_num >= NODE_CAPACITY[job.resource_quota.gpu_type]):
                # Only consider single-node partially-occupied jobs
                continue
            
            if job.uuid in self.grown_job_init_resources:
                # This job has already been grown and not shrinked. 
                # TODO(chunyu): Currently we do not support consistant grown among 
                #               different rounds.
                continue

            crs_table = self.resource_interface.get_crs_table()
            (gpu_type, gpu_num, 
             _, node_id_list) = self._get_job_rt_stat(job.uuid, crs_table)
            assert len(node_id_list) == 1

            # Grow
            new_gpu_num = gpu_num
            while (new_gpu_num * 2 <= 
                   self.resource_interface.node_pool[node_id_list[0]].idle_gpu_num + gpu_num):
                if self._get_delta_thr(job, gpu_type, new_gpu_num * 2, [new_gpu_num * 2], 
                                       gpu_type, new_gpu_num, [new_gpu_num]) <= 0:
                    # The new config leads to worse throughput
                    break
                # Double gpu num
                new_gpu_num = new_gpu_num * 2
                assert new_gpu_num <= NODE_CAPACITY[gpu_type]
            
            if new_gpu_num > gpu_num:
                # A grow plan is found
                print(f"[I][GROW] Job '{job.alias}' (alias) has been growed " + 
                      f"from {gpu_num} GPUs to {new_gpu_num} GPUs.")

                is_modified = self._clear_placement(job.uuid, crs_table)
                assert is_modified
                # Place with new GPU num
                is_placed = self._place_job(job, new_gpu_num, node_id_list, crs_table)
                assert is_placed
                # Record tmp resources and job init resources
                _cnt = 0
                for _gid in crs_table[node_id_list[0]]:
                    if crs_table[node_id_list[0]][_gid]['used_job_id'] == job.uuid:
                        _cnt += 1
                        if _cnt <= gpu_num:
                            # Record as the init resources of the job
                            if job.uuid not in self.grown_job_init_resources:
                                self.grown_job_init_resources[job.uuid] = dict()
                            if node_id_list[0] not in self.grown_job_init_resources[job.uuid]:
                                self.grown_job_init_resources[job.uuid][node_id_list[0]] = list()
                            self.grown_job_init_resources[job.uuid][node_id_list[0]].append(_gid)
                        else:
                            # Record as the tmp resources
                            if node_id_list[0] not in self.tmp_resources:
                                self.tmp_resources[node_id_list[0]] = dict()
                            self.tmp_resources[node_id_list[0]][_gid] = job.uuid
                assert _cnt == new_gpu_num
                # Apply the scheduling decision
                self.resource_interface.apply_sched(crs_table)
                # Update the resource status of running jobs
                self._update_run_jobs(crs_table)

    ######################################
    #      Restart Related Functions     #
    ######################################
    
    def _gandiva_pending_jobs_restart_trial(self):
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
            # For pending job restart in Gandiva, we exploit optimal parallelism
            # since it cannot modify gpu num or type.
            is_fsb = self._gandiva_fit_one_job(job, crs_table, force_opt=True)
            
            if is_fsb:
                print(f"[I] Pending job '{job.alias}' (alias) has been restarted...")
                # Analyze the resource contention between the new job and the grown jobs
                prev_crs_table = self.resource_interface.get_crs_table()
                conflict_job_id_list = self._gandiva_analyze_resource_contention(job.uuid, 
                                                                                 prev_crs_table, 
                                                                                 crs_table)
                # Restore all conflicted grown jobs
                for _job_id in conflict_job_id_list:
                    self._shrink_one_job(_job_id, prev_crs_table)
                
                is_conflict = False
                # Color the occupied resources of the new job
                for _nid in crs_table:
                    for _gid in crs_table[_nid]:
                        if crs_table[_nid][_gid]["used_job_id"] == job.uuid:
                            if prev_crs_table[_nid][_gid]["status"] != IDLE_STATUS:
                                is_conflict = True
                                break
                            # Color
                            prev_crs_table[_nid][_gid]["status"] = USED_STATUS
                            prev_crs_table[_nid][_gid]["used_job_id"] = job.uuid
                    if is_conflict:
                        break
                if is_conflict:
                    continue

                # Apply the scheduling decision and deploy on real resources
                self.resource_interface.apply_sched(prev_crs_table)
                # Update job status
                job.update_status(JOB_RUNNING_STATUS)
                # Record
                _restarted_job_queue.append(job)
                # Add to the running job queue in the scheduler
                self.running_job_queue.append(job)
                # Update the allocated resources of all running jobs.
                self._update_run_jobs(prev_crs_table)
            else:
                print(f"[I] Pending job '{job.alias}' (alias) cannot be restarted...")
                assert job.resource_quota.gpu_type not in _blocked_job_gpu_type
                _blocked_job_gpu_type.append(job.resource_quota.gpu_type)
        # Remove all restarted jobs from the pending queue
        for _job in _restarted_job_queue:
            self.pending_job_queue.remove(_job)

    ######################################
    #        Top-level Functions         #
    ######################################
    
    def _gandiva_fit_one_job(self, job: Job, crs_table: dict, 
                             force_opt: bool = False):
        """ Try with best locality -> relaxed locality. """
        # Update crs_table with tmp resources
        crs_table = self._release_tmp_resources(crs_table)

        # Step 1. Try to place with the best locality
        is_fsb = self._place_with_best_locality(job, crs_table, force_opt)

        # Step 2. (if Step 1 failed) Try to place with relaxed locality
        if not is_fsb:
            is_fsb = self._place_with_relaxed_locality(job, crs_table, force_opt)
        
        return is_fsb

    def _gandiva_fit_new_jobs(self):
        """ Try fit one job -> pending, disallowing migration. """
        print("")
        print("#################################################")
        print("#        Gandiva-Style New Job(s) Arrival       #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.submit_init_job_queue)} new jobs to be fitted...")

        # Sort jobs by init priority
        self.submit_init_job_queue = self._sort_jobs(self.submit_init_job_queue)

        for job in self.submit_init_job_queue:
            # If this job cannot be satisfied if with the best locality, 
            # drop it since cannot be scaling
            _best_locality = self._get_best_locality(job.resource_quota.gpu_num, 
                                                     job.resource_quota.gpu_type)
            _force_opt = not self.is_runtime
            if not self._is_alloc_feasible(job, job.resource_quota.gpu_type, 
                                           _best_locality, _force_opt):
                # Drop this job
                print(f"[I] Since job {job.alias} cannot be placed with best " + 
                      f"locality {_best_locality} and cannot be scaled, drop it...")
                continue
            
            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            # For new jobs arrival in Gandiva, we exploit optimal parallelism
            # since it cannot modify gpu num or type.
            is_fsb = self._gandiva_fit_one_job(job, crs_table, force_opt=True)
            
            if (is_fsb and not self._is_exist_pend_job_same_gpu_type(job.resource_quota.gpu_type)):
                # Analyze the resource contention between the new job and the grown jobs
                prev_crs_table = self.resource_interface.get_crs_table()
                conflict_job_id_list = self._gandiva_analyze_resource_contention(job.uuid, 
                                                                                 prev_crs_table, 
                                                                                 crs_table)
                # Restore all conflicted grown jobs
                for _job_id in conflict_job_id_list:
                    self._shrink_one_job(_job_id, prev_crs_table)
                
                is_conflict = False
                # Color the occupied resources of the new job
                for _nid in crs_table:
                    for _gid in crs_table[_nid]:
                        if crs_table[_nid][_gid]["used_job_id"] == job.uuid:
                            # assert prev_crs_table[_nid][_gid]["status"] == IDLE_STATUS
                            if prev_crs_table[_nid][_gid]["status"] != IDLE_STATUS:
                                is_conflict = True
                                break
                            # Color
                            prev_crs_table[_nid][_gid]["status"] = USED_STATUS
                            prev_crs_table[_nid][_gid]["used_job_id"] = job.uuid
                    if is_conflict:
                        break
                if is_conflict:
                    continue

                # No need to pend this job, directly apply the scheduling decision and 
                # deploy on real resources
                self.resource_interface.apply_sched(prev_crs_table)
                # Update job status
                job.update_status(JOB_RUNNING_STATUS)
                # Add to the running job queue in the scheduler
                self.running_job_queue.append(job)
                # Update the allocated resources of all running jobs.
                self._update_run_jobs(prev_crs_table)
            else:
                print(f"[I] Job '{job.alias}' (alias) has been pending as a vanilla pending job.")
                # No feasible plan found, need to pend this job
                job.update_status(JOB_PENDING_STATUS)
                # Add to the pending job queue in the scheduler
                self.pending_job_queue.append(job)
        
        print(f"[I] There are {len(self.running_job_queue)} jobs in running status...")
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")

    def _gandiva_optimize_running_jobs(self):
        """ Try migration search -> try restart pending jobs -> try grow jobs. """
        print("")
        print("#################################################")
        print("#    Gandiva-Style Running Jobs Optimization    #")
        print("#################################################")
        print("")
        print(f"[I] Totally {len(self.ended_job_queue)} jobs are ended...")
        
        # Step 1. Release all related resources of ended jobs and update decision queue
        ended_job_id_list = self._release_resources_and_update_decision_queue()

        # Step 2. Gandiva Migration Search
        self._gandiva_migration_search()

        # Step 3. Gandiva Pending Jobs Restart Trial
        self._gandiva_pending_jobs_restart_trial()

        # Step 4. Gandiva job grow search
        self._gandiva_job_grow_search()

        print(f"[I] There are {len(self.running_job_queue)} jobs in running status...")
        print(f"[I] There are {len(self.pending_job_queue)} jobs in pending status...")

        return ended_job_id_list

    def _clear_grow_shrink_records_of_ended_jobs(self):
        """ 
        Clear all records in grown_job_init_resources / tmp_resources 
        of the ended jobs. 
        """
        for job in self.ended_job_queue:
            # First, clear grown_job_init_resources
            if job.uuid in self.grown_job_init_resources:
                _ = self.grown_job_init_resources.pop(job.uuid)
            # Then, clear all tmp resources
            for _nid in job.resource_alloc.node_to_gpu_table:
                if _nid not in self.tmp_resources:
                    continue
                for _gid in job.resource_alloc.node_to_gpu_table[_nid]:
                    if _gid in self.tmp_resources[_nid]:
                        self.tmp_resources[_nid].pop(_gid)
                # Check whether need to pop this _nid
                if len(list(self.tmp_resources[_nid].keys())) == 0:
                    self.tmp_resources.pop(_nid)
    
    def schedule(self):
        """
        Periodically schedule to decide resource allocation and job placement. 
        Scheduling events: (1) Job arrival; (2) Job departure.
        """

        print(f"[I] Idle GPU num in cluster:", self.resource_interface.get_gtype_num_table(only_idle_gpus=True))
        
        # Update the remained iteration num of the running jobs and end
        self._update_and_check_end()
        # Clear all records of the ended jobs
        self._clear_grow_shrink_records_of_ended_jobs()
        makespan_list, jct_list, queue_time_list = self._get_end_job_metrics()
        before_resched_crs_table = self.resource_interface.get_crs_table()
        
        try:
            # Recheck that all admitted jobs are in running status in case some runtime errors
            to_del_job_ids = list()
            for _job_id in self.grown_job_init_resources:
                job = self.get_job_by_uuid(_job_id)
                if not job or job.status != JOB_RUNNING_STATUS or job not in self.running_job_queue:
                    to_del_job_ids.append(_job_id)
            for _job_id in to_del_job_ids:
                del self.grown_job_init_resources[_job_id]

                _removed_nid_list = list()          # Node uuids to be removed from tmp resources
                _removed_gid_list_table = dict()    # Gpu uuids to be removed from tmp resources
                for _node_id in self.tmp_resources:
                    _removed_gid_list_table[_node_id] = list()
                    for _gpu_id in self.tmp_resources[_node_id]:
                        if self.tmp_resources[_node_id][_gpu_id] == _job_id:
                            _removed_gid_list_table[_node_id].append(_gpu_id)
                    # Remove node record if needed
                    if (len(list(self.tmp_resources[_node_id])) == 
                        len(_removed_gid_list_table[_node_id])):
                        _removed_nid_list.append(_node_id)
                # Remove
                for _nid in _removed_gid_list_table:
                    if _nid in _removed_nid_list:
                        _ = self.tmp_resources.pop(_nid)
                    else:
                        for _gid in _removed_gid_list_table[_nid]:
                            _ = self.tmp_resources[_nid].pop(_gid)
        except Exception as e:
            print(f"[WARN] Failed to recheck and update grown jobs and tmp resources: {e}")
        
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization
            self._gandiva_optimize_running_jobs()
            self.ended_job_queue.clear()
        
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._gandiva_fit_new_jobs()
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
        
        try:
            # Recheck that all admitted jobs are in running status in case some runtime errors
            to_del_job_ids = list()
            for _job_id in self.grown_job_init_resources:
                job = self.get_job_by_uuid(_job_id)
                if not job or job.status != JOB_RUNNING_STATUS or job not in self.running_job_queue:
                    to_del_job_ids.append(_job_id)
            for _job_id in to_del_job_ids:
                del self.grown_job_init_resources[_job_id]

                _removed_nid_list = list()          # Node uuids to be removed from tmp resources
                _removed_gid_list_table = dict()    # Gpu uuids to be removed from tmp resources
                for _node_id in self.tmp_resources:
                    _removed_gid_list_table[_node_id] = list()
                    for _gpu_id in self.tmp_resources[_node_id]:
                        if self.tmp_resources[_node_id][_gpu_id] == _job_id:
                            _removed_gid_list_table[_node_id].append(_gpu_id)
                    # Remove node record if needed
                    if (len(list(self.tmp_resources[_node_id])) == 
                        len(_removed_gid_list_table[_node_id])):
                        _removed_nid_list.append(_node_id)
                # Remove
                for _nid in _removed_gid_list_table:
                    if _nid in _removed_nid_list:
                        _ = self.tmp_resources.pop(_nid)
                    else:
                        for _gid in _removed_gid_list_table[_nid]:
                            _ = self.tmp_resources[_nid].pop(_gid)
        except Exception as e:
            print(f"[WARN] Failed to recheck and update grown jobs and tmp resources: {e}")
        
        if len(self.ended_job_queue) > 0:
            # Running jobs optimization (we record the ended job id list to 
            # update the crius runtime after the this job is removed)
            ended_job_id_list = self._gandiva_optimize_running_jobs()
            for _job_id in ended_job_id_list:
                ended_job_info_table[_job_id] = self.get_job_by_uuid(_job_id).alias
            self.ended_job_queue.clear()
        
        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._gandiva_fit_new_jobs()
            self.submit_init_job_queue.clear()
        
        return ended_job_info_table
