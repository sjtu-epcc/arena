#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" The implementation of Sia: Heterogeneity-aware, goodput-optimized ML-cluster 
scheduling (https://dl.acm.org/doi/10.1145/3600006.3613175). The implementation 
is referred to Sia's public artifacts: https://github.com/siasosp23/artifacts/tree/main 
"""

from typing import (Sequence, Any, Callable, List)
import traceback
import numpy as np
import cvxpy as cp

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import Scheduler, AblationOptions
from job.job import Job
from db_querier import QueryConfigs
from resources.hardware_specs import NODE_CAPACITY
from utils import dict_counter_add, deepcopy
from macro.macro_def import (SUPPORTED_GPU_NUM_LIST, INFEASIBLE_THR, MAX_SUPPORTED_GPU_NUM, 
                             JOB_RUNNING_STATUS, JOB_PENDING_STATUS, MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE)

# Hyperparameters adopted from: https://github.com/siasosp23/artifacts/tree/main 
p_fairness = 0.5                # Fairness of allocations in Sia as selected in the paper
sia_goodput_clip_val = 3000.0   # Clip normalized goodput values to this if any larger
not_alloc_penalty = -1          # Penalty factor for not allocating a device configuration. As our scheduler always best-effort
                                # admit jobs, we set this penalty large (as described in Sia paper) for fairness.
sia_solver = "glpk"             # Used MILP solver
MAX_MGRT_JOB_NUM = 5            # Job num to be migrated one job-allocating round
MAX_JOB_NUM_PARTITION = 200     # Maximum job num in each optimizing partition to control problem size


class SiaSched(Scheduler):
    """ Sia-style scheduler with heterogeneity- and adaptivity-aware scheduling. 
    
    Batch-size tuning is disabled in all baseline schedulers, since as Sia says: "Large batch sizes result in high 
    throughput and GPU utilization, but may result in a generalization gap for the trained model".
    
    All jobs are forced to be non-preemptive, i.e., once a job is admitted and lanuched, it cannot be stopped 
    to release resources for another job. 
    """

    def __init__(
        self, node_pool: dict, supported_gpu_types: List[str], enable_alpa: bool = False, is_runtime: bool = False, 
        verbose: bool = False, dummy_test: bool = False, sched_with_opt: bool = False,
    ) -> None:
        super().__init__(
            node_pool, supported_gpu_types, AblationOptions(force_dp=True), is_runtime, verbose, 
            dummy_test, sched_with_opt,
        )
        # assert not is_runtime, "Currently Sia policy only supports simulation."
        self.admitted_job_gn_table = dict()     # Job uuid -> gpu num
        self.prev_allocation_matrix = None      # Allocation matrix last job-alloc event
        self.gtype_num_table = self.resource_interface.get_gtype_num_table()
        self.gpu_type_list = list(self.gtype_num_table.keys())
        self.num_gpu_types = len(self.gpu_type_list)
        self.avail_gpu_num = len(SUPPORTED_GPU_NUM_LIST)
        self.per_gpu_profile_overhead = 300     # Referred to Sia's paper, 6min/GPU

        self.device_configs_list = [
            (_gpu_type, _gpu_num) for _gpu_type in self.gpu_type_list for _gpu_num in SUPPORTED_GPU_NUM_LIST
        ]
        self.num_dev_configs = len(self.device_configs_list)
        self._updated_thr_cache = {}            # Online updated throughput info
        self._thr_model = self._init_throughput_model()

        if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
            # Further tuning Sia for better performance in revision mode
            global not_alloc_penalty, p_fairness, MAX_MGRT_JOB_NUM
            not_alloc_penalty = -1e7   # In revision mode, we force the new job to be allocated
            p_fairness = 0.5
            MAX_MGRT_JOB_NUM = 10
    
    def _init_throughput_model(self) -> Callable:
        """ Initialize Sia's throughput model for each job. """

        def __thr_model_impl(
            num_gpus: int, gpu_type: str, model_name: str, param_num: str, batch_size: int,            
        ) -> float:
            """ Callable implementation of throughput model. """
            if self._sched_with_opt:
                force_opt = True
                only_opt = True
                only_dp = False
                search_max = False
            else:
                # raise NotImplementedError(
                #     "Currently due to lack of DP profiling data, all baselines should be scheduled with AP data.",
                # )
                force_opt = False   # Do not skip OOM check, since Sia should schedule jobs with DP
                only_opt = False
                only_dp = True
                search_max = False
            
            (_, thr_single_gpu, _) = self.db_querier.query_db(
                QueryConfigs(
                    1, 1, gpu_type, model_name, param_num, batch_size, 
                    only_opt=only_opt, 
                    only_dp=only_dp,
                    search_max=search_max,
                ), force_opt=force_opt,
            )
            if num_gpus == 1 and thr_single_gpu > INFEASIBLE_THR:   # Single-GPU case
                return thr_single_gpu

            num_hosts = max(1, num_gpus // NODE_CAPACITY[gpu_type])
            num_devices_per_host = min(num_gpus, NODE_CAPACITY[gpu_type])
            cache_key = f"{model_name}_{param_num}_{batch_size}_{num_hosts}_{gpu_type}_" + \
                        f"{num_hosts}_n_{num_devices_per_host}_d"
            if cache_key in self._updated_thr_cache:   # Online profiled and cached
                return self._updated_thr_cache[cache_key]
            
            # (For ablation) Minimal GPU num that has precise AP performance 
            # NOTE: Cannot directly use `sched_with_opt`, because it also uses linear estimation for AP
            # NOTE: We try enable random pending job restart...
            # min_gpu_num_precise_ap = 1
            min_gpu_num_precise_ap = 16
            if num_gpus <= min_gpu_num_precise_ap:
                # Directly query AP performance
                (_, thr, _) = self.db_querier.query_db(
                    QueryConfigs(num_hosts, num_devices_per_host, gpu_type, model_name, param_num, 
                                 batch_size, only_opt=True),
                    force_opt=True,
                )
                return thr
            
            # Check OOM. If so, return infeasible throughput
            # We don't use the profiled throughput directly, only for OOM check. Instead, we 
            # use the throughput model that is claimed in Sia paper.
            (is_fsb, _, _) = self.db_querier.query_db(
                QueryConfigs(
                    num_hosts, num_devices_per_host, gpu_type, model_name, param_num, batch_size,
                    only_opt=only_opt,
                    only_dp=only_dp,
                    search_max=search_max,
                ), force_opt=force_opt,
            )
            if not is_fsb:
                return INFEASIBLE_THR
            
            if thr_single_gpu <= INFEASIBLE_THR:
                # This model cannot run on single GPU
                # In this case, we search for the minimal required GPU num of this job, and use the 
                # linear scaling (like what Sia does) to estimate single-GPU throughput.
                num_gpus_scaled = 1
                while num_gpus_scaled <= MAX_SUPPORTED_GPU_NUM:
                    num_hosts_scaled = max(1, num_gpus_scaled // NODE_CAPACITY[gpu_type])
                    num_devices_per_host_scaled = min(num_gpus_scaled, NODE_CAPACITY[gpu_type])
                    (_, thr_single_gpu_scaled, _) = self.db_querier.query_db(
                        QueryConfigs(
                            num_hosts_scaled, num_devices_per_host_scaled, gpu_type, model_name, 
                            param_num, batch_size, 
                            only_opt=only_opt,
                            only_dp=only_dp,
                            search_max=search_max,
                        ), force_opt=force_opt,
                    )

                    num_gpus_scaled *= 2
                    if thr_single_gpu_scaled > 0:
                        thr_single_gpu = thr_single_gpu_scaled / num_gpus_scaled

            if thr_single_gpu <= INFEASIBLE_THR:
                # This model cannot run on any profiled GPU num of this type
                return INFEASIBLE_THR
            
            # Linear estimation from another cached GPU type
            for est_gpu_type in self.supported_gpu_types:
                est_cache_key = f"{model_name}_{param_num}_{batch_size}_{num_hosts}_{est_gpu_type}_" + \
                                f"{num_hosts}_n_{num_devices_per_host}_d"
                if est_cache_key not in self._updated_thr_cache:
                    continue

                (_, est_thr_single_gpu, _) = self.db_querier.query_db(
                    QueryConfigs(
                        1, 1, est_gpu_type, model_name, param_num, batch_size,
                        only_opt=only_opt,
                        only_dp=only_dp,
                        search_max=search_max,
                    ), force_opt=force_opt,
                )
                return (thr_single_gpu / est_thr_single_gpu) * self._updated_thr_cache[est_cache_key]
            
            # Fallback to linear scaling
            return (thr_single_gpu * num_gpus)
        
        return __thr_model_impl

    ######################################
    #   Optimization Related Functions   #
    ######################################

    def _optimize_global_allocation(self, goodput_matrix: Sequence[Sequence[float]], 
                                    prev_alloc_matrix: Any = None):
        """ Optimize the global allocation of all admitted jobs. 

        The structure of the global allocation matrix is: 

        | Type: A, Num: 1 | Type: A, Num: 2 | Type: B, Num: 1 | Type: B, Num: 2 |
        |        1        |        0        |         0       |        0        |   Job A 
        |        1        |        0        |         0       |        0        |   Job B
        |        0        |        0        |         1       |        0        |   Job C 
        |        0        |        0        |         0       |        1        |   Job D 

        In the example, job A and B are allocated with 1 A GPU, job C with 1 B GPU, and job 
        D with 2 B GPUs. Job placement follows the packing style. 
        
        As described in Sia's paper, the optimization goal of Sia is to maximize the sum of 
        normalized goodputs for all jobs. i.e.: 

        max sum_{job} ( sum_{config} (A_{ij} x goodput_{ij}) + \lambda x (1 - ||A_{i}||) )

        where A is the allocation matrix, \lambda is the penalty for not allocating a job. 
        """

        # Apply factors and clip as described in the paper
        for i in range(len(goodput_matrix)):
            job_id = list(self.admitted_job_gn_table.keys())[i]
            job = self.get_job_by_uuid(job_id)
            resched_num = 0 if job_id not in self.resched_num_table else self.resched_num_table[job_id]
            realloc_factor = max(
                (job.exec_time - resched_num * MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE) / \
                    (job.exec_time + MAX_RESCHED_OVERHEAD_WITHOUT_PRUNE),
                0.01,
            )
            for j in range(len(goodput_matrix[0])):
                if goodput_matrix[i][j] > INFEASIBLE_THR:
                    goodput_matrix[i][j] = np.power(goodput_matrix[i][j] * realloc_factor, p_fairness)
                    # Clip
                    goodput_matrix[i][j] = min(goodput_matrix[i][j], sia_goodput_clip_val)
        
        goodput_matrix = np.array(goodput_matrix)
        num_jobs = goodput_matrix.shape[0]
        # Total GPU type num matrix
        total_gn_vector = np.array([
            self.gtype_num_table[_key] for _key in self.gtype_num_table])
        gpu_num_vector = []
        for i in range(total_gn_vector.shape[0]):
            gpu_num_vector += SUPPORTED_GPU_NUM_LIST
        gpu_num_vector = np.array(gpu_num_vector)
        gpu_num_matrix = np.tile(gpu_num_vector, (num_jobs, 1))
        num_partitions = (num_jobs - 1) // MAX_JOB_NUM_PARTITION + 1

        overall_allocation, overall_goodput = None, 0.0
        for partition_index in range(num_partitions):
            print(f"[TMP] Solving the {partition_index + 1}-th problem partition...")
            
            left_index = partition_index * MAX_JOB_NUM_PARTITION
            right_index = min((partition_index + 1) * MAX_JOB_NUM_PARTITION, num_jobs)
            num_jobs_partition = right_index - left_index
            is_last_partition = (partition_index == num_partitions - 1)

            # Allocation matrix
            # `x[i, j]` denotes whether job j(axis=1) is allocated with device configuration i (axis=0).
            x = cp.Variable((num_jobs_partition, goodput_matrix.shape[1]), boolean=True)
            # Objective function
            if is_last_partition:
                if p_fairness > 0:
                    objective = cp.Maximize(
                        cp.sum(
                            cp.sum(cp.multiply(goodput_matrix[left_index:right_index], x), axis=1), axis=0,
                        ) + (1 - cp.sum(x[-1]) * not_alloc_penalty)
                    )
                else:
                    objective = cp.Minimize(
                        cp.sum(
                            cp.sum(cp.multiply(goodput_matrix[left_index:right_index], x), axis=1), axis=0,
                        ) + (1 - cp.sum(x[-1]) * not_alloc_penalty)
                    )
            else:
                if p_fairness > 0:
                    objective = cp.Maximize(
                        cp.sum(cp.sum(cp.multiply(goodput_matrix[left_index:right_index], x), axis=1), axis=0)
                    )
                else:
                    objective = cp.Minimize(
                        cp.sum(cp.sum(cp.multiply(goodput_matrix[left_index:right_index], x), axis=1), axis=0)
                    )

            # Constraints
            constraints = [ x >= 0 ]
            # - Each previously admitted job must only be allocated one device configuration
            if x.shape[0] > 1:
                if is_last_partition:
                    constraints.append(cp.sum(x[:-1], axis=1) == 1)
                else:
                    constraints.append(cp.sum(x, axis=1) == 1)
            # - The newly to-allocate job can be allocated at most one device configuration
            if is_last_partition:
                # if not os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                # # if False:
                #     constraints.append(cp.sum(x[-1]) <= 1)
                # else:
                #     # In revision mode, we force the new job to be allocated to further allocate more jobs,
                #     # This significantly improves the performance of Sia.
                #     constraints.append(cp.sum(x[-1]) <= 1)
                constraints.append(cp.sum(x[-1]) <= 1)
            # - The allocated num of a GPU type cannot exceed total amount
            for i in range(total_gn_vector.shape[0]):
                constraints.append(
                    cp.sum(
                        cp.sum(
                            cp.multiply(x, gpu_num_matrix[left_index:right_index]), axis=0,
                        )[i * self.avail_gpu_num : (i + 1) * self.avail_gpu_num]
                    ) <= total_gn_vector[i]
                )
            # - The type-changed job amount should be no more than a threshold
            if prev_alloc_matrix is not None:
                constraints.append(cp.sum(cp.sum(cp.multiply(x, prev_alloc_matrix[left_index:right_index]), axis=1)) 
                                >= num_jobs_partition - (MAX_MGRT_JOB_NUM // num_partitions))
            
            # Problem
            cvxprob = cp.Problem(objective, constraints)
            # Solver
            if sia_solver == "glpk":
                solver = cp.GLPK_MI
            elif sia_solver == "cplex":
                solver = cp.CPLEX
            elif sia_solver == "cbc":
                solver = cp.CBC
            else:
                raise ValueError(f"Unsupported solver: {sia_solver}")
            # Solve
            if sia_solver == "glpk":
                _ = cvxprob.solve(solver=solver, tm_lim=30000, verbose=False)
            else:
                _ = cvxprob.solve(solver=solver, verbose=False)

            print(f"[TMP] Optimization status: {cvxprob.status}")

            # if cvxprob.status != "optimal":
            if "optimal" not in cvxprob.status:
                return None, None
            
            allocation = np.where(np.array(x.value) >= 0.5, 1, 0)
            cluster_goodput = np.sum(np.sum(
                np.multiply(goodput_matrix[left_index:right_index], allocation), axis=1
            ))
            overall_goodput += cluster_goodput

            # Update total GPU type num matrix
            for i in range(allocation.shape[0]):
                if 1 not in allocation[i]:
                    assert i == allocation.shape[0] - 1, "Only the new job can be not allocated."
                    continue

                dev_config_index = list(allocation[i]).index(1)
                gpu_type_index = dev_config_index // self.avail_gpu_num
                num_gpus = SUPPORTED_GPU_NUM_LIST[dev_config_index % self.avail_gpu_num]
                total_gn_vector[gpu_type_index] -= num_gpus
                assert total_gn_vector[gpu_type_index] >= 0
            
            overall_allocation = allocation if overall_allocation is None \
                else np.concatenate((overall_allocation, allocation), axis=0)

        return overall_goodput, overall_allocation 
    
    ######################################
    #    Job Arrival Related Functions   #
    ######################################

    def _place_with_best_locality(self, target_job: Job, gpu_type: str, 
                                  crs_table: dict, force_opt: bool = False):
        """ Place the new job with the best locality. """
        gpu_num = target_job.resource_quota.gpu_num
        best_locality = self._get_best_locality(gpu_num, gpu_type, None, None)
        if not self._is_alloc_feasible(target_job, gpu_type, best_locality, force_opt):
            return False   # Cannot placed with best locality
        
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

    def _gen_goodput_matrix(self):
        """ Generate goodput matrix, refer to: https://dl.acm.org/doi/10.1145/3600006.3613175 """
        job_id_list = list(self.admitted_job_gn_table.keys())
        goodput_matrix = []
        for job_id in job_id_list:
            job = self.get_job_by_uuid(job_id)
            (model_name, param_num) = job.model_name.split("__")
            goodputs = []
            num_gpus_min = 1e9
            for gpu_type in self.gtype_num_table:
                for num_gpus in SUPPORTED_GPU_NUM_LIST:
                     # Throughput
                    thr = self._thr_model(
                        num_gpus, gpu_type, model_name, param_num, job.batch_size,
                    )
                    if thr > INFEASIBLE_THR and num_gpus < num_gpus_min:
                        num_gpus_min = num_gpus
                    # Statistical efficiency
                    efficiency = self._efficiency()
                    # Goodput
                    goodputs.append(thr * efficiency)

            # Normalize goodputs of job i as (described in Sia's paper):
            #    G_{ij} = N^{min}_{i} x (G_{ij} / min_j G_{ij}) 
            if any([_g > INFEASIBLE_THR for _g in goodputs]):
                min_g = min([_g for _g in goodputs if _g > INFEASIBLE_THR])
                for i in range(self.num_dev_configs):
                    if goodputs[i] > INFEASIBLE_THR:
                        goodputs[i] = num_gpus_min * (goodputs[i] / min_g)                    

            goodput_matrix.append(goodputs)

        return goodput_matrix
    
    def _efficiency(self) -> float:
        """ Compute statistical efficiency.
        
        Accroding to Pollux (https://www.usenix.org/conference/osdi21/presentation/qiao), The 
        equation should be: (PGNS + bs0) / (PGNS + bs). In our scenarios, since the global batch
        size is fixed, the efficiency would constantly be 1.
        """
        return 1.0

    def _sia_admission_control(self, target_job: Job):
        """ 
        The sia-style admission control. We decide whether to admit the target
        job based on whether the cluster overall goodput will be degraded after adimtting 
        this job.
        """
        # Temporarily add target job
        self.admitted_job_gn_table[target_job.uuid] = target_job.resource_quota.gpu_num
        
        prev_allocation_matrix = np.append(
            self.prev_allocation_matrix, [np.ones((self.num_dev_configs, ))], axis=0
        ) if self.prev_allocation_matrix is not None else None
        goodput_matrix = self._gen_goodput_matrix()

        # Optimize allocation
        try:
            cluster_goodput, allocation = self._optimize_global_allocation(
                goodput_matrix, prev_allocation_matrix,
            )
        except Exception as e:
            print(f"[E] Meet optimization error, reject this job: {e}")
            traceback.print_exc()
            if not self.is_runtime:
                exit(0)
            else:
                return False, None

        if not cluster_goodput or 1 not in allocation[-1]:
            # Currently this job cannot be admitted

            print(cluster_goodput)
            print(allocation)

            del self.admitted_job_gn_table[target_job.uuid]
            return False, None
        
        # This job can be admitted, parse gpu type modification
        mdf_gtype_table = {}
        job_id_list = list(self.admitted_job_gn_table.keys())
        # Other jobs
        for i in range(allocation.shape[0] - 1):
            job = self.get_job_by_uuid(job_id_list[i])
            # GPU num
            cur_index = list(self.prev_allocation_matrix[i]).index(1) % self.avail_gpu_num 
            job.resource_quota.gpu_num = SUPPORTED_GPU_NUM_LIST[cur_index]
            # GPU type
            prev_index = list(self.prev_allocation_matrix[i]).index(1) // self.avail_gpu_num \
                if self.prev_allocation_matrix is not None else None
            cur_index_ = list(allocation[i]).index(1) // self.avail_gpu_num
            if prev_index is None or prev_index != cur_index_:
                job_id = list(self.admitted_job_gn_table.keys())[i]
                prev_gpu_type = self.gpu_type_list[prev_index] if prev_index is not None else None
                cur_gpu_type = self.gpu_type_list[cur_index_]
                mdf_gtype_table[job_id] = (prev_gpu_type, cur_gpu_type)
        # Target job
        target_n_index = list(allocation[-1]).index(1) % self.avail_gpu_num
        target_job.resource_quota.gpu_num = SUPPORTED_GPU_NUM_LIST[target_n_index]
        target_t_index = list(allocation[-1]).index(1) // self.avail_gpu_num
        mdf_gtype_table[target_job.uuid] = (None, self.gpu_type_list[target_t_index])
        # Update allocation matrix
        self.prev_allocation_matrix = allocation

        return True, mdf_gtype_table
    
    def _sia_resource_allocation(self, target_job: Job, crs_table: dict, 
                                   mdf_gtype_table: dict, force_opt: bool = False):
        """ 
        Sia-style resource allocation. Given the modifying info after admitting the new 
        job, sia allocates resources in a packing style.
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
            is_fsb = self._place_with_best_locality(job, new_gpu_type, crs_table, force_opt)
            # Try place with relaxed locality
            if not is_fsb:
                is_fsb = self._place_with_relaxed_locality(job, new_gpu_type, crs_table, force_opt)
            
            if not is_fsb:   # Cannot be placed
                return False, crs_table
        
        return True, crs_table

    ######################################
    #      Restart Related Functions     #
    ###################################### 

    def _sia_pending_jobs_restart_trial(self):
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
            is_feasible, crs_table = self._sia_fit_one_job(job, crs_table)

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
                    # We allow Sia to restart in random order
                    # _blocked_job_gpu_type.append(job.resource_quota.gpu_type)
                    pass

        # Remove all restarted jobs from the pending queue
        for _job in _restarted_job_queue:
            self.pending_job_queue.remove(_job)
    
    ######################################
    #        Top-level Functions         #
    ######################################

    def _sia_fit_one_job(self, job: Job, crs_table: dict, force_opt: bool = False):
        """ Similar to Gavel policy. """
        # Admission control, use DP results for optimization process
        is_admit, mdf_gtype_table = self._sia_admission_control(job)
        if not is_admit:   # # Can not be admitted, need pending
            print(f"[I] Job '{job.alias}' is not admitted.")
            return False, crs_table

        print(f"[I] Job '{job.alias}' has been admitted.")
        
        # Resource allocation
        _crs_table = deepcopy(crs_table)
        # We modify crs_table in this function, use optimal profiled data to allocate resources
        is_fsb, _crs_table = self._sia_resource_allocation(job, _crs_table, 
                                                           mdf_gtype_table, force_opt)
        
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

    def _sia_fit_new_jobs(self):
        """ Compute goodput matrix and maximize overall goodput. """
        print("")
        print("#################################################")
        print("#         Sia-Style New Job(s) Arrival          #")
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

            # Add job profiling overhead only in revision mode
            if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                self.queue_time_table[job.uuid] = self.per_gpu_profile_overhead

            crs_table = self.resource_interface.get_crs_table()
            # Fit one job
            is_feasible, crs_table = self._sia_fit_one_job(job, crs_table, force_opt=True)

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

    def _sia_optimize_running_jobs(self):
        """ Optimize running jobs. """
        print("")
        print("#################################################")
        print("#       Sia-Style Running Jobs Optimization     #")
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
        self._sia_pending_jobs_restart_trial()

        return ended_job_id_list

    def schedule(self):
        """ Periodically schedule to decide resource allocation and job placement. 
        
        Scheduling events: (1) Job arrival; (2) Job departure.
        """

        print(f"[I] Idle GPU num in cluster:", self.resource_interface.get_gtype_num_table(only_idle_gpus=True))
        
        # Update the remained iteration num of the running jobs and end
        self._update_and_check_end()
        makespan_list, jct_list, queue_time_list = self._get_end_job_metrics()
        before_resched_crs_table = self.resource_interface.get_crs_table()

        if len(self.ended_job_queue) > 0:
            # Running jobs optimization
            self._sia_optimize_running_jobs()
            self.ended_job_queue.clear()

        if len(self.submit_init_job_queue) > 0:
            # New jobs arrival
            self._sia_fit_new_jobs()
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
