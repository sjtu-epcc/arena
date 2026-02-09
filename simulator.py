#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to the simulator to simulate scheduling scenarios with job workload trace.
"""

import os
import uuid
import argparse
import numpy as np
from typing import (
    Sequence, Any, Dict, Union, List)

from resources.resource_abs import GPU, Node
from resources.hardware_specs import NODE_CAPACITY, gpu_specs_suite, supported_gpu_type
from job.job import Job, ResourceQuota
from trace_manager import TraceManager, ArrivalEvent
from scheduler import (
    Scheduler, AblationOptions)
from baselines.fcfs_sched import FCFSSched
from baselines.gandiva_sched import GandivaSched
from baselines.elasticflow_sched import ElasticFlowSched
from baselines.gavel_sched import GavelSched
from baselines.sia_sched import SiaSched
from plot.plotter import (
    plot_cluster_thr, plot_queuing_time, read_jct_data_and_plot)
from utils import (
    read_json_content, mkdir_if_not_exist, get_int_upper_bound,
    path_join_and_check)
from macro.macro_def import (
    PREC, LOOSEN_DEADLINE, MGRT_CFGS, DDL_LOOSEN_RATIO, 
    ITER_NUM_SCALE_FACTOR_SIM)


class Simulator:
    """ 
    The class of the simulator, which performs simulation experiments on the 
    global scheduler with input workload trace. 
    """

    def __init__(
        self,
        policy: str = "crius",
        trace_type: str = "philly",
        enable_alpa: bool = True,
        dummy_test: bool = False,
        sched_with_opt: bool = False,
        job_submit_density: float = 1.0,
        prepend_profile_overhead: bool = False,
        disable_ap_prune: bool = False,
        disable_single_device_profiler: bool = False,
        verbose: bool = False,
    ) -> None:
        """ Initialization. """

        self.policy = policy
        self.trace_type = trace_type
        self.enable_alpa = enable_alpa
        # # FIXME: When the profiling database only contains profiling data of optimal parallelism,
        # #        this option should always be turned on.
        # sched_with_opt = True
        self.prepend_profile_overhead = prepend_profile_overhead
        self.disable_ap_prune = disable_ap_prune
        self.disable_single_device_profiler = disable_single_device_profiler

        # Global environmental variables
        if not os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
            os.environ["CRIUS_PROF_DB_PATH"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "database/prof_database.pkl"
            )
            supported_gpu_types = supported_gpu_type
        else:
            os.environ["CRIUS_PROF_DB_PATH"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "database/prof_database_revision.json"
            )
            os.environ["DOWNGRADE_MAX_DEPTH"] = "3"     # Downgrade search depth
            supported_gpu_types = ["h100", "l20"]
        
        os.environ["CRIUS_TRACE_DIR"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "traces"
        )
        os.environ["SCHED_POLICY"] = str(policy)

        # Cluster resources
        node_pool, gpu_types = self._construct_cluster()
        # Trace manager
        self.trace_manager = TraceManager(job_submit_density=job_submit_density)

        if self.enable_alpa:
            # Force to deploy jobs with AP
            os.environ["CRIUS_FORCE_DEPLOY_WITH_AP"] = "1"

        if self.prepend_profile_overhead:
            os.environ["CRIUS_PREPEND_PROFILE_OVERHEAD"] = "1"
            # raise ValueError(
            #     "'--prepend_profile_overhead' has not been well tested yet, as our end-event driven pending " + 
            #     "job restart would not restart jobs that finish profiling on time."
            # )

        if self.disable_ap_prune:
            os.environ["CRIUS_DISABLE_AP_PRUNE"] = "1"
            if policy.split("-")[0] != "crius":
                raise ValueError(
                    "'--disable_ap_prune' is only valid for Crius policy."
                )
            else:
                print("[I] Disable AP pruning for Crius.")  

        if self.disable_single_device_profiler:
            os.environ["CRIUS_DISABLE_SINGLE_DEVICE_PROFILER"] = "1"
            if policy.split("-")[0] != "crius":
                raise ValueError(
                    "'--disable_single_device_profiler' is only valid for Crius policy."
                )
            else:
                print("[I] Disable single GPU profiler for Crius.")      

        # Scheduler
        if policy.split("-")[0] == "crius":
            # Crius series
            _ablation_options = AblationOptions(force_dp=(policy == "crius-dp"),
                                                disable_scaling=(policy == "crius-ns"),
                                                disable_htc=(policy == "crius-nh"),
                                                disable_opportunistic=(policy == "crius-no"),
                                                disable_migration=(policy == "crius-nm"),
                                                enable_ddl=(policy == "crius-ddl"))
            self.scheduler = Scheduler(node_pool, supported_gpu_types, _ablation_options, verbose=verbose, 
                                       dummy_test=dummy_test, sched_with_opt=sched_with_opt)
        elif policy.split("-")[0] == "fcfs":
            # First-come-first-serve series
            self.scheduler = FCFSSched(node_pool, supported_gpu_types, is_allow_relaxed=(policy == "fcfs-r"), 
                                       enable_alpa=enable_alpa, verbose=verbose, 
                                       dummy_test=dummy_test, sched_with_opt=sched_with_opt)
        elif policy == "gandiva":
            # Gandiva scheduler
            self.scheduler = GandivaSched(node_pool, enable_alpa=enable_alpa, 
                                          verbose=verbose, dummy_test=dummy_test, 
                                          sched_with_opt=sched_with_opt)
        elif policy.split("-")[0] == "elasticflow":
            # Elasticflow series
            self.scheduler = ElasticFlowSched(node_pool, supported_gpu_types, enable_ddl=(policy != "elasticflow-l"), 
                                              enable_alpa=enable_alpa, 
                                              verbose=verbose, dummy_test=dummy_test, 
                                              sched_with_opt=sched_with_opt)
        elif policy == "gavel":
            # Gavel scheduler
            self.scheduler = GavelSched(node_pool, supported_gpu_types, enable_alpa=enable_alpa, 
                                        verbose=verbose, dummy_test=dummy_test, 
                                        sched_with_opt=sched_with_opt)
        elif policy == "sia":
            # Sia scheduler
            self.scheduler = SiaSched(node_pool, supported_gpu_types, enable_alpa=enable_alpa, verbose=verbose, 
                                      dummy_test=dummy_test, sched_with_opt=sched_with_opt)
        else:
            raise ValueError(f"Invalid policy: {policy}")
        
        # Lists of performance metrics
        self.thr_list = list()                  # Throughput
        self.avg_job_thr_per_gpu_list = list()  # Average job-level throughput per GPU
        self.makespan_list = list()             # Makespan = max jct of all running jobs
        self.jct_list = list()                  # JCT = job queuing time + job execution time
        self.queuing_time_list = list()         # Queuing time

    def _construct_cluster(self):
        """ Construct the node resources of the cluster. """

        cluster_spec_json_path = "./resources/sim_cluster_spec.json" if not os.environ.get(
            "CRIUS_REVISION_MODE", "false") == "true" else "./resources/h100_l20_cluster_spec.json"
        if os.environ.get("CRIUS_REVISION_MODE", "false") == "true" and os.environ.get("CRIUS_HOMOGENEOUS_CLUSTER", "false") == "true":
            cluster_spec_json_path = "./resources/h100_cluster_spec.json"

        # Load node configurations
        node_cfgs = dict()
        cluster_specs = read_json_content(json_path=cluster_spec_json_path)[0]
        gpu_types = []
        for _gpu_type in cluster_specs:
            gpu_types.append(_gpu_type)
            node_cfgs[_gpu_type] = cluster_specs[_gpu_type]["node_num"]
        
        # Construct resource pool in granularity of node
        node_pool = dict()
        node_cnt = 0
        for gpu_type in node_cfgs:
            node_num, node_cap = node_cfgs[gpu_type], NODE_CAPACITY[gpu_type]
            gpu_specs = {
                "max_mem": gpu_specs_suite[gpu_type].max_mem,
                "max_bw": gpu_specs_suite[gpu_type].max_bw,
                "sm_num": gpu_specs_suite[gpu_type].sm_sum,
            }
            for _ in range(node_num):
                node_cnt += 1
                node_id = str(uuid.uuid1())
                # Add gpus inside node
                gpu_list = [
                    GPU(
                        uuid=str(uuid.uuid1()),
                        alias="gpu_" + str(node_cnt).zfill(2) + "_" + str(_i + 1).zfill(2),
                        type=gpu_type,
                        node_id=node_id, 
                        gpu_specs=gpu_specs
                    ) for _i in range(NODE_CAPACITY[gpu_type])
                ]
                # Add node
                node_pool[node_id] = Node(
                    uuid=node_id, alias="node_" + str(node_cnt).zfill(2), type="none",
                    capacity=NODE_CAPACITY[gpu_type], gpu_list=gpu_list
                )

        return node_pool, gpu_types

    def _load_trace(
        self,
        num_skip_round: int = 0,
        start_with_nonempty_round: bool = False,
        use_dummy_trace: bool = False,
    ) -> List[Any]:
        """ Load job workload. """

        if self.trace_type == "philly":
            trace_name = "simulate_trace_philly.csv" \
                if not use_dummy_trace else "dummy_trace.csv"
        elif self.trace_type == "helios":
            trace_name = "simulate_trace_venus.csv"
        elif self.trace_type == "pai":
            trace_name = "simulate_trace_pai.csv"
        else:
            raise ValueError(f"Invalid trace type: {self.trace_type}")
        
        if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
            trace_name = "runtime_trace_revision.csv"
            if os.environ.get("CRIUS_HOMOGENEOUS_CLUSTER", "false") == "true":
                trace_name = "runtime_trace_revision_homo.csv"

        arrival_trace = self.trace_manager.load_trace(trace_name)
        if num_skip_round > 0:
            if start_with_nonempty_round:
                while len(arrival_trace[num_skip_round]) == 0:
                    num_skip_round += 1
            arrival_trace = arrival_trace[num_skip_round:]

        return arrival_trace

    def _submit_jobs(
        self, 
        arrival_events: Sequence[ArrivalEvent],
        eval_overhead: bool = False,
    ) -> None:
        """ Submit arrived jobs in current scheduling round. """
        
        for event in arrival_events:
            # Job instance
            resource_quota = ResourceQuota(event.job_id, event.job_gpu_num, event.job_gpu_type)
            # Query profiling database
            best_locality = self.scheduler._get_best_locality(event.job_gpu_num, event.job_gpu_type)
            iter_time = self.scheduler._get_iter_time(
                event.model_name, event.batch_size, event.job_gpu_type, best_locality, force_opt=True
            )
            # Job deadline
            # loosen_ddl = self.scheduler.global_timer + iter_time * event.iter_num * DDL_LOOSEN_RATIO
            loosen_ddl = self.scheduler.global_timer + LOOSEN_DEADLINE
            # Loose job deadline
            _time_budget = event.deadline - self.scheduler.global_timer
            deadline = self.scheduler.global_timer + _time_budget * 5.0 \
                            if self.policy != "elasticflow-l" else loosen_ddl
            iter_num = event.iter_num // ITER_NUM_SCALE_FACTOR_SIM \
                if not eval_overhead else event.iter_num
            job = Job(
                {
                    "job_id": event.job_id,
                    "alias": event.job_alias,
                    # "user_id": "default_user",
                    # "vc_id": "default_vc",
                    "sub_time": event.sub_time,
                    "deadline": deadline,
                    "iter_num": iter_num,
                    "resource_quota": resource_quota,
                    "model_name": event.model_name,
                    "batch_size": event.batch_size
                }
            )
            self.scheduler.submit_job(job, self.scheduler.global_timer)
    
    def _get_sched_stats(self):
        """ 
        Get scheduler stats on average current scheduling overhead and 
        rescheduled job num. 
        """

        # Average sched overhead
        avg_sched_overhead, sched_num = 0, 0
        for job_id in self.scheduler.job_sched_overhead_table:
            avg_sched_overhead += np.sum(
                self.scheduler.job_sched_overhead_table[job_id]
            )
            sched_num += len(self.scheduler.job_sched_overhead_table[job_id])
        avg_sched_overhead = avg_sched_overhead / sched_num if sched_num > 0 else -1.0

        # Average resched job num
        avg_resched_num = np.sum([
            self.scheduler.resched_num_table[_job_id] 
                for _job_id in self.scheduler.resched_num_table
        ])
        job_num = len(list(self.scheduler.resched_num_table.keys()))
        avg_resched_num = avg_resched_num / job_num if job_num > 0 else -1

        print("")
        print(f"[I] Current global average scheduling overhead (s): {avg_sched_overhead}")
        print(f"[I] Current global average job rescheduling num: {avg_resched_num}")
        print("")

    def _update_and_store_sched_metrics(
        self,
        metrics: Dict[str, Any],
        result_dir: str, 
    ) -> None:
        """ Update runtime scheduling metrics and store. """

        # Update metrics
        self.thr_list.append(round(metrics["thr"], PREC))
        self.avg_job_thr_per_gpu_list.append(round(metrics["avg_job_thr_per_gpu"], PREC))
        self.jct_list.extend(metrics["jct_list"])
        self.queuing_time_list.extend(metrics["queue_time_list"])

        # Result dirs
        suffix = "" if self.trace_type == "philly" else f"_{self.trace_type}"
        thr_pth = path_join_and_check(result_dir, f"thr{suffix}")
        jct_pth = path_join_and_check(result_dir, f"jct{suffix}")
        queue_time_pth = path_join_and_check(result_dir, f"queuing_time{suffix}")
        resched_num_pth = path_join_and_check(result_dir, f"resched_num{suffix}")
        sched_overhead_pth = path_join_and_check(result_dir, f"sched_overhead{suffix}")
        
        # Store metrics
        np.save(os.path.join(thr_pth, f"{self.policy}_thr.npy"), self.thr_list)
        np.save(os.path.join(jct_pth, f"{self.policy}_jct.npy"), self.jct_list)
        np.save(os.path.join(queue_time_pth, f"{self.policy}_queuing_time.npy"), 
                self.queuing_time_list)
        np.save(os.path.join(resched_num_pth, f"{self.policy}_resched_num.npy"), 
                self.scheduler.resched_num_table)
        np.save(os.path.join(sched_overhead_pth, f"{self.policy}_sched_overhead.npy"), 
                self.scheduler.job_sched_overhead_table)

    def _record_metrics_unfinished_jobs(
        self,
        result_dir: str,
    ) -> None:
        """ 
        Record performance metrics of unfinished jobs. 

        For queuing time, we set to the time spent in pending queue for 
        each unfinished job.
        """

        queuing_time_list = list()
        for job in self.scheduler.running_job_queue + self.scheduler.pending_job_queue:
            # Queuing time
            assert job.uuid in self.scheduler.queue_time_table, \
                f"Job {job.alias} should be recorded in queue time table once " + \
                f"submitted to the scheduler."
            queuing_time_list.append(self.scheduler.queue_time_table[job.uuid])

        self.queuing_time_list.extend(queuing_time_list)

        # Store metrics
        suffix = "" if self.trace_type == "philly" else f"_{self.trace_type}"
        queue_time_pth = path_join_and_check(result_dir, f"queuing_time{suffix}")
        np.save(os.path.join(queue_time_pth, f"{self.policy}_queuing_time.npy"), 
                self.queuing_time_list)
    
    def _eval_trace_job_submit(
        self,
        arrival_trace: List[List[ArrivalEvent]],
    ) -> None:
        """ 
        Get the average number of submitted jobs in each non-empty round of the trace. 
        """

        num_submit_jobs_per_round = []
        for round_chunk in arrival_trace:
            if len(round_chunk) > 0:
                num_submit_jobs_per_round.append(len(round_chunk))
        
        print(f"[I] Maximum submitted job num in a round: {max(num_submit_jobs_per_round)}")
        print(f"[I] Average submitted job num in a round: {np.mean(num_submit_jobs_per_round)}")

    def simulate(
        self,
        result_dir: str = "./plot",
        max_sched_round: int = -1,
        eval_overhead: bool = False,
        eval_trace_job_submit: bool = False,
        use_dummy_trace: bool = False,
        no_res_stored: bool = False,
        init_global_index: int = 0,
        runout: bool = False,
    ) -> None:
        """ Simulation entry function. """
        
        # Workload trace
        if eval_overhead:
            # Evaluate scheduling overhead with extreme large workload
            arrival_trace = self._load_trace(
                num_skip_round=800,
                start_with_nonempty_round=True,
                use_dummy_trace=use_dummy_trace,
            )
        else:
            arrival_trace = self._load_trace(
                use_dummy_trace=use_dummy_trace,
            )
        
        if eval_trace_job_submit:
            return self._eval_trace_job_submit(arrival_trace)
        
        # Init global timer to the submission time
        self.scheduler.init_timer(timestamp=arrival_trace[0][0].sub_time)
        # Global sched index
        global_idx = 0

        if max_sched_round <= 0:
            # Overwrite max sched round
            max_sched_round = get_int_upper_bound(len(arrival_trace))
        
        if init_global_index > 0:
            # Start from a specific global index
            global_idx = init_global_index
            max_sched_round += init_global_index
        
        if no_res_stored:
            # No result stored
            print("[TMP] No simulation results will be stored.")
        else:
            # Result dir
            print(f"[TMP] Store simulation results in {result_dir}.")
            mkdir_if_not_exist(result_dir)
        
        # Main loop
        while True:
            # Verify exit condition
            if not runout and global_idx >= max_sched_round:
                # Max sched round arrived
                break
            elif (runout and (len(self.scheduler.running_job_queue) == 0 and 
                              len(self.scheduler.submit_init_job_queue) == 0 and 
                              len(self.scheduler.pending_job_queue) == 0) 
                  and global_idx >= len(arrival_trace)):
                # Run out all jobs
                break

            print("")
            print("")
            print("############################################################")
            print("#          Event-stream scheduling round:  %-05s           #" % global_idx)
            print("############################################################")
            
            print(f"[I] Runtime job status: {len(self.scheduler.running_job_queue)} " + 
                  f"in running | {len(self.scheduler.pending_job_queue)} in pending | " + 
                  f"{len(self.scheduler.in_profile_job_queue)} in profiling (or wait for it)")
            
            if global_idx < len(arrival_trace):
                # Submit arrived jobs
                self._submit_jobs(arrival_trace[global_idx], eval_overhead)
            
            # Schedule
            _metrics = self.scheduler.schedule()
            global_idx += 1

            # Scheduler stats include sched overhead and resched job num
            self._get_sched_stats()

            if not no_res_stored:
                # Update and store results
                self._update_and_store_sched_metrics(_metrics, result_dir)
        
        if not no_res_stored:
            # Record unfinished jobs
            self._record_metrics_unfinished_jobs(result_dir)
        
        if self.scheduler.enable_ddl:
            # Extended to ddl-aware
            print(f"[I] (for ablation) Totally {self.scheduler.ddl_satisfied_job_num} " + 
                  f"jobs are ddl-satisfied.")
            
        print(f"\n\n[I] (for rebuttal) Average job-level throughput per GPU: {np.mean(self.avg_job_thr_per_gpu_list)}")
        
        # Log OOM event statistics
        print(f"\n\n[I] (for motivation) OOM event statistics:")
        print(f" - OOM event count: {self.scheduler.db_querier._oom_event_count}")
        print(f" - Total event count: {self.scheduler.db_querier._all_event_count}")


def main():
    """ Main entrypoint. """

    if args.revision_mode:
        os.environ["CRIUS_REVISION_MODE"] = "true"
        if args.homogeneous:
            os.environ["CRIUS_HOMOGENEOUS_CLUSTER"] = "true"

    if args.visual:
        suffix = "" if args.trace_type == "philly" else f"_{args.trace_type}"
        if args.visualized_metric == "thr":
            plot_cluster_thr(
                work_dir=f"{args.result_dir}/thr{suffix}", 
                out_dir=args.out_dir,
                trace_type=args.trace_type,
                all_policies=args.all_policies,
            )
        elif args.visualized_metric == "queuing_time":
            # Visualize the queuing time
            plot_queuing_time(
                work_dir=f"{args.result_dir}/queuing_time{suffix}", 
                trace_type=args.trace_type,
            )
        elif args.visualized_metric == "jct":
            # Visualize and cal jct
            read_jct_data_and_plot(
                work_dir=f"{args.result_dir}/jct{suffix}",
                trace_type=args.trace_type,
                all_policies=args.all_policies,
            )
        else:
            raise ValueError(f"Invalid visualized metric: {args.visualized_metric}")
        
        return
    
    # Run simulation
    simulator = Simulator(
        policy=args.policy,
        trace_type=args.trace_type,
        enable_alpa=args.enable_alpa,
        dummy_test=args.dummy_test,
        sched_with_opt=args.sched_with_opt,
        job_submit_density=float(args.job_submit_density),
        prepend_profile_overhead=args.prepend_profile_overhead,
        disable_ap_prune=args.disable_ap_prune,
        disable_single_device_profiler=args.disable_single_device_profiler,
        verbose=args.verbose,
    )
    simulator.simulate(
        result_dir=args.result_dir,
        max_sched_round=args.max_sched_round,
        eval_overhead=args.eval_overhead,
        eval_trace_job_submit=args.eval_trace_job_submit,
        use_dummy_trace=args.use_dummy_trace,
        no_res_stored=args.no_res_stored,
        init_global_index=args.init_global_index,
        runout=args.runout,
    )


if __name__ == "__main__":
    
    # Args
    parser = argparse.ArgumentParser()
    # Common
    parser.add_argument("--policy", default="crius", type=str)
    parser.add_argument("--trace_type", default="philly", type=str)
    parser.add_argument("--max_sched_round", default=-1, type=int, 
                        help="Overwrite maximum scheduling round.")
    parser.add_argument("--enable_alpa", default=False, action='store_true', 
                        help="Whether to enable alpa search for the current policy.")
    parser.add_argument("--init_global_index", default=0, type=int, 
                        help="The global index in the trace to start simulation.")
    parser.add_argument("--result_dir", default="./plot", type=str)
    # Visualization
    parser.add_argument("--visual", default=False, action='store_true', 
                        help="Visualize the simulation results.")
    parser.add_argument("--visualized_metric", default="thr", type=str)
    parser.add_argument("--out_dir", default="./figures", type=str)
    parser.add_argument("--all_policies", default=False, action='store_true', 
                        help="Visualize all policies including ddl-aware and so on.")
    # Ablation
    parser.add_argument("--job_submit_density", default=1.0, type=float, 
                        help="The density of job submission event, functioned as scaling " + 
                             "job iteration num in the trace (larger is more dense).")
    parser.add_argument("--prepend_profile_overhead", default=False, action='store_true', 
                        help="Whether to prepend the profile overhead to each job " + 
                             "once submitted to the scheduler.")
    parser.add_argument("--disable_ap_prune", default=False, action='store_true', 
                        help="Whether to disable AP pruning for Crius.")
    parser.add_argument("--disable_single_device_profiler", default=False, action='store_true', 
                        help="Whether to disable single GPU profiler for Crius.")
    # Dummy test & eval settings
    parser.add_argument("--dummy_test", default=False, action='store_true', 
                        help="Run dummy test with dummy data rather than profiled data.")
    parser.add_argument("--use_dummy_trace", default=False, action='store_true', 
                        help="Run scheduling with dummy trace.")
    parser.add_argument("--runout", default=False, action='store_true', 
                        help="Whether to run out all jobs in the trace.")
    parser.add_argument("--sched_with_opt", default=False, action='store_true', 
                        help="Run scheduler and make scheduling decisions with the " + 
                             "profiled throughput of optimal parallelism.")
    parser.add_argument("--eval_overhead", default=False, action='store_true', 
                        help="Whether to run extreme heavy test to evaluate scheduling overhead.")
    parser.add_argument("--eval_trace_job_submit", default=False, action='store_true', 
                        help="Evaluate the average number of job submission in the trace.")
    parser.add_argument("--no_res_stored", default=False, action='store_true', 
                        help="Run the simulation without storing results.")
    # Other settings
    parser.add_argument("--verbose", default=False, action='store_true', 
                        help="Run the simulation within debug mode.")
    # For revision
    parser.add_argument("--revision_mode", default=False, action='store_true',
                        help="Run the simulator in revision mode.")
    parser.add_argument("--homogeneous", default=False, action='store_true',
                        help="Run the simulator with a homogeneous cluster in revision mode.")
    args = parser.parse_args()
    
    main()
