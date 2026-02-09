#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to the querier to interact with the profiling database.
"""

import os
from typing import Sequence, Any, Tuple
from dataclasses import dataclass
import pickle
import json
import numpy as np

from resources.hardware_specs import (
    NODE_CAPACITY, gpu_specs_suite, pre_alloc_memory_fraction_for_xla,
    supported_model_cfgs)
from utils import read_json_content
from macro.macro_def import (
    INFEASIBLE_THR, SCHEDULING_INTERVAL, 
    RUNTIME_SCHEDULING_INTERVAL, PREC, LOCALITY_SCALE_FACTOR,
    PARAM_NUM_SCALE_FACTOR)


@dataclass
class QueryConfigs:
    """ Dataclass of one query towards profiling database. """
    num_hosts: int
    num_devices_per_host: int = None
    gpu_type: str = None
    model_name: str = None
    param_num: str = None
    batch_size: int = None
    # Only query profiling data of optimized parallelism (default: False)
    only_opt: bool = False
    # Only query profiling data of vanilla data parallelism (default: False)
    only_dp: bool = False
    # Search currently profiled maximum throughput of the target job among all 
    # parallelisms, and return a list of ProfData that records profiled data 
    # with all parallelisms (default: False)
    search_max: bool = False


@dataclass
class ProfData:
    """ Dataclass of one entry in profiled data. """
    e2e_iter_time: float = None
    comp_time: float = None
    intra_stage_comm_time: float = None
    inter_stages_comm_time: float = None
    parallel_degrees: Tuple[int] = None


class DatabaseQuerier:
    """ The class of database querier, which provide APIs to query profiling database. """
    def __init__(self, runtime_sched: bool = False):
        self.prof_db = self._load_prof_database()
        self.dp_mem_table = self._load_dp_mem()
        self.runtime_sched = runtime_sched

        self._oom_event_count = 0
        self._all_event_count = 0
    
    def _load_prof_database(self):
        """ Load global profiling database. """
        prof_db = None
        prof_db_pth = os.environ.get("CRIUS_PROF_DB_PATH", "./database/prof_database.pkl")
        
        if os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
            # Value: [niter, e2e_iter_time, avg_lat, local_total_time, local_avg_lat, max_mem_gb, ]
            with open(prof_db_pth, "r") as f:
                prof_db = json.load(f)
                return prof_db

        with open(prof_db_pth, "rb") as f:
            prof_db = pickle.load(f)
        return prof_db
    
    def _load_dp_mem(self):
        """ Load comsumed GPU memory data of data parallelism. """
        _pth = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "database/dp_mem.json"
        )
        return read_json_content(_pth)[0]

    def query_db(self, query_cfgs: QueryConfigs, force_opt: bool = False):
        """ 
        API of querying profiling data with specified querying configurations. 

        Args:
            query_cfgs: Query configurations.
            force_opt: Skip OOM check of data parallelism. This option should NOT be set for baseline schedulers as 
                       they use DP data to schedule.
        
        Return:
            is_fsb: Whether the queried data is feasible.
            thr: Profiled throughput of the query.
            prof_datas: A list of ProfData that record detailed profiling data.
        """

        def __gen_l1_l2_keys(_query_cfgs: QueryConfigs, l0_key: str):
            """ Generate l1 and l2 keys. """
            l1_key = f"{_query_cfgs.model_name}_{_query_cfgs.param_num}_{_query_cfgs.batch_size}"
            l2_key = f"{_query_cfgs.num_hosts}_{_query_cfgs.gpu_type}_{_query_cfgs.num_hosts}_n_" + \
                     f"{_query_cfgs.num_devices_per_host}_d"

            if l1_key in self.prof_db[l0_key] and l2_key in self.prof_db[l0_key][l1_key]:
                # Target configuration is found in profiling database
                # print(f"[TMP] Target configuration is found in profiling database.")
                return l1_key, l2_key, None, None

            # Probably with relaxed locality, estimate with it best locality
            node_cap = NODE_CAPACITY[_query_cfgs.gpu_type]
            num_devices = _query_cfgs.num_hosts * _query_cfgs.num_devices_per_host
            best_locality = [num_devices] if num_devices <= node_cap else [node_cap for _ in range(num_devices // node_cap)]
            mgrt_l2_key = f"{len(best_locality)}_{_query_cfgs.gpu_type}_{len(best_locality)}_n_{best_locality[0]}_d"

            if l1_key in self.prof_db[l0_key] and mgrt_l2_key in self.prof_db[l0_key][l1_key]:
                # print(f"[TMP] Target configuration with best locality {best_locality} is found in profiling database.")
                # Migration times from originally relaxed locality to the best locality
                mgrt_factor = int(np.log2(best_locality[0] // _query_cfgs.num_devices_per_host))
                # Target configuration with its best locality is found in profiling database
                return l1_key, mgrt_l2_key, mgrt_factor, None

            # Probably with larger parameter size
            param_idx = supported_model_cfgs[_query_cfgs.model_name]["param_num"].index(_query_cfgs.param_num)
            for _i, _param_num in enumerate(supported_model_cfgs[_query_cfgs.model_name]["param_num"]):
                if _i <= param_idx:
                    continue
                grow_l1_key = f"{_query_cfgs.model_name}_{_param_num}_{_query_cfgs.batch_size}"
                if grow_l1_key in self.prof_db[l0_key] and l2_key in self.prof_db[l0_key][grow_l1_key]:
                    # print(f"[TMP] Target configuration with grown param num {_param_num} is found in profiling database.")
                    # Grow times form original param num to target param num
                    grow_factor = _i - param_idx
                    # Target configuration with grown param num is found in profiling database
                    return grow_l1_key, l2_key, None, grow_factor

            # Target configuration (even with its best locality or grown param num) not found in profiling database
            return None, None, None, None

        def __valid_iter_time(avg_e2e_iter_time: float):
            """ Validate iteration time. """
            _sched_interval = SCHEDULING_INTERVAL if not self.runtime_sched else RUNTIME_SCHEDULING_INTERVAL
            return (avg_e2e_iter_time and avg_e2e_iter_time <= _sched_interval)
        
        def __gen_thr(iter_time: float, batch_size: int):
            """ Generate throughput. """
            return round(batch_size / float(iter_time), PREC) if iter_time else 0.0 
        
        def __parse_para_degrees(para_degrees: str):
            """ Parse str-style parallel degrees into tuple. """
            return tuple(
                [int(_c) for _c in para_degrees.replace("(", "").replace(")", "").replace(" ", "").split(",")]
            )
        
        self._all_event_count += 1

        sched_policy = os.environ.get("SCHED_POLICY", "crius")
        if (
            not force_opt and 
            (sched_policy.split("-")[0] != "crius" or sched_policy == "crius-dp")   # Only check OOM for non-Crius baselines, or DP-aware Crius
        ):
            # Compare the needed memory of one DP replica with the available per-GPU memory
            max_mem = gpu_specs_suite[query_cfgs.gpu_type].max_mem
            _key = f"{query_cfgs.model_name}_{query_cfgs.param_num}_{query_cfgs.batch_size}"

            if not os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                if _key in self.dp_mem_table:
                    gpu_num_list = [pow(2, _i) for _i in range(len(self.dp_mem_table[_key]))]
                    _gpu_num_idx = gpu_num_list.index(query_cfgs.num_hosts * query_cfgs.num_devices_per_host)
                    needed_mem = self.dp_mem_table[_key][_gpu_num_idx]    
                    if needed_mem != 0 and (needed_mem == -1 or 
                                            needed_mem * 1.2 > max_mem * pre_alloc_memory_fraction_for_xla):
                        # Out-of-memory
                        # print(
                        #     f"[WARN] OOM would occur in: model name: {query_cfgs.model_name} | param num: {query_cfgs.param_num} | " + 
                        #     f"GPU type: {query_cfgs.gpu_type} | GPU num: {query_cfgs.num_hosts * query_cfgs.num_devices_per_host}"
                        # )
                        self._oom_event_count += 1
                        return False, INFEASIBLE_THR, None
                else:
                    # Not profiled memory size of dp, must be too large for dp
                    self._oom_event_count += 1
                    return False, INFEASIBLE_THR, None
            
            else:
                # In revision mode, we only estimate DP memory from that of the optimal parallelism
                dp_to_ap_mem_ratio = 1.2   # Assume DP needs 1.5x memory of AP
                
                model_key = _key
                device_key = f"{query_cfgs.num_hosts}_{query_cfgs.gpu_type}_{query_cfgs.num_hosts}_n_{query_cfgs.num_devices_per_host}_d"
                if model_key not in self.prof_db["optimal"] or device_key not in self.prof_db["optimal"][model_key]:
                    # Not found in profiling database
                    # print(f"[WARN] Not found in prof database for OOM check: {model_key} | {device_key}")
                    self._oom_event_count += 1
                    return False, INFEASIBLE_THR, None

                needed_mem_ap = self.prof_db["optimal"][model_key][device_key][-1]
                if (
                    needed_mem_ap <= 0 or 
                    needed_mem_ap * dp_to_ap_mem_ratio * 1.2 > max_mem * pre_alloc_memory_fraction_for_xla
                ):
                    # Out-of-memory
                    self._oom_event_count += 1
                    return False, INFEASIBLE_THR, None

        if query_cfgs.only_opt:
            # Query data of optimal parallelism to evaluate performance in simulation
            # assert not self.runtime_sched, \
            #     "Only support querying optimal data in simulation for performance evaluation."
            l0_key = "optimal"
            l1_key, l2_key, mgrt_factor, grow_factor = __gen_l1_l2_keys(query_cfgs, l0_key)
            
            if not l1_key:   # Not found in profiling database even with best locality and grown param num
                return False, INFEASIBLE_THR, None

            # Load data
            if not os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                iter_t = self.prof_db[l0_key][l1_key][l2_key]
            else:
                (niter, e2e_iter_time, avg_lat, local_total_time, local_avg_lat, max_mem_gb) = self.prof_db[l0_key][l1_key][l2_key]
                iter_t = e2e_iter_time

            if not __valid_iter_time(iter_t):
                return False, INFEASIBLE_THR, None
            thr = __gen_thr(iter_t, query_cfgs.batch_size)
            if mgrt_factor:
                # Found with best locality, need to estimate throughput with 
                # the orginal relaxed locality by simply scaling.
                thr *= pow(LOCALITY_SCALE_FACTOR, mgrt_factor)
            if grow_factor:
                # Found with grown param num, need to estimate throughput with 
                # the orginal param num by simply scaling.
                thr *= pow(PARAM_NUM_SCALE_FACTOR, grow_factor)
            
            if thr <= 0:
                # OOM
                # print(
                #     f"[WARN] OOM would occur in: model name: {query_cfgs.model_name} | param num: {query_cfgs.param_num} | " + 
                #     f"GPU type: {query_cfgs.gpu_type} | GPU num: {query_cfgs.num_hosts * query_cfgs.num_devices_per_host}"
                # )
                self._oom_event_count += 1
            
            return (thr > 0), thr, None
        
        elif query_cfgs.only_dp:
            # Query data of data parallelism to schedule baseline strategies
            l0_key = "all" if "all" in self.prof_db else "optimal"   # In case that DP-only is not profiled
            l1_key, l2_key, _, _ = __gen_l1_l2_keys(query_cfgs, l0_key)
            
            if not l1_key:
                # Not found in profiling database
                return False, INFEASIBLE_THR, None
            
            _num_devices = query_cfgs.num_hosts * query_cfgs.num_devices_per_host
            para_degrees = (1, _num_devices, 1)  # Data parallelism only 
            l3_key = str(para_degrees)
            # Load data
            if l0_key == "all":
                assert not os.environ.get("CRIUS_REVISION_MODE", "false") == "true"
                (comp_t, intra_stage_comm_t, 
                inter_stages_comm_t, iter_t) = self.prof_db[l0_key][l1_key][l2_key][l3_key]
            else:
                # NOTE(chunyu): Only accessed when no DP-only data available
                if not os.environ.get("CRIUS_REVISION_MODE", "false") == "true":
                    iter_t = self.prof_db[l0_key][l1_key][l2_key]
                else:
                    (niter, e2e_iter_time, avg_lat, local_total_time, local_avg_lat, max_mem_gb) = self.prof_db[l0_key][l1_key][l2_key]
                    iter_t = e2e_iter_time

                comp_t, intra_stage_comm_t, inter_stages_comm_t = None, None, None

            if not __valid_iter_time(iter_t):
                return False, INFEASIBLE_THR, None
            thr = __gen_thr(iter_t, query_cfgs.batch_size)
            
            if thr <= 0:
                # OOM
                # print(
                #     f"[WARN] OOM would occur in: model name: {query_cfgs.model_name} | param num: {query_cfgs.param_num} | " + 
                #     f"GPU type: {query_cfgs.gpu_type} | GPU num: {query_cfgs.num_hosts * query_cfgs.num_devices_per_host}"
                # )
                self._oom_event_count += 1
            
            return (thr > 0), thr, [ProfData(iter_t, comp_t, intra_stage_comm_t, 
                                            inter_stages_comm_t, para_degrees)]
        
        elif query_cfgs.search_max:
            # Obtain profiled data of all parallelisms to (1) schedule crius strategies 
            # with the max thr and (2) prune search space of the tuner with data of multiple 
            # parallelism degrees.
            l0_key = "all" if "all" in self.prof_db else "optimal"   # In case that DP-only is not profiled
            l1_key, l2_key, mgrt_factor, grow_factor = __gen_l1_l2_keys(query_cfgs, l0_key)
            if not l1_key:
                # Not found in profiling database even with best locality
                return False, INFEASIBLE_THR, None
            
            if isinstance(self.prof_db[l0_key][l1_key][l2_key], (int, float)):
                # Only one parallelism degree
                iter_t = self.prof_db[l0_key][l1_key][l2_key]
                if not __valid_iter_time(iter_t):
                    return False, INFEASIBLE_THR, None
                thr = __gen_thr(iter_t, query_cfgs.batch_size)
                if mgrt_factor:
                    # Found with best locality, need to estimate throughput with 
                    # the orginal relaxed locality by simply scaling.
                    thr *= pow(LOCALITY_SCALE_FACTOR, mgrt_factor)
                if grow_factor:
                    # Found with grown param num, need to estimate throughput with 
                    # the orginal param num by simply scaling.
                    thr *= pow(PARAM_NUM_SCALE_FACTOR, grow_factor)
                
                if thr <= 0:
                    # OOM
                    # print(
                    #     f"[WARN] OOM would occur in: model name: {query_cfgs.model_name} | param num: {query_cfgs.param_num} | " + 
                    #     f"GPU type: {query_cfgs.gpu_type} | GPU num: {query_cfgs.num_hosts * query_cfgs.num_devices_per_host}"
                    # )
                    self._oom_event_count += 1
                
                return (thr > 0), thr, None

            # Traverse prof database
            min_iter_t = 1e9
            prof_datas = list()
            for _l3_key in self.prof_db[l0_key][l1_key][l2_key]:
                # Parallel degrees
                para_degrees = __parse_para_degrees(_l3_key)
                # Load data
                (comp_t, intra_stage_comm_t, 
                 inter_stages_comm_t, iter_t) = self.prof_db[l0_key][l1_key][l2_key][_l3_key]
                if __valid_iter_time(iter_t):
                    min_iter_t = min(min_iter_t, iter_t)   
                    prof_datas.append(
                        ProfData(iter_t, comp_t, intra_stage_comm_t, inter_stages_comm_t,
                                para_degrees)
                    )
            if len(prof_datas) == 0:
                # No valid iter time
                return False, INFEASIBLE_THR, None
            thr = __gen_thr(min_iter_t, query_cfgs.batch_size)
            if mgrt_factor:
                # Found with best locality, need to estimate throughput with 
                # the orginal relaxed locality by simply scaling.
                thr *= pow(LOCALITY_SCALE_FACTOR, mgrt_factor)
            if grow_factor:
                # Found with grown param num, need to estimate throughput with 
                # the orginal param num by simply scaling.
                thr *= pow(PARAM_NUM_SCALE_FACTOR, grow_factor)
            return (thr > 0), thr, prof_datas
        else:
            raise RuntimeError("Please specify one mode (e.g., only_opt) in querying configurations.")


def test_query():
    """ Dummy test. """
    os.environ["CRIUS_PROF_DB_PATH"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "database/prof_database.pkl"
    )
    querier = DatabaseQuerier()
    query_cfgs = QueryConfigs(1, 4, "v100", "wide_resnet", "500M", 256, only_dp=True)
    is_feasible, thr, prof_datas = querier.query_db(query_cfgs)
    print(is_feasible, thr, prof_datas)


# test_query()
