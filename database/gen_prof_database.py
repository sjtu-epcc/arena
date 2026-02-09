#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
A script related to load all data in pre-processed pickle file and construct 
global pickle database. 
This script should be called in the main dir of 'Crius'.
-----------------------------------------------------------
The structure of the global profiling database:
    --> Level 0 key 1: all (for generating scheduling decisions)
        --> Level 1 key: {model_name}_{param_num}_{batch size}
            --> Level 2 key: {devices_name}_{num_hosts}_{num_devices_per_host}
                --> Level 3 key: prallelism degrees: (#pp, #dp, #mp)
                    -----> Value: ([1] computation time, [2] intra-stage communication time, 
                                   [3] inter-stages communication time, [4] e2e iteration time)

    --> Level 0 key 2: optimal (only for evaluate throughput in simulation)
        --> Level 1 key: {model_name}_{param_num}_{batch size}
            --> Level 2 key: {devices_name}_{num_hosts}_{num_devices_per_host}
                -----> Value: iteration time of optimal parallelism
"""

import os
import argparse
import pickle
from typing import Sequence, Any
import numpy as np
import json

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resources.hardware_specs import (
    supported_model_cfgs, NODE_CAPACITY)
from macro.macro_def import (MAX_GPU_NUM_TABLE, CAND_PARALLEL_FLAGS)


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--gen_optimal", default=False, action='store_true')
parser.add_argument("--gen_estimate", default=False, action='store_true')
parser.add_argument("--gen_optimal_revision", default=False, action='store_true')
args = parser.parse_args()


def _parse_l1_key(key: str):
    """ Parse l1 key to get model name, param num and batch size. """
    strs = key.split("_")
    if "wide_resnet" in key:
        model_name = strs[0] + "_" + strs[1]
        param_num = strs[2]
        batch_size = int(strs[3])
    else:
        model_name, param_num = strs[0], strs[1]
        batch_size = int(strs[2])
    return f"{model_name}_{param_num}_{batch_size}"


def _parse_l2_key(key: str):
    """ 
    Parse l2 key to get devices name, num hosts and num devices per host. 
    """
    strs = key.split("_")
    devices_name = strs[0] + "_" + strs[1]
    num_hosts = int(strs[2])
    num_devices_per_host = int(strs[4])
    return f"{devices_name}_{num_hosts}_n_{num_devices_per_host}_d"


def _process_optimal(optimal_data_pth: str, prof_database: dict):
    """ Process profiled data with optimal parallelism. """
    l0_key = "optimal"
    prof_database[l0_key] = dict()
    # Read pre-processed pickle data
    with open(optimal_data_pth, "rb") as f:
        prof_database_opt = pickle.load(f)

    for _l1_ley in prof_database_opt:
        l1_key = _parse_l1_key(_l1_ley)
        prof_database[l0_key][l1_key] = dict()
        
        print(_l1_ley)
        print(prof_database_opt[_l1_ley].keys())
        print("")

        for _l2_key in prof_database_opt[_l1_ley]:
            l2_key = _parse_l2_key(_l2_key)
            prof_database[l0_key][l1_key][l2_key] = prof_database_opt[_l1_ley][_l2_key]

    exit(0)

    return prof_database


def _process_estimate(estimate_data_pth: str, prof_database: dict):
    """ Process estimated data with crius profiler. """
    l0_key = "all"
    prof_database[l0_key] = dict()
    # Read pre-processed pickle data
    with open(estimate_data_pth, "rb") as f:
        prof_database_est = pickle.load(f)
    
    for _l1_ley in prof_database_est:
        l1_key = _parse_l1_key(_l1_ley)
        prof_database[l0_key][l1_key] = dict()
        for _l2_key in prof_database_est[_l1_ley]:
            l2_key = _parse_l2_key(_l2_key)
            prof_database[l0_key][l1_key][l2_key] = dict()
            for _rec in prof_database_est[_l1_ley][_l2_key]:
                _para_degrees = _rec[0]
                prof_database[l0_key][l1_key][l2_key][_para_degrees] = _rec[1:]

            print(prof_database_est[_l1_ley][_l2_key])
            
            print(prof_database[l0_key][l1_key][l2_key])

            exit(0)

    return prof_database


def process_pickle_files():
    """ Load all pre-processed pickle files, store as the global pickle database. """
    prof_database = dict()
    
    if args.gen_optimal:
        # Optimal
        _optimal_data_pth = "./runtime/jaxpr/prof_log/optimal/prof_database_opt.pkl"
        prof_database = _process_optimal(_optimal_data_pth, prof_database)

    if args.gen_estimate:
        # Estimate
        _estimate_data_pth = "./runtime/jaxpr/prof_log/estimate_all/prof_database_est.pkl"
        prof_database = _process_estimate(_estimate_data_pth, prof_database)
    
    # Store as pickle
    pth = os.environ.get("CRIUS_PROF_DB_PATH")
    print(f"[TMP] Storing the generated profiling database in `{pth}`...")
    with open(pth, "wb") as f:
        pickle.dump(prof_database, f)


def process_json_files():
    """ Load all pre-processed json files, store as the global pickle database. """
    prof_database = {}
    
    if args.gen_optimal_revision:
        # Optimal revision
        optimal_data_pth = "./runtime/jaxpr/prof_log/optimal_revision/prof_database_opt.json"
        l0_key = "optimal"
        prof_database[l0_key] = {}

        # Read pre-processed pickle data
        with open(optimal_data_pth, "r") as f:
            prof_database_opt = json.load(f)

        for l1_ley in prof_database_opt:
            l1_key = _parse_l1_key(l1_ley)
            prof_database[l0_key][l1_key] = dict()
            
            print(l1_ley)
            print(prof_database_opt[l1_ley].keys())
            print("")

            for _l2_key in prof_database_opt[l1_ley]:
                l2_key = _parse_l2_key(_l2_key)
                prof_database[l0_key][l1_key][l2_key] = prof_database_opt[l1_ley][_l2_key]

    # Store as pickle
    pth = os.environ.get("CRIUS_PROF_DB_REVISION_PATH")
    print(f"[TMP] Storing the generated profiling database in `{pth}`...")
    with open(pth, "w") as f:
        json.dump(prof_database, f)    


def main():
    """ Entrypoint. """
    # Environmental variables
    os.environ["CRIUS_PROF_DB_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof_database.pkl")
    os.environ["CRIUS_PROF_DB_REVISION_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prof_database_revision.json")

    if args.gen_optimal_revision:
        process_json_files()
        return

    process_pickle_files()


if __name__ == "__main__":
    main()
