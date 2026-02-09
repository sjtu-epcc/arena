#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to process all profiling results. """

import os
import argparse
import pickle
from typing import Sequence, Any
import numpy as np
import json


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--merge_optimal", default=False, action='store_true')
parser.add_argument("--merge_estimate", default=False, action='store_true')
parser.add_argument("--merge_optimal_revision", default=False, action='store_true')
args = parser.parse_args()


def merge(l1_dir: str, out_pth: str):
    """ 
    Merge profiling results from varying gpu types and export as one pickle file. 
    """
    merged_results = dict()

    for _l1_fn in os.listdir(l1_dir):
        _l2_dir = os.path.join(l1_dir, _l1_fn)
        for _l2_fn in os.listdir(_l2_dir):
            # Load pickle
            _pth = os.path.join(_l2_dir, _l2_fn)
            with open(_pth, "rb") as f:
                _results = pickle.load(f)
            key = _l2_fn.split(".pkl")[0]
            if key not in merged_results.keys():
                merged_results[key] = dict()
            # Add
            for _subkey in _results:
                assert _subkey not in merged_results[key].keys(), \
                    f"Subkey {_subkey} should not be in recorded content for key {key}."
                merged_results[key][_subkey] = _results[_subkey]
            
    for _key in merged_results:
        print("")
        print(f"[TMP] Model info: {_key} | Profiling info: {merged_results[_key]}")
    
    # Store as pickle
    print("")
    print(f"[TMP] Writing merged profiling data to '{out_pth}'...")
    with open(out_pth, "wb") as f:
        pickle.dump(merged_results, f)


def merge_json(l1_dir: str, out_pth: str):
    """ 
    Merge profiling results from varying gpu types and export as one json file. 
    """
    merged_results = dict()

    for _l1_fn in os.listdir(l1_dir):
        _l2_dir = os.path.join(l1_dir, _l1_fn)
        for _l2_fn in os.listdir(_l2_dir):
            
            # Load json
            _pth = os.path.join(_l2_dir, _l2_fn)
            with open(_pth, "r") as f:
                _results = json.load(f)
            key = _l2_fn.split(".json")[0]
            if key not in merged_results.keys():
                merged_results[key] = dict()
            # Add
            for _subkey in _results:
                assert _subkey not in merged_results[key].keys(), \
                    f"Subkey {_subkey} should not be in recorded content for key {key}."
                merged_results[key][_subkey] = _results[_subkey]
     
    for _key in merged_results:
        print(f"\n[TMP] Model info: {_key} | Profiling info: {merged_results[_key]}")
    
    print(f"\nTotally {len(merged_results)} model configurations merged.")

    # Store as json
    print(f"\n[TMP] Writing merged profiling data to '{out_pth}'...")
    with open(out_pth, "w") as f:
        json.dump(merged_results, f)


def main():
    """ Entrypoint. """
    base_dir = "./jaxpr/prof_log"

    if args.merge_optimal_revision:
        # Merge optimal revision
        l1_dir = os.path.join(base_dir, "optimal_revision/raw_data")
        out_pth = os.path.join(base_dir, "optimal_revision/prof_database_opt.json")
        merge_json(l1_dir, out_pth)

    if args.merge_optimal:
        # Merge optimal
        l1_dir = os.path.join(base_dir, "optimal/raw_data")
        out_pth = os.path.join(base_dir, "optimal/prof_database_opt.pkl")
        merge(l1_dir, out_pth)

    if args.merge_estimate:
        # Merge estimate
        l1_dir = os.path.join(base_dir, "estimate_all/raw_data")
        out_pth = os.path.join(base_dir, "estimate_all/prof_database_est.pkl")
        merge(l1_dir, out_pth)


if __name__ == "__main__":
    main()
