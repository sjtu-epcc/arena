#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to process all execution logs in raw data (.out from slurm) and construct pickle database. """

import os
import argparse
import pickle
from typing import Sequence, Any
import numpy as np

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    remove_all
)

INCOMPLETE_GROUP_LEN = 100


def _match_target_patterns(lines: Sequence[Any], comm_time_table: dict):
    """ Match target patterns from the loaded content. """
    lines = [_line.decode() for _line in lines]

    for _line in lines:
        if ") OP type:" in _line:
            # Op shape
            op_type = _line.split("|")[0].split(": ")[1].replace(" ", "")
            # Data shape
            shape = _line.split("|")[1].split(": ")[1].replace(" ", "")
            _tmp = remove_all(shape.split(","), ["", ")"])
            _tmp = [_c.replace("(", "") for _c in _tmp]
            assert len(_tmp) == 1, f"{_tmp}"
            shape = (int(_tmp[0]),)
            # Dtype
            dtype = "float32"
            # Replica groups
            replica_groups = _line.split("|")[3].split(": ")[1].replace(" ", "")
            _tmp = [_c.replace("[[", "").replace("]]", "") for _c in replica_groups.split("],[")]
            replica_groups = list()
            for _c in _tmp:
                replica_groups.append([int(_s) for _s in _c.split(",")])
            # Comm time
            comm_time = float(_line.split("|")[-1].split(": ")[1].replace(" ", "").replace("\n", ""))

            _key = str((op_type, replica_groups))
            if _key not in comm_time_table:
                comm_time_table[_key] = [[shape, dtype, comm_time, replica_groups]]
            else:
                comm_time_table[_key].append([shape, dtype, comm_time, replica_groups])
    
    to_del_keys = list()
    for _key in comm_time_table.keys():
        if len(comm_time_table[_key]) < INCOMPLETE_GROUP_LEN:
            to_del_keys.append(_key)
    
    print(to_del_keys)
    print(list(comm_time_table.keys()))
    
    for _key in to_del_keys:
        del comm_time_table[_key]
    
    print(list(comm_time_table.keys()))


def _parse_one_log(log_path: str, comm_time_table: dict):
    """ Parse the content of one log file. """
    lines = None
    with open(log_path, "rb") as f:
        lines = f.readlines()
    return _match_target_patterns(lines, comm_time_table) if lines else None


def process():
    """ Load all log files, parse content in each file and store as pickle database. """
    # Database
    comm_time_table = dict()  # _key -> [shape, dtype, comm_time, replica_groups]
    
    # Parse profiled raw data
    _workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
    for _l1_fn in os.listdir(_workdir):
        print(_l1_fn)
        _log_pth = os.path.join(_workdir, _l1_fn)
        _parse_one_log(_log_pth, comm_time_table)
    
    # Store as pickle
    prof_db_pth = os.environ.get("COMM_PROF_DB_PATH")
    print(f"[TMP] All profiled logfiles in `./raw_data` have been parsed and saved in `{prof_db_pth}`.")
    with open(prof_db_pth, "wb") as f:
        pickle.dump(comm_time_table, f)


def main():
    """ Entrypoint. """
    # Environmental variables
    os.environ["COMM_PROF_DB_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1_v100_1_n_16_d.pkl")

    process()


if __name__ == "__main__":
    main()
