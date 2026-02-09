#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" A script related to parse the profiling result of Nsight System, including PCIe bandwidth and SM active ratio. """

import os
import argparse
from typing import Sequence, Any
from collections import namedtuple
import numpy as np
# from scipy import stats

# Args 
parser = argparse.ArgumentParser()
parser.add_argument("--parallelism_type", default='dp', type=str, choices=['dp', 'mp', 'pp'])
parser.add_argument("--iter_num", default=-1, type=int)
parser.add_argument("--micro_batch_num_per_iter", default=-1, type=int)
args = parser.parse_args()
assert (args.iter_num > 0) and (args.micro_batch_num_per_iter > 0)

"""
Macro definitions.
"""
# Scale factor of the PCIe throughput from the idle to the used throughput ratio.
THR_SCALE = 10
# Scale factor of the SM active ratio from the idle to the used SM active ratio.
SM_SCALE = 20
# Decide whether the current usage is consistantly stable
FUR_LEN = 10
# Decide whether the current usage is consistantly stable for pipeline
FUR_LEN_PP = 5
# Decide whether the current SM active ratio is consistantly stable
FUR_LEN_SM = 10
# Minimum length of the valid interval, messured in the number of timestamp
MIN_LEN = 3
# Bias threshold of the communication amount when determining the stability of nearing intervals
AMT_BIAS = 0.2
# Scale factor of the surging point, in which the comm_amt of the current interval is smaller than the largest one
SURGE_FACTOR = 3

"""
A collection of Nsys GPU event.
-------------------------------------------------------------------
Entries:
    - timestamp: Timestamp of the event (ns).
    - pcie_ratio: PCIe ratio of the event (percentage).
"""
Event = namedtuple('Event', [
    'timestamp', 'pcie_ratio', 'sm_ratio',
])


class Parser:
    """ The class of the parser to analyse the profiling result of model training on GPU. """
    def __init__(self):
        self.parallelism = args.parallelism_type
    
    ######################################
    #       Data Related Functions       #
    ######################################
    
    def constr_db(self, rep_pth: str, db_pth: str):
        """ Construct SQLite database from the target Nsys report. """
        # Shell cmd
        _cmd = [
            f"nsys export -t sqlite -o {db_pth} -f true {rep_pth}",
            "bash ./query_db.sh",
        ]
        os.system(f"{_cmd[0]}; {_cmd[1]}")

    def read_pcie_thr_data(self, txt_pth):
        """ Read PCIe RX Throughput data. """
        # Raw data
        raw_data = list()
        with open(txt_pth, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                _event = Event(timestamp=int(line.replace("\n", "").split("|")[0]), pcie_ratio=int(line.replace("\n", "").split("|")[1]), sm_ratio=None)
                raw_data.append(_event)
            file.close()
        # Record events for each GPU
        gpu_data_list, _data = list(), list()
        for _event in raw_data:
            if _event.timestamp > 0:
                _data.append(_event)
            elif len(_data) > 0:
                gpu_data_list.append(_data)
                _data = list()
        # Add events for the last GPU
        if len(_data) > 0:
            gpu_data_list.append(_data)
        return gpu_data_list
    
    def read_sm_active_data(self, txt_pth):
        """ Read SM Active Ratio data. """
        # Raw data
        raw_data = list()
        with open(txt_pth, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                _event = Event(timestamp=int(line.replace("\n", "").split("|")[0]), pcie_ratio=None, sm_ratio=int(line.replace("\n", "").split("|")[1]))
                raw_data.append(_event)
            file.close()
        # Record events for each GPU
        gpu_data_list, _data = list(), list()
        for _event in raw_data:
            if _event.timestamp > 0:
                _data.append(_event)
            elif len(_data) > 0:
                gpu_data_list.append(_data)
                _data = list()
        # Add events for the last GPU
        if len(_data) > 0:
            gpu_data_list.append(_data)
        return gpu_data_list
    
    ######################################
    #  Interval Filter Related Functions #
    ######################################

    def is_et_reached(self, data: Sequence[Any], ptr: int, base_ratio: int, data_type: str = 'pcie'):
        """ Check whether the end timestamp of current interval is reached. """
        _cnt = 0
        _len = FUR_LEN if data_type == 'pcie' else FUR_LEN_SM
        if self.parallelism == 'pp':
            # Relatively smaller communication amount
            _len = FUR_LEN_PP
        _scale = THR_SCALE if data_type == 'pcie' else SM_SCALE
        while ptr < len(data):
            _ratio = data[ptr].pcie_ratio if data_type == 'pcie' else data[ptr].sm_ratio
            if _ratio >= _scale * base_ratio and ptr < len(data) - 1:
                return False
            # Count
            _cnt += 1
            if (ptr == len(data) - 1) or (_cnt == _len):
                return True
            # Update
            ptr += 1

    def is_abnormal_interval(self, interval: Sequence[Any], data_type: str = 'pcie'):
        """ Check whether the interval is abnormal (long consistant same value). """
        _same_cnt = 0
        for _event in interval:
            _i = interval.index(_event)
            if _i == len(interval) - 1:
                return False
            (_ratio, _next_ratio) = (_event.pcie_ratio, interval[_i + 1].pcie_ratio) if data_type == 'pcie' \
                                    else (_event.sm_ratio, interval[_i + 1].sm_ratio)
            _same_cnt = _same_cnt + 1 if _ratio == _next_ratio else 0
            if _same_cnt > FUR_LEN:
                return True
        return False

    def filter_comm_interval(self, data: Sequence[int], data_type: str = 'pcie'):
        """ 
        Filter the communication workload interval of each micro-batch. 
        --------------------------------------------------------
        Rules:
            - Find the base_ratio throughput ratio in the data as the min value.
            - Start timestamp: Surged throughput and at least THR_SCALE as much as the base_ratio
            - End timestamp: Throughput remains no larger than the THR_SCALE * base_ratio for the future FUR_LEN timestamps
        """
        if data_type == 'pcie':
            # base_ratio = stats.mode([_e.pcie_ratio for _e in data], keepdims=True)[0][0]
            base_ratio = int(np.min([_e.pcie_ratio for _e in data]))
        elif data_type == 'sm':
            base_ratio = max(1, int(np.min([_e.sm_ratio for _e in data])))
        # Filter
        intervals = list()
        idx_ptr = 0
        while idx_ptr < len(data):
            _event = data[idx_ptr]
            _ratio = _event.pcie_ratio if data_type == 'pcie' else _event.sm_ratio
            _scale = THR_SCALE if data_type == 'pcie' else SM_SCALE
            if _ratio >= _scale * base_ratio:
                # Start point
                _interval = list()
                while not self.is_et_reached(data=data, ptr=idx_ptr, base_ratio=base_ratio, data_type=data_type):
                    # Traverse the entire interval
                    _interval.append(data[idx_ptr])
                    idx_ptr += 1
                # if len(_interval) >= MIN_LEN and not self.is_abnormal_interval(interval=_interval, data_type=data_type):
                if len(_interval) >= MIN_LEN:
                    # Record valid interval
                    intervals.append(_interval)
                else:
                    idx_ptr += 1
            else:
                # Normal timestamp
                idx_ptr += 1
        return intervals

    ######################################
    #    Calculating Related Functions   #
    ######################################

    def _cal_comm_amt(self, interval: Sequence[Any]):
        """ Calculate the communication amount of the interval. """
        comm_amt = 0
        for _event in interval:
            _i = interval.index(_event)
            _t = (interval[_i + 1].timestamp - _event.timestamp) if _i < len(interval) - 1 else (interval[_i].timestamp - interval[_i - 1].timestamp)
            comm_amt += _event.pcie_ratio * _t
        return comm_amt

    def _cal_t_comp(self, interval: Sequence[Any]):
        """ Calculate the computing time of the interval. """
        return int(interval[-1].timestamp - interval[0].timestamp)
    
    def get_micro_batch_begin_index(self, comm_amts: Sequence[int]):
        """ Get the index of the micro batch begin in sorted comm_amts, removing pre-processing intervals. """
        for comm_amt in comm_amts:
            _i = comm_amts.index(comm_amt)
            _cnt = 0
            for _j in range(_i, min(_i + args.iter_num, len(comm_amts)), 1):
                if _j > len(comm_amts) - 1 or abs(comm_amts[_j] - comm_amt) / comm_amt > AMT_BIAS:
                    break
                _cnt += 1
            if _cnt == args.iter_num:
                return _i
        return -1
    
    def cal_t_comm(self, intervals: Sequence[Any]):
        """ 
        Calculate T_comm by: (1) Sort comm_amts in decreasing order; 
                             (2) Identify the surging point and classify intervals of BP (larger comm_amts) and FP (smaller comm_amts).
        """
        # Calculate communication amount
        comm_amts = [self._cal_comm_amt(interval=_interval) for _interval in intervals]
        # Remove the pre-processing data point
        if self.parallelism != 'mp':
            # Sort in decreasing order
            comm_amts = sorted(comm_amts, reverse=True)
            header_idx = self.get_micro_batch_begin_index(comm_amts=comm_amts)
            assert header_idx > -1, "Failed to find the begin index."
            comm_amts = comm_amts[header_idx:]
        # Identify the surging point
        if self.parallelism == 'dp':
            # Data parallelism
            surge_idx = -1
            base_ratio = comm_amts[0]
            for comm_amt in comm_amts:
                _i = comm_amts.index(comm_amt)
                if (_i + 1) >= args.iter_num and (_i + 1) % args.iter_num == 0 and \
                   _i < len(comm_amts) - 1 and comm_amts[_i + 1] * SURGE_FACTOR < base_ratio:
                    # Surging point
                    surge_idx = _i
                    break
            assert surge_idx > 0, "No surging point is found in comm_amts."
            # Calculate
            avg_fp = int(np.sum(comm_amts[(surge_idx + 1):]) / (args.iter_num * args.micro_batch_num_per_iter))
            avg_bp = int(np.sum(comm_amts[:(surge_idx + 1)]) / args.iter_num)
        elif self.parallelism == 'pp':
            # Pipeline
            micro_batch_idx = -1
            for comm_amt in comm_amts:
                _cnt = 0
                _ptr = comm_amts.index(comm_amt) + 1
                while _ptr < len(intervals):
                    if abs(comm_amt - comm_amts[_ptr]) / comm_amt > AMT_BIAS:
                        break
                    _ptr, _cnt = _ptr + 1, _cnt + 1
                    if (_ptr == len(comm_amts) - 1) or (_cnt == FUR_LEN):
                        micro_batch_idx = comm_amts.index(comm_amt)
                        break
                if micro_batch_idx > -1:
                    break
            assert micro_batch_idx > -1, "No begin of the micro batch is found."
            # Cut
            comm_amts = comm_amts[micro_batch_idx:]
            # Check the surging point of FP and BP
            surge_idx = -1
            base_ratio = comm_amts[0]
            for comm_amt in comm_amts:
                _i = comm_amts.index(comm_amt)
                if (_i + 1) >= args.iter_num and (_i + 1) % args.iter_num == 0 and \
                   _i < len(comm_amts) - 1 and comm_amts[_i + 1] * SURGE_FACTOR < base_ratio:
                    # Surging point
                    surge_idx = _i
                    break
            assert surge_idx > 0, "No surging point is found in comm_amts."
            # Calculate (comm_amts of FP is larger than BP)
            avg_fp = int(np.sum(comm_amts[:(surge_idx + 1)]) / (args.iter_num * args.micro_batch_num_per_iter))
            avg_bp = int(np.sum(comm_amts[(surge_idx + 1):]) / args.iter_num) if len(comm_amts[(surge_idx + 1):]) > 0 else avg_fp
        elif self.parallelism == 'mp':
            # Model parallelism
            avg_fp = avg_bp = int(np.sum(comm_amts) / (args.iter_num * args.micro_batch_num_per_iter))
        else:
            raise RuntimeError("Unsupported parallelism type.")
        # Return 
        return avg_fp, avg_bp

    def cal_t_comp(self, intervals: Sequence[Any]):
        """ 
        Calculate T_comp by: (1) The last timestamp - the first timestamp for each interval, sort in decreasing order; 
                             (2) Identify the surging point and classify intervals of FP (longer t_comp) and BP (shorter t_comp).
        """
        # Calculate computing time
        t_comps = [self._cal_t_comp(interval=_interval) for _interval in intervals]
        # Sort in decreasing order
        t_comps = sorted(t_comps, reverse=True)
        # Identify the surging point
        surge_idx = -1
        base_t = t_comps[0]
        for t_comp in t_comps:
            _i = t_comps.index(t_comp)
            if (_i + 1) >= args.iter_num and (_i + 1) % args.iter_num == 0 and \
                _i < len(t_comps) - 1 and t_comps[_i + 1] * SURGE_FACTOR < base_t:
                # Surging point
                surge_idx = _i
                break
        assert surge_idx > 0, "No surging point is found in t_comps."
        # Calculate
        avg_t_fp = int(np.sum(t_comps[:(surge_idx + 1)]) / (args.iter_num * args.micro_batch_num_per_iter))
        avg_t_bp = int(np.sum(t_comps[(surge_idx + 1):]) / args.iter_num)

        print(t_comps)
        print(surge_idx, len(t_comps))
        print(avg_t_fp, avg_t_bp)

        # Return 
        return avg_t_fp, avg_t_bp
    
    def parse_pcie_thr(self, txt_pth: str = './tmp/pcie_thr.txt'):
        """ Parse PCIe throughput data. """
        # FP/BP communication amount of each profiled GPU
        fp_list, bp_list = list(), list()
        # Read PCIe thr data
        gpu_data_list = self.read_pcie_thr_data(txt_pth=txt_pth)
        # Parse
        for _data in gpu_data_list:
            _intervals = self.filter_comm_interval(data=_data)
            avg_fp, avg_bp = self.cal_t_comm(intervals=_intervals)
            fp_list.append(avg_fp)
            bp_list.append(avg_bp)
        # Return
        return fp_list, bp_list
    
    def parse_sm_active(self, txt_pth: str = './tmp/sm_active.txt'):
        """ Parse SM active ratio data. """
        # FP/BP SM active ratio of each profiled GPU
        fp_list, bp_list = list(), list()
        # Read SM active ratio data
        gpu_data_list = self.read_sm_active_data(txt_pth=txt_pth)
        # Parse
        for _data in gpu_data_list:
            _intervals = self.filter_comm_interval(data=_data, data_type='sm')
            self.cal_t_comp(intervals=_intervals)
        # Return
        return fp_list, bp_list
    
    def parse(self, rep_path: str = "./nsys-rep/output.nsys-rep"):
        """ Entrypoint function of the parser. """
        # Construct sqlite database
        self.constr_db(rep_pth=rep_path, db_pth='./db/tmp_db.sqlite')
        # Parse PCIe throughput
        # fp_list, bp_list = self.parse_pcie_thr(txt_pth='./tmp/pcie_thr.txt')
        # Parse SM active ratio
        self.parse_sm_active(txt_pth='./tmp/sm_active.txt')

        # print(fp_list, bp_list)


parser = Parser()

parser.parse()
