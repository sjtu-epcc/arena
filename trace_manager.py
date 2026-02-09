#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
A script related to the job server to handle with runtime job workload.
"""

import os
from typing import Sequence, Any

from macro.macro_def import (
    SCHEDULING_INTERVAL, IS_ARRIVAL_EVENT)
from utils import (ArrivalEvent, read_csv_file)


class TraceManager:
    """ The class of trace manager, which provides APIs for loading simulated/runtime job workload. """
    def __init__(self, job_submit_density: float = 1.0) -> None:
        self.trace_dir = os.environ.get("CRIUS_TRACE_DIR", "./traces")
        self.job_submit_density = job_submit_density

    def _parse_one_entry(self, rec: Sequence[Any]):
        """ 
        Parse the entry in elasticflow trace. 
        -----------------------------------------------
        Return: (job id, submission timestamp, iteration num, model name, deadline, batch size, gpu num, gpu type)
        """
        return (rec[0], int(rec[1]), int(rec[2]), rec[3], int(rec[4]), int(rec[5]), int(rec[6]), rec[7])
    
    def load_trace(self, file_name: str = "dummy_trace.csv", is_runtime: bool = False):
        """ 
        Read csv file to load and formulate the job trace in an event-chunk style (for simulation) 
        or event style (for runtime). 
        """
        arrival_trace = list()
        trace_data = read_csv_file(file_path=os.path.join(self.trace_dir, file_name), 
                                   style="iterate_row_to_list")
        timestamp = int(trace_data[0][1])       # Init to the submission time of the first job
        job_cnt, rec_idx = 0, 0
        chunk = list()                          # Jobs inside one sched interval (only for simulation)
        
        while rec_idx < len(trace_data):
            # Parse entry
            (_job_id, _sub_time, _iter_num, _model_name, 
             _deadline, _batch_size, _gpu_num, _gpu_type) = self._parse_one_entry(trace_data[rec_idx])

            if self.job_submit_density != 1.0:
                # Scale the iteration number according to the density
                _iter_num = max(1, int(_iter_num * self.job_submit_density))

            if is_runtime:
                # Runtime trace
                job_cnt, rec_idx = job_cnt + 1, rec_idx + 1
                arrival_trace.append(
                    ArrivalEvent(IS_ARRIVAL_EVENT, _job_id, 
                                 "job_" + str(job_cnt).zfill(len(str(len(trace_data)))),
                                 _sub_time, _iter_num, _model_name, _deadline, _batch_size, 
                                 _gpu_type, _gpu_num)
                )
                continue

            # Simulation trace
            if timestamp + SCHEDULING_INTERVAL >= _sub_time:
                # Submission timestamp of this job has been covered by current timestamp
                job_cnt, rec_idx = job_cnt + 1, rec_idx + 1
                chunk.append(
                    ArrivalEvent(IS_ARRIVAL_EVENT, _job_id, 
                                 "job_" + str(job_cnt).zfill(len(str(len(trace_data)))),
                                 _sub_time, _iter_num, _model_name, _deadline, _batch_size,
                                 _gpu_type, _gpu_num)
                )
            else:
                # All jobs inside this chunk has been recorded
                arrival_trace.append(chunk)
                chunk = list()
                while timestamp + SCHEDULING_INTERVAL < _sub_time:
                    # Add empty chunk until reaching the next event
                    timestamp += SCHEDULING_INTERVAL
                    if timestamp + SCHEDULING_INTERVAL < _sub_time:
                        arrival_trace.append(list())
        
        # Add the last chunk
        if not is_runtime and len(chunk) > 0:
            arrival_trace.append(chunk)   

        return arrival_trace
