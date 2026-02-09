#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
A script related to the customized performance migrator towards kernel execution time.
Ref: https://www.usenix.org/conference/atc21/presentation/yu
"""

import math
import argparse
from typing import Sequence, Tuple
import pickle

# Back to upper dir
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxpr.utils import (
    load_device_info_table, NS_TO_S)


class PrefMigrator:
    """ 
    The class of customized performance migrator towards kernel execution, which migrate
    kernel iteration time from base_gpu to dst_gpu. 
    """
    def __init__(self, base_gpu_type: str = ""):
        self.base_gpu_type = base_gpu_type
        self.metadata_pth = os.environ.get("KERNEL_METADATA_PATH")
        self.device_info_table = load_device_info_table()
    
    def set_base_gpu_type(self, base_gpu_type: str):
        """ Set the base gpu type of the migrator. """
        self.base_gpu_type = base_gpu_type 
    
    def save_base_perf_data(self, per_kernel_infos: Sequence[Tuple], job_id: str, module_idx: int):
        """ Save the profiled kernel performance data on the base gpu type. """
        # File path
        if not os.path.exists(self.metadata_pth):
            os.mkdir(self.metadata_pth)
        _file_name = f"job_{job_id}.pkl"
        pth = os.path.join(self.metadata_pth, _file_name)
        # Read existed or create
        if os.path.exists(pth):
            print(f"[TMP] Existed profiled kernel data in `{pth}`, updating it...")
            with open(pth, "rb") as f:
                per_kernel_infos_table = pickle.load(f)
        else:
            print(f"[TMP] Profiled kernel data not found in `{pth}`, creating it...")
            per_kernel_infos_table = dict() # _key -> per_kernel_infos
        # Update dict
        _pp_degree = os.environ.get("NUM_GENERAL_STAGES")
        _dp_degree = os.environ.get("DP_DEGREE")
        _mp_degree = os.environ.get("MP_DEGREE")
        _key = f"pp_{_pp_degree}_dp_{_dp_degree}_mp_{_mp_degree}_module_{module_idx}"
        per_kernel_infos_table[_key] = per_kernel_infos
        # Record
        with open(pth, "wb") as f:
            pickle.dump(per_kernel_infos_table, f)
        print(f"[TMP] Base performance data of job ({job_id}) has been saved to {pth}.")
    
    def read_base_perf_data(self, job_id: str, module_idx: str):
        """ Read the profiled kernel performance data on the base gpu type. """
        # File path
        _file_name = f"job_{job_id}.pkl"
        _pth = os.path.join(self.metadata_pth, _file_name)
        with open(_pth, "rb") as f:
            per_kernel_infos_table = pickle.load(f)
        # Read dict
        _pp_degree = os.environ.get("NUM_GENERAL_STAGES")
        _dp_degree = os.environ.get("DP_DEGREE")
        _mp_degree = os.environ.get("MP_DEGREE")
        _key = f"pp_{_pp_degree}_dp_{_dp_degree}_mp_{_mp_degree}_module_{module_idx}"
        return per_kernel_infos_table[_key]
    
    def _query_gamma_value(self, kernel_name: str):
        """ Query the gamma value of the given kernel from the offline profiled database. """
        # TODO(chunyu) Implement this.
        return 0.5
    
    def _migrate_one_kernel_perf(self, kernel_name: str, kernel_iter_t: float, 
                                 kernel_block_num: int, thread_block_occupancies: dict, 
                                 dst_gpu_type: str):
        """ 
        Migrate the kernel iteration time from src_gpu to dst_gpu with wave scaling method
        proposed in habitat. 
        Modified from: https://github.com/geoffxy/habitat/blob/5f01e523a1dc30dbfbaaa39cf4880a534c7781a2/
                       analyzer/habitat/analysis/wave_scaling/roofline.py
        """
        # GPU spec
        base_gpu_spec = self.device_info_table[self.base_gpu_type]
        dst_gpu_spec = self.device_info_table[dst_gpu_type] 
        base_sm_num = base_gpu_spec["num_sms"]
        dst_sm_num = dst_gpu_spec["num_sms"]
        # Base wave size
        base_occupancy = thread_block_occupancies[self.base_gpu_type]
        base_wave_size = base_sm_num * base_occupancy
        # Dst wave size
        dst_occupancy = thread_block_occupancies[dst_gpu_type]
        dst_wave_size = dst_sm_num * dst_occupancy
        # Clock frequency factor
        _base_clock_freq = base_gpu_spec["base_clock_mhz"]
        _dst_clock_freq = dst_gpu_spec["base_clock_mhz"]
        clock_factor = _base_clock_freq / _dst_clock_freq
        # Gamma value
        gamma_value = self._query_gamma_value(kernel_name)

        # Check if the kernel is too "small" - if it doesn't fill a single wave on the 
        # current device AND if it doesn't fill a single wave on the destination device
        if base_wave_size > 0 and kernel_block_num // base_wave_size == 0 and \
            dst_wave_size > 0 and kernel_block_num // dst_wave_size == 0:
            # Scale kernel execution time with computing factor only
            base_max_occupancy = math.ceil(kernel_block_num / base_sm_num)
            dst_max_occupancy = math.ceil(kernel_block_num / dst_sm_num)
            _occupancy_factor = dst_max_occupancy / base_max_occupancy
            partial_compute_factor = clock_factor * _occupancy_factor

            return kernel_iter_t * math.pow(partial_compute_factor, (1.0 - gamma_value))

        # Scaling kernel execution time based on spec factors
        _base_mem_bw = base_gpu_spec["mem_bandwidth_gb"]
        _dst_mem_bw = dst_gpu_spec["mem_bandwidth_gb"]
        bw_factor = _base_mem_bw / _dst_mem_bw
        sm_factor = base_sm_num / dst_sm_num

        return (kernel_iter_t * math.pow(bw_factor, gamma_value) * 
                math.pow(clock_factor, (1.0 - gamma_value)) * 
                math.pow(sm_factor, (1.0 - gamma_value)))
    
    def migrate_kernel_perfs(self, per_kernel_infos: Sequence[Tuple], dst_gpu_type: str):
        """ 
        Migrate all kernel execution times to dst_gpu_type.
        """
        assert self.base_gpu_type != "", "Base GPU type of the migrator is not set."
        total_kernel_time = 0
        for (_name, _num_blocks, _run_time_ns, _occupancies) in per_kernel_infos:
            _est_time_ns = self._migrate_one_kernel_perf(_name, _run_time_ns, _num_blocks, 
                                                         _occupancies, dst_gpu_type)
            total_kernel_time += (_est_time_ns * NS_TO_S)
        return total_kernel_time

    def offline_generate_roofline_model(self, dst_gpu_type: str):
        """ 
        Offline generate the gamma value for varying kernels in roofline model on the destination
        gpu type. The inherent reason is that ... (TODO, wrong with 'computation amount linearly 
        scale with input spec')
        """
        # TODO(chunyu) Implement this by enumerating compute-intensive kernels with varying input spec.
        # TODO(chunyu) How does the input size affects the arithmetic intensity of a kernel?


def offline_generate_roofline_models():
    """ Offline generating roofline models on all candidate gpu types. """
    # Environmental variables
    os.environ["DEVICE_INFO_PATH"] = "./jaxpr/device_info/device_infos.json"

    perf_migrator = PrefMigrator(args.base_gpu_type)


if __name__ == "__main__":
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_gpu_type", default="a40", type=str)
    # parser.add_argument("--estimate_e2e", default=False, action='store_true', 
    #                     help="Whether to estimate e2e pipeline iteration time of model.")
    args = parser.parse_args()

    offline_generate_roofline_models()
