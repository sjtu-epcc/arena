#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

""" 
Implementations of offline roofline profiler to generate the GPU-specific roofline model. 
"""


class RooflineProfiler:
    """
    The class of roofline profiler to offline generate GPU-specific roofline model.
    """

    def __init__(
        self,
        gpu_type: str,
        gpu_mem_bw_gb: int,
        gpu_peak_gflops: int,
    ) -> None:
        """ Initialize a roofline profiler object. """

        self.gpu_type = gpu_type
        self.gpu_mem_bw_gb = gpu_mem_bw_gb
        self.gpu_peak_gflops = gpu_peak_gflops 
        
        self._intensity_threshold = self.gpu_peak_gflops / self.gpu_mem_bw_gb
        self._roofline = None
        self._constr()

    def _constr(self) -> None:
        """ 
        Construct a roofline model based on GPU memory bandwidth and peak GFLOPS.
        
        The roofline model is two-segment linear function, with the intensity (FLOPs / memory access) 
        threshold equals to GPU peak FLOPS / memory bandwidth:

         - When the computation intensity is lower than the threshold, the attainable performance is 
           computation intensity * memory bandwidth;
        
         - When the computation intensity is higher than the threshold, the attainable performance 
           constantly equals to the peak FLOPS of the GPU.
        """

        def __roofline(x):
            """ Roofline model callable function. """
            return (x * self.gpu_mem_bw_gb) if x < self._intensity_threshold else self.gpu_peak_gflops

        self._roofline = __roofline

    def query(
        self,
        gflops: int,
        memory_access_gb: int,
    ) -> int:
        """ 
        Query the attainable performance of the specific configurations. 

        Args:
         - `gflops`: The GFLOPs of the operator.
         - `memory_access_gb`: The memory access amount in GB of the operator. 
        """
        
        if gflops <= LOWER_BOUND_GFLOPS and memory_access_gb <= LOWER_BOUND_MEM_ACCESS_GB:
            return 0
        
        # TODO(chunyu): We should simulteneously set the lower bound of flops and memory access, since
        #               some time-consuming operators can be almost no flops but large memory access.
        #               To evaluate the performance of some zero-flop operators (e.g., copy), we use its
        #               memory access / max gpu memory bandwidth to estimate its computation load.

        return self._roofline(x=(gflops / memory_access_gb))


####################################
#       Hardware Information       #
####################################

GPU_MEMORY_BW_GB = {
    "a40": 696,
    # "a40": 528,
}


GPU_PEAK_GFLOPS = {
    "a40": 37400, 
    # "a40": 74800, 
}

LOWER_BOUND_GFLOPS = 0.1
LOWER_BOUND_MEM_ACCESS_GB = 0.1
