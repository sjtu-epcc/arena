/* !/usr/bin/env cuda
 * -*- coding:utf-8 -*-
 * Author: Chunyu Xue
 */

// Compiled by `nvcc -o inspect_hardware inspect_hardware.cu`
// Executed by `./inspect_hardware`

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;


void inspect_hardware() {
    // Device count
    int device_count;
    cudaGetDeviceCount(&device_count);

    int dev_idx;
    for (dev_idx = 0; dev_idx < device_count; dev_idx++) {
        cout << "" << endl;
        // Driver and runtime version
        int driver_version(0), runtime_version(0);
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        cout << "CUDA driver version:" << driver_version / 1000 << "." << (driver_version % 1000) / 10 << endl;
        cout << "CUDA runtime version:" << runtime_version / 1000 << "." << (runtime_version % 1000) / 10 << endl;

        // Device properties
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev_idx);
        cout << "Compute major: " << device_prop.major << endl;
        cout << "Compute minor: " << device_prop.minor << endl;
        // Clock frequency
        cout << "Base clock frequency: " << device_prop.clockRate * 1e-3f << " MHz" << endl;
        cout << "Max clock frequency: " << device_prop.memoryClockRate * 1e-3f << " MHz" << endl;
        // Memory bandwidth
        cout << "Memory buswidth: " << device_prop.memoryBusWidth << " bit" << endl;
        cout << "Memory bandwidth: " << 2.0 * (device_prop.memoryBusWidth / 8.0) * device_prop.memoryClockRate / 1.0e6 << " GB/s" << endl;
        // Memory size
        cout << "Total GPU memory size: " << device_prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << endl;
        cout << "Total constant memory size: " << device_prop.totalConstMem / 1024.0 << " KB" << endl;
        // Shared memory
        cout << "Shared memory per sm: " << device_prop.sharedMemPerMultiprocessor << endl;
        cout << "Shared memory per block: " << device_prop.sharedMemPerBlock << " B" << endl;
        cout << "Shared memory per block (optin): " << device_prop.sharedMemPerBlockOptin << " B" << endl;
        // SM num
        cout << "SM num: " << device_prop.multiProcessorCount << endl;
        // Thread num 
        cout << "Max thread num per sm: " << device_prop.maxThreadsPerMultiProcessor << endl;
        cout << "Max thread num per block: " << device_prop.maxThreadsPerBlock << endl;
        // Register num
        cout << "Max register num per sm: " << device_prop.regsPerMultiprocessor << endl;
        cout << "Max register num per block: " << device_prop.regsPerBlock << endl;
        // Wrap size
        cout << "Wrap size: " << device_prop.warpSize << endl;
        // Peak gflops per second
        cout << "Peak GFLOPS per second: (please check Nvidia GPU Spec)" << endl;
    }
}


int main() {
    inspect_hardware();
    return 0;
}
