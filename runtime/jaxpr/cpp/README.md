# C++ Source Codes of Crius Runtime Profiler

> This directory contains the C++ source code of Crius runtime profiler. CMake is used for compilation. Referance: https://github.com/geoffxy/habitat/tree/master/cpp


## Usages

 - (Done in `install.sh`) Before compiling this cpp backend, perform the following file replacements:
    - Copy all files in `src` and `include` from `/usr/local/cuda/cuda-12.1/extras/CUPTI/samples/extensions` into `../../external/habitat-cu116/cpp/external/cupti_profilerhost_util`.
    
    - Replace `../../external/habitat-cu116/cpp/src/cuda/cuda_occupancy.h` with `/usr/local/cuda/include/cuda_occupancy.h`.

 - 
