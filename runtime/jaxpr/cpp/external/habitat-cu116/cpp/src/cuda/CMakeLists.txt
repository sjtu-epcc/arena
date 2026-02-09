# This library encapsulates Habitat's bindings to CUDA related utilities, such as CUPTI and the standalone
# occupancy calculator. It should not be dependent on other Habitat libraries.
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 75 80)
endif()

add_library(habitat-cuda-lib
  diagnostics.cu
  cupti_exceptions.cpp
  cupti_manager.cpp
  cupti_profiler.cpp
  cupti_tracer.cpp
  kernel.cpp
  # legacy_cupti_profiler.cpp
  new_cupti_profiler.cpp
)

# (modified by crius). If `CUDA::cupti` target not found, modify to `${CUPTI_LIBRARIES}`

target_link_libraries(habitat-cuda-lib
  PRIVATE
    ${CUPTI_LIBRARIES}
    # CUDA::cupti
    cuda
    CUDA::cudart
    CUDA::cuda_driver
    cupti_profilerhost_util
    ${NVPerf_LIBRARIES}
)

target_include_directories(habitat-cuda-lib
  PRIVATE
    ${CUPTI_INCLUDE_DIRS}
)

target_compile_options(
  habitat-cuda-lib
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_60,code=sm_60>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_61,code=sm_61>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_70,code=sm_70>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_75,code=sm_75>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode arch=compute_80,code=sm_80>"
)
