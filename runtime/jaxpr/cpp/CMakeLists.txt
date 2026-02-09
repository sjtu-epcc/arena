# Root CMakeLists for compiling cpp backend of crius profiler.
# Modified from: https://github.com/UofT-EcoSystem/habitat-cu116/blob/master/cpp/CMakeLists.txt

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(crius_profiler LANGUAGES C CXX CUDA)

#############################
#   Locate Basic Packages   #
#############################

# Include our custom find module files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find all dependencies
find_package(CUDAToolkit REQUIRED)
# We need CUPTI here because CUDAToolkit does not include the nvperf-related
# libraries and headers that we need.
find_package(CUPTI REQUIRED)
find_package(NVPerf REQUIRED)

# Habitat source path
set(HABITAT_SRC_PATH ${HABITAT_SRC_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/external/habitat-cu116/cpp")
# Externals of habitat
add_subdirectory(${HABITAT_SRC_PATH}/external)
add_subdirectory(${HABITAT_SRC_PATH}/src/cuda)
# add_subdirectory(${HABITAT_SRC_PATH}/src/frontend)

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 75 80)
endif()
message(STATUS "CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}")


##############################
#   Set Python Importables   #
##############################

set(CriusCUPTI "crius_cupti")
pybind11_add_module(${CriusCUPTI} src/crius_cupti.cpp)
set_property(TARGET ${CriusCUPTI} PROPERTY CXX_STANDARD 11)


##############################
#      Link Source files     #
##############################

cmake_policy(SET CMP0076 NEW)
add_subdirectory(src)

# Specify dependencies for each target
target_link_libraries(${CriusCUPTI} PRIVATE habitat-cuda-lib)
set(CriusCUPTITransitiveDeps habitat-cuda-lib cupti_profilerhost_util)

# Since pybind11 modules are shared libraries, all the static libraries it
# depends on must be compiled as position independent code.
foreach(LIB ${CriusCUPTITransitiveDeps})
set_property(TARGET ${LIB} PROPERTY POSITION_INDEPENDENT_CODE ON)
endforeach()

# Turn on all compile warnings
set(AllTargets ${CriusCUPTI})
foreach(TGT ${AllTargets})
  if(CMAKE_COMPILER_IS_GNUCC)
    target_compile_options(${TGT} PRIVATE "-Wall")
  endif()
endforeach()
