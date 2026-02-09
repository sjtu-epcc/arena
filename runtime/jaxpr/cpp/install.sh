# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: The installation script to enable cpp backend of the crius profiler.

# Customize your path
# NOTE: Instead of using include/li64 under `cuda/cuda-12.1`, we must use include/lib64 
# under `cuda` to match the running cuda version
export CUPTI_DIR="/usr/local/cuda"  # Rather than "/usr/local/cuda/cuda-12.1/extras/CUPTI"
CUPTI_SAMPLE_DIR="/usr/local/cuda/extras/CUPTI"
# Module name
MODULE_NAME=crius_cupti
# .so file name: crius_cupti.cpython-38-x86_64-linux-gnu.so
SO_NAME=${MODULE_NAME}$(python3-config --extension-suffix)
# Script directory
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH
# Habitat path
export HABITAT_PATH="${SCRIPT_PATH}/external/habitat-cu116"


# Rewrite pushd/popd operations to visit dirs
function pushd() {
    command pushd "$@" > /dev/null
}


function popd() {
    command popd "$@" > /dev/null
}


function compile_crius_cupti() {
    echo "[I][SHELL] Compiling the Crius cpp backend..."
    if [ -d "build" ];then
        rm -rf build
    fi

    pushd .
    mkdir -p build
    pushd build

    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j crius_cupti

    if [ ! -f $SO_NAME ]; then
        echo "[E][SHELL] Compilation Error: Could not find $SO_NAME after compilation."
        exit 1
    fi

    echo "[I][SHELL] Compilation is done."
    popd
    popd
}


function symlink() {
    echo "[I][SHELL] Symbollink to the Crius cpp backend..."
    # if [ ! -d "${SCRIPT_PATH}/../cpp_backend" ];then
    #     mkdir -p ${SCRIPT_PATH}/../cpp_backend
    # fi
    # if [ ! -h ${SCRIPT_PATH}/../cpp_backend/$SO_NAME ]; then
    #     ln -s build/$SO_NAME ${SCRIPT_PATH}/../cpp_backend/$SO_NAME
    # fi

    if [ ! -h ${SCRIPT_PATH}/../$SO_NAME ]; then
        ln -s ${SCRIPT_PATH}/build/$SO_NAME ${SCRIPT_PATH}/../
    fi
}


function copy_cupti_sample() {
    echo "[I][SHELL] Copying CUPTI examples from" $CUPTI_SAMPLE_DIR
    cp -r ${CUPTI_SAMPLE_DIR}/samples/extensions/src ${HABITAT_PATH}/cpp/external/cupti_profilerhost_util/
    cp -r ${CUPTI_SAMPLE_DIR}/samples/extensions/include ${HABITAT_PATH}/cpp/external/cupti_profilerhost_util/
    cp /usr/local/cuda/include/cuda_occupancy.h ${HABITAT_PATH}/cpp/src/cuda/cuda_occupancy.h
}


function main() {
    copy_cupti_sample
    compile_crius_cupti
    symlink
}

main $@
