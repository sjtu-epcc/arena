# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: The setup script to prepare the crius profiler.

# Script directory
SCRIPT_PATH=$(cd $(dirname $0) && pwd -P)
cd $SCRIPT_PATH
# Module name
SO_NAME="crius_cupti.cpython-38-x86_64-linux-gnu.so"

# Symbollink to the pre-built cpp backend of crius profiler
if [ -h ${SCRIPT_PATH}/$SO_NAME ]; then
    rm -rf ${SCRIPT_PATH}/$SO_NAME 
fi

echo "[I] (Re)linking library to the pre-built cpp backend of crius profielr..."
ln -s ${SCRIPT_PATH}/cpp/build/$SO_NAME ${SCRIPT_PATH}
