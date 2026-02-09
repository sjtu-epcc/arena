# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: The installation script to prepare the Alpa environment for Crius worker.


# NOTE: Please use 'conda activate alpa' before this script.
# Install packages
conda install -c conda-forge coin-or-cbc -y
pip3 install pandas 
pip3 install --upgrade pip 
pip3 install cupy-cuda113 
pip3 install alpa 
pip3 install jaxlib==0.3.22+cuda113.cudnn820 -f https://alpa-projects.github.io/wheels.html 
pip3 install numpy==1.23.5 
pip3 install flask

# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH