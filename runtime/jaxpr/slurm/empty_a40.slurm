#!/bin/bash

# @ Info: The automated script to: 
#           - Apply singularity to pull crius profiler image from docker Hub.
#           - Use singularity exec to execute a .sh script in the container and perform job auto-profiling.


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=crius_profile
#SBATCH -p gpu-a40-bigdata
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --exclusive
#SBATCH --output=./output/%j.outs
#SBATCH --error=./err_info/%j.err
# #SBATCH --mail-user=dicardo@sjtu.edu.cn
# #SBATCH --mail-type=ALL

python3 empty.py
