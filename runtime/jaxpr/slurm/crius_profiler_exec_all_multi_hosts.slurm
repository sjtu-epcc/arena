#!/bin/bash

# @ Info: The automated script to: 
#           - Apply singularity to pull crius profiler image from docker Hub.
#           - Use singularity exec to execute a .sh script in the container and perform job auto-profiling.


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=crius_profile
#SBATCH -p a100
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# #SBATCH --exclude=vol06
#SBATCH --exclusive
#SBATCH --wait-all-nodes 1
#SBATCH --output=./output/%j.outs
#SBATCH --error=./err_info/%j.err
# #SBATCH --mail-user=dicardo@sjtu.edu.cn
# #SBATCH --mail-type=ALL

# Clean cache
singularity cache clean --force

########################################
#  Step 1. Pull Image From Docker Hub  #
########################################
# Pull 
unset XDG_RUNTIME_DIR
unset SINGULARITY_BIND
unset MODULEPATH
# singularity pull --force crius-profiler.sif docker://dicardo/crius-profiler:v1
singularity pull crius-profiler.sif docker://dicardo/crius-profiler:v1


########################################
#  Step 2. Obtain Allocated Node Info  #
########################################
# Get node list
echo "[I][SHELL] Allocated node list: ${SLURM_JOB_NODELIST}"
# Assign the first node as the head node
HEAD_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "[I][SHELL] The address of the head node: ${HEAD_ADDR}"


########################################
# Step 3. Execute Script in Container  #
########################################
# VITAL: You need to modify the hardware related info in 'singularity exec' command.

# Change permission
chmod 777 ./crius_profile_multi_hosts.sh

# Check communication data dir
if [ ! -d "./comm_data" ];then
    mkdir ./comm_data
fi
# Check output dir
if [ ! -d "./prof_log" ];then
    mkdir ./prof_log
fi
# Check tmp dir
if [ ! -d "./tmp" ];then
    mkdir ./tmp
fi
# Check tmp dir
if [ ! -d "./tmp_res" ];then
    mkdir ./tmp_res
fi

# Establish multi-hosts ray cluster and profile communication
# srun -N 2 -n 2 -c 30 singularity exec --writable-tmpfs --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/ground_truth:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -x 2_a100 -n 2 -d 4 -m default -p default -b default -o 

# Establish multi-hosts ray cluster and profile ground truth
# srun -N 2 -n 2 -c 30 singularity exec --writable-tmpfs --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/ground_truth:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -x 2_a100 -n 2 -d 4 -m default -p default -b default 
# wait

###########################
#        On 8 GPUs        #
###########################
# Profile Wide-ResNet (1B, expected for 4-GPU)
srun -N 2 -n 2 -c 30 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/ground_truth:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -x 2_a100 -n 2 -d 4 -m wide_resnet -p 1B -b 256 -g
wait
# Profile Bert (1.3B, expected for 4-GPU)
srun -N 2 -n 2 -c 30 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/ground_truth:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -x 2_a100 -n 2 -d 4 -m bert -p 1.3B -b 128 -g
wait
# # Profile MoE (1.3B, expected for 4-GPU)
# srun -N 2 -n 2 -c 30 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/ground_truth:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -x 2_a100 -n 2 -d 4 -m moe -p 1.3B -b 256 -g
