#!/bin/bash

# @ Info: The automated script to: 
#           - Apply singularity to pull crius profiler image from docker Hub.
#           - Use singularity exec to execute a .sh script in the container and perform job auto-profiling.


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=crius_profile
#SBATCH -p dgx2
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gres=gpu:16
#SBATCH --cpus-per-task=96
# #SBATCH --exclude=vol06
# # SBATCH --exclusive
#SBATCH --wait-all-nodes 1
#SBATCH --output=./output/%j.outs
#SBATCH --error=./err_info/%j.err
#SBATCH --mail-user=dicardo@sjtu.edu.cn
#SBATCH --mail-type=ALL

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


# Generate alpa's profiling database
# 2 x 8 GPUs
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -g -x 2_v100 -n 2 -d 8 -m default -p default
wait
# 2 x 16 GPUs
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -g -x 2_v100 -n 2 -d 16 -m default -p default
wait


###########################
#       On 16 GPUs        #
###########################
# Profile Wide-ResNet (2B, expected for 8-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m wide_resnet -p 2B
wait
# Profile Bert (2.6B, expected for 8-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m bert -p 2.6B
wait
# Profile MoE (2.4B, expected for 8-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m moe -p 2.4B
wait

# Profile Wide-ResNet (4B, expected for 16-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m wide_resnet -p 4B
wait
# Profile Bert (6.7B, expected for 16-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m bert -p 6.7B
wait
# Profile MoE (10B, expected for 16-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m moe -p 10B
wait

# Profile Wide-ResNet (6.8B, expected for 32-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m wide_resnet -p 6.8B
wait
# Profile Bert (15B, expected for 32-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m bert -p 15B
wait
# Profile MoE (27B, expected for 32-GPU)
srun -N 2 -n 2 -c 48 --gres=gpu:8 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 8 -m moe -p 27B
wait


###########################
#       On 32 GPUs        #
###########################
# Profile Wide-ResNet (4B, expected for 16-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m wide_resnet -p 4B
wait
# Profile Bert (6.7B, expected for 16-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m bert -p 6.7B
wait
# Profile MoE (10B, expected for 16-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m moe -p 10B
wait

# Profile Wide-ResNet (6.8B, expected for 32-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m wide_resnet -p 6.8B
wait
# Profile Bert (15B, expected for 32-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m bert -p 15B
wait
# Profile MoE (27B, expected for 32-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m moe -p 27B
wait

# Profile Wide-ResNet (13B, expected for 64-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m wide_resnet -p 13B
wait
# Profile Bert (39B, expected for 64-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m bert -p 39B
wait
# Profile MoE (70B, expected for 64-GPU)
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_v100 -n 2 -d 16 -m moe -p 70B
wait




# Establish multi-hosts ray cluster and profile communication
srun -N 2 -n 2 -c 96 --gres=gpu:16 singularity exec --writable-tmpfs --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/ground_truth:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -x 2_v100 -n 2 -d 16 -m default -p default -b default -o 
wait
