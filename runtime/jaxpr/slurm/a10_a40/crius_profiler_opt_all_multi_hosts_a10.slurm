#!/bin/bash

# @ Info: The automated script to: 
#           - Apply singularity to pull crius profiler image from docker Hub.
#           - Use singularity exec to execute a .sh script in the container and perform job auto-profiling.


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=crius_profile
#SBATCH -p gpu-a10
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
# #SBATCH --exclude=vol06
# #SBATCH --exclude=gpu17
# #SBATCH --exclude=gpu22
#SBATCH --exclusive
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
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_log:/app/jaxpr/prof_log" --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -g -x 2_a10 -n 2 -d 2 -m default -p default
wait

# singularity run --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv -B /etc/libibverbs.d crius-profiler.sif

# ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --node-ip-address 192.168.1.66 --port=6379 --num-cpus 16 --object-store-memory 10737418240 --disable-usage-stats

# ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=192.168.1.66:6379 --node-ip-address 192.168.1.71 --num-cpus 16 --object-store-memory 10737418240 --disable-usage-stats


###########################
#        On 4 GPUs        #
###########################
# Profile Wide-ResNet (500M, expected for 2-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m wide_resnet -p 500M
# Profile Bert (760M, expected for 2-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m bert -p 760M
# Profile MoE (690M, expected for 2-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m moe -p 690M

# Profile Wide-ResNet (1B, expected for 4-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m wide_resnet -p 1B
# Profile Bert (1.3B, expected for 4-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m bert -p 1.3B
# Profile MoE (1.3B, expected for 4-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m moe -p 1.3B

# Profile Wide-ResNet (2B, expected for 8-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m wide_resnet -p 2B
# Profile Bert (2.6B, expected for 8-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m bert -p 2.6B
# Profile MoE (2.4B, expected for 8-GPU)
srun -N 2 -n 2 -c 32 --gres=gpu:2 singularity exec --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profile_multi_hosts.sh -a ${HEAD_ADDR} -z -x 2_a10 -n 2 -d 2 -m moe -p 2.4B
