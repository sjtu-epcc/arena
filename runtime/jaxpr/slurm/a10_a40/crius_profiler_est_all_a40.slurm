#!/bin/bash

# @ Info: The automated script to: 
#           - Apply singularity to pull crius profiler image from docker Hub.
#           - Use singularity exec to execute a .sh script in the container and perform job auto-profiling.


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=crius_profile
#SBATCH -p gpu-a40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
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
# Step 2. Execute Script in Container  #
########################################
# VITAL: You need to modify the hardware related info in 'singularity exec' command.

# Change permission
chmod 777 ./crius_profiler.sh

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


# (Optional) Only inspect hardware information (or use `salloc` and `./inspect_hardware` directly)
# singularity exec --writable-tmpfs --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -x 1_a40 -n 1 -d 1 -m default -p default -b default -i


# (Optional) Only profile communication operators
# singularity exec --writable-tmpfs --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -x 1_a40 -n 1 -d 4 -m default -p default -b default -o


# ###########################
# #         On 1 GPU        #
# ###########################
# # Profile Wide-ResNet (500M, expected for 2-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 1 -m wide_resnet -p 500M
# # Profile Bert (760M, expected for 2-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 1 -m bert -p 760M
# # Profile MoE (690M, expected for 2-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 1 -m moe -p 690M

###########################
#        On 2 GPUs        #
###########################
# Profile Wide-ResNet (500M, expected for 2-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m wide_resnet -p 500M
# Profile Bert (760M, expected for 2-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m bert -p 760M
# Profile MoE (690M, expected for 2-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m moe -p 690M

# Profile Wide-ResNet (1B, expected for 4-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m wide_resnet -p 1B
# Profile Bert (1.3B, expected for 4-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m bert -p 1.3B
# Profile MoE (1.3B, expected for 4-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m moe -p 1.3B

# Profile Wide-ResNet (2B, expected for 8-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m wide_resnet -p 2B
# Profile Bert (2.6B, expected for 8-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m bert -p 2.6B
# Profile MoE (2.4B, expected for 8-GPU)
singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 2 -m moe -p 2.4B


###########################
#        On 4 GPUs        #
###########################
# # Profile Wide-ResNet (500M, expected for 2-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m wide_resnet -p 500M
# # Profile Bert (760M, expected for 2-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m bert -p 760M
# # Profile MoE (690M, expected for 2-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m moe -p 690M

# # Profile Wide-ResNet (1B, expected for 4-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m wide_resnet -p 1B -b 256
# # Profile Bert (1.3B, expected for 4-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m bert -p 1.3B -b 128
# # Profile MoE (1.3B, expected for 4-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m moe -p 1.3B

# # Profile Wide-ResNet (2B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m wide_resnet -p 2B
# # Profile Bert (2.6B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m bert -p 2.6B
# # Profile MoE (2.4B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m moe -p 2.4B

# # Profile Wide-ResNet (4B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m wide_resnet -p 4B
# # Profile Bert (6.7B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m bert -p 6.7B
# # Profile MoE (10B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 4 -m moe -p 10B


###########################
#        On 8 GPUs        #
###########################
# # Profile Wide-ResNet (1B, expected for 4-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 2_a400 -n 2 -d 4 -m wide_resnet -p 1B -b 256
# # Profile Bert (1.3B, expected for 4-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 2_a400 -n 2 -d 4 -m bert -p 1.3B -b 128
# # Profile MoE (1.3B, expected for 4-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 2_a400 -n 1 -d 8 -m moe -p 1.3B

# # Profile Wide-ResNet (2B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m wide_resnet -p 2B
# # Profile Bert (2.6B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m bert -p 2.6B
# # Profile MoE (2.4B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m moe -p 2.4B

# # Profile Wide-ResNet (4B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m wide_resnet -p 4B
# # Profile Bert (6.7B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m bert -p 6.7B
# # Profile MoE (10B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m moe -p 10B

# # Profile Wide-ResNet (6.8B, expected for 32-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m wide_resnet -p 6.8B
# # Profile Bert (15B, expected for 32-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m bert -p 15B
# # Profile MoE (27B, expected for 32-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 8 -m moe -p 27B


###########################
#       On 16 GPUs        #
###########################
# # Profile Wide-ResNet (2B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m wide_resnet -p 2B
# # Profile Bert (2.6B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m bert -p 2.6B
# Profile MoE (2.4B, expected for 8-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m moe -p 2.4B

# # Profile Wide-ResNet (4B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m wide_resnet -p 4B
# # Profile Bert (6.7B, expected for 16-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m bert -p 6.7B
# # Profile MoE (10B, expected for 16-GPU)
# # singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m moe -p 10B

# # Profile Wide-ResNet (6.8B, expected for 32-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m wide_resnet -p 6.8B
# # Profile Bert (15B, expected for 32-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m bert -p 15B
# # Profile MoE (27B, expected for 32-GPU)
# # singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m moe -p 27B

# # Profile Wide-ResNet (13B, expected for 64-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m wide_resnet -p 13B
# # Profile Bert (39B, expected for 64-GPU)
# singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m bert -p 39B
# # Profile MoE (70B, expected for 64-GPU)
# # singularity exec --writable-tmpfs --bind "./tmp:/app/jaxpr/tmp" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/estimate_all:/app/tmp" --bind "./tmp_res:/app/tmp_res" --network host --nv crius-profiler.sif ./crius_profiler.sh -a -x 1_a40 -n 1 -d 16 -m moe -p 70B
