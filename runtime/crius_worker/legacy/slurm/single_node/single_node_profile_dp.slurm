#!/bin/bash

# @ Info: The automated script to: 
#           - Apply Singularity to pull customized Alpa Docker image from Docker Hub.
#           - Use Singularity Exec to execute a .sh script in the container and perform single-node auto-profiling.


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=single_node_profile
#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
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
# singularity pull --force alpa-profile.sif docker://dicardo/alpa-profile:v1
singularity pull alpa-profile.sif docker://dicardo/alpa-profile:v1


########################################
# Step 2. Execute Script in Container  #
########################################
# VITAL: You need to modify the hardware related info in 'singularity exec' command.

# Change permission
chmod 777 ./single_node_profile.sh
chmod 777 ./gen_prof_database.sh

# Check output dir
if [ ! -d "./profile_result" ];then
    mkdir ./profile_result
fi
# Check tmp dir
if [ ! -d "./tmp" ];then
    mkdir ./tmp
fi
# Check tmp dir
if [ ! -d "./tmp_res" ];then
    mkdir ./tmp_res
fi
# Check database dir
if [ ! -d "./prof_database" ];then
    mkdir ./prof_database
fi

# # Generate profiling database
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./gen_prof_database.sh


# ###########################
# #         On 1 GPU        #
# ###########################
# # Profile Wide-ResNet (500M, expected for 2-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 1 -m wide_resnet -p 500M -t 1 -o
# # Profile Bert (760M, expected for 2-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 1 -m bert -p 760M -t 1 -o
# # Profile MoE (690M, expected for 2-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 1 -m moe -p 690M -t 1 -o

# # Profile Wide-ResNet (1B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 1 -m wide_resnet -p 1B -t 1 -o
# # Profile Bert (1.3B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 1 -m bert -p 1.3B -t 1 -o
# # Profile MoE (1.3B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 1 -m moe -p 1.3B -t 1 -o


###########################
#        On 2 GPUs        #
###########################
# # Profile Wide-ResNet (500M, expected for 2-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m wide_resnet -p 500M -t 1 -o
# # Profile Bert (760M, expected for 2-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m bert -p 760M -t 1 -o
# # Profile MoE (690M, expected for 2-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m moe -p 690M -t 1 -o

# # Profile Wide-ResNet (1B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m wide_resnet -p 1B -t 1 -o
# # Profile Bert (1.3B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m bert -p 1.3B -t 1 -o
# # Profile MoE (1.3B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m moe -p 1.3B -t 1 -o

# # Profile Wide-ResNet (2B, expected for 8-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m wide_resnet -p 2B -t 1 -o
# # Profile Bert (2.6B, expected for 8-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m bert -p 2.6B -t 1 -o
# # Profile MoE (2.4B, expected for 8-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 2 -m moe -p 2.4B -t 1 -o


###########################
#        On 4 GPUs        #
###########################
# Profile Wide-ResNet (500M, expected for 2-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m wide_resnet -p 500M -t 1 -o
# Profile Bert (760M, expected for 2-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m bert -p 760M -t 1 -o
# Profile MoE (690M, expected for 2-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m moe -p 690M -t 1 -o

# Profile Wide-ResNet (1B, expected for 4-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m wide_resnet -p 1B -t 1 -o
# Profile Bert (1.3B, expected for 4-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m bert -p 1.3B -t 1 -o
# Profile MoE (1.3B, expected for 4-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m moe -p 1.3B -t 1 -o

# Profile Wide-ResNet (2B, expected for 8-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m wide_resnet -p 2B -t 1 -o
# Profile Bert (2.6B, expected for 8-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m bert -p 2.6B -t 1 -o
# Profile MoE (2.4B, expected for 8-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m moe -p 2.4B -t 1 -o

# Profile Wide-ResNet (4B, expected for 16-GPU)
singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m wide_resnet -p 4B -t 1 -o
# Profile Bert (6.7B, expected for 16-GPU)
# (OOM) singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m bert -p 6.7B -t 1 -o
# Profile MoE (10B, expected for 16-GPU)
# (OOM) singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 4 -m moe -p 10B -t 1 -o


###########################
#        On 8 GPUs        #
###########################
# # Profile Wide-ResNet (1B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m wide_resnet -p 1B -t 1 -o
# # Profile Bert (1.3B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m bert -p 1.3B -t 1 -o
# # Profile MoE (1.3B, expected for 4-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m moe -p 1.3B -t 1 -o

# # Profile Wide-ResNet (2B, expected for 8-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m wide_resnet -p 2B -t 1 -o
# # Profile Bert (2.6B, expected for 8-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m bert -p 2.6B -t 1 -o
# # Profile MoE (2.4B, expected for 8-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m moe -p 2.4B -t 1 -o

# # Profile Wide-ResNet (4B, expected for 16-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m wide_resnet -p 4B -t 1 -o
# # Profile Bert (6.7B, expected for 16-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m bert -p 6.7B -t 1 -o
# # Profile MoE (10B, expected for 16-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m moe -p 10B -t 1 -o

# # Profile Wide-ResNet (6.8B, expected for 32-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m wide_resnet -p 6.8B -t 1 -o
# # Profile Bert (15B, expected for 32-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m bert -p 15B -t 1 -o
# # Profile MoE (27B, expected for 32-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 8 -m moe -p 27B -t 1 -o


###########################
#       On 16 GPUs        #
###########################
# # # Profile Wide-ResNet (2B, expected for 8-GPU)
# # singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m wide_resnet -p 2B -t 1 -o
# # # Profile Bert (2.6B, expected for 8-GPU)
# # singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m bert -p 2.6B -t 1 -o
# # Profile MoE (2.4B, expected for 8-GPU)
# # singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m moe -p 2.4B -t 1 -o

# # Profile Wide-ResNet (4B, expected for 16-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m wide_resnet -p 4B -t 1 -o
# # Profile Bert (6.7B, expected for 16-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m bert -p 6.7B -t 1 -o
# # Profile MoE (10B, expected for 16-GPU)
# # singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m moe -p 10B -t 1 -o

# # Profile Wide-ResNet (6.8B, expected for 32-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m wide_resnet -p 6.8B -t 1 -o
# # Profile Bert (15B, expected for 32-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m bert -p 15B -t 1 -o
# # Profile MoE (27B, expected for 32-GPU)
# # singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m moe -p 27B -t 1 -o

# # Profile Wide-ResNet (13B, expected for 64-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m wide_resnet -p 13B -t 1 -o
# # Profile Bert (39B, expected for 64-GPU)
# singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m bert -p 39B -t 1 -o
# # Profile MoE (70B, expected for 64-GPU)
# # singularity exec --bind "./prof_database:/app/jax/prof_database" --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --bind "./tmp_res:/app/jax/tmp_res" --network host --nv alpa-profile.sif ./single_node_profile.sh -x 1_v100 -n 1 -d 16 -m moe -p 70B -t 1 -o
