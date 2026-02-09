#!/bin/bash

# @ Info: The automated script to: 
#           - Apply Singularity to pull customized Alpa Docker image from Docker Hub.
#           - Obtain IP addresses of each node and decide the head node.
#           - Use Singularity Exec to execute a .sh script in the container.
# References:
#           - Obtain slurm nodes info: https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904
#           - Slurm job script for multiple nodes: https://stackoverflow.com/questions/52117145/slurm-job-script-for-multiple-nodes
#           - Slurm multiple program configuration: https://slurm.schedmd.com/srun.html#SECTION_MULTIPLE-PROGRAM-CONFIGURATION


########################################
#            TEST VERSION              #
########################################


########################################
#        Hardware Configurations       #
########################################
#SBATCH --job-name=multi_nodes_profile
#SBATCH -p dgx2
#SBATCH -N 2
# #SBATCH -n 2
#SBATCH --ntasks-per-node 1
#SBATCH --exclusive
#SBATCH --gres=gpu:4                                # GPU num per node (must be 4 in multi-nodes scenario)
#SBATCH --cpus-per-task=24
# #SBATCH --mem 256gb                               # Memory allocated per node
#SBATCH --wait-all-nodes 1                          # Do not begin execution until all nodes are ready for use.  
#SBATCH --output=./output/%j.outs
#SBATCH --error=./err_info/%j.err
# #SBATCH --mail-user=dicardo@sjtu.edu.cn
# #SBATCH --mail-type=ALL


########################################
#  Step 1. Pull Image From Docker Hub  #
########################################
# Pull 
unset XDG_RUNTIME_DIR
unset SINGULARITY_BIND
unset MODULEPATH
singularity pull alpa-profile.sif docker://dicardo/alpa-profile:v1


########################################
#  Step 2. Obtain Allocated Node Info  #
########################################
# Get node list
echo "[I][SHELL] Allocated node list: ${SLURM_JOB_NODELIST}"
# Assign the first node as the head node
HEAD_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "[I][SHELL] The address of the head node: ${HEAD_ADDR}"


########################################
# Step 2. Execute Script in Container  #
########################################
# Change permission
chmod 777 ./multi_nodes_profile.sh
# Check output dir
if [ ! -d "./profile_result" ];then
    mkdir ./profile_result
fi
# Check tmp dir
if [ ! -d "./tmp" ];then
    mkdir ./tmp
fi
# Execute
# Lanuch on node 0
srun -lN1 -n1 -r 0 singularity exec --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --network host --nv alpa-profile.sif ./multi_nodes_profile.sh -m ${HEAD_ADDR}
# Lanuch on node 1
srun -lN1 -n1 -r 1 singularity exec --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --network host --nv alpa-profile.sif ./multi_nodes_profile.sh -m ${HEAD_ADDR}
# Wait
sleep 1
wait


# singularity exec --bind "./profile_result:/app/jax/profile_result" --bind "./tmp:/app/jax/tmp" --network host --nv alpa-profile.sif ./single_node_profile.sh
