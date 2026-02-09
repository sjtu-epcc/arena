#!/bin/bash
# Author: Chunyu Xue

# @ Info: The automated script to perform multi-nodes alpa profiling with Slurm support.


########################################
#            TEST VERSION              #
########################################


# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./multi_nodes_profile.sh [-m HEAD_IP_ADDR] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - Use SLURMD_NODENAME to get the underlying IP address of the current node. "
	echo "################################################"
	echo "Description:"
    echo " - HEAD_IP_ADDR (required): The IP address of the head node (e.g., '10.2.37.141')."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


# IP address
head_ip_addr='none'


########################################
#               Get Args               #
########################################
while getopts "m:h" opt
do
	case ${opt} in
        m)
        head_ip_addr=${OPTARG};;
		h)
		helpMessage
        exit 1;;
		\?)
		echo ""
		echo "[E] Error: Invalid argument received..."
		helpMessage
		exit 1;;
	esac
done

# Check whether ip_addr and head_ip_addr are properly set
if [ ${head_ip_addr} == 'none' ];then
    echo "[E][SHELL] Error: The value of required variables (head_ip_addr) are not properly set, please use 'bash ./multi_nodes_profile.sh -h' to check."
    exit 1
fi


########################################
#         Step 1. Preparation          #
########################################
# Try activate python env and check dependencies
. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && pip list | grep -E "jax|ray|jaxlib|flax|alpa"
# Try enter work dir
cd /app/jax && pwd
# Get default network interface
NET_IF=$(route | grep default | grep -o "eno.")
# Specify the network interface
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Stop ray processes
ray stop --force


########################################
#          Step 2. Profiling           #
########################################
if [ ${SLURMD_NODENAME} == ${head_ip_addr} ];then
    # Head node
    echo "I'm the head node, the underlying node is ${SLURMD_NODENAME}."
    . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
    ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats && \
    sleep 30 && \
    # Profiling script. 
    # You need to modify the command line arguments to profile different configurations.
    # bash ./run.sh -x 1_node -n 1 -d 2 -l 16
    ray status
else
    # Worker node
    echo "I'm the worker node, the underlying node is ${ip_addr}."
    . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
    ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=${head_ip_addr}:6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats
fi

# TODO: How to stop the ray processes in worker node after the profiling is compeleted?
