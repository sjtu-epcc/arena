#!/bin/bash
# Author: Chunyu Xue

# @ Info: The automated script to perform single-node alpa profiling in the Singularity container.


# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./single_node_profile.sh [-x DEVICES_NAME] [-n NODES_NUM] [-d DEVICES_NUM_PER_NODE] [-m MODEL_NAME] [-p PARAM_NUM] [-t TRY_TIMES] [-o] [-f] [-g] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - The required arguments should be carefully review before running profiling experiments with varying configurations."
	echo "################################################"
	echo "Description:"
    echo " - DEVICES_NAME (required): The concatenated devices name (e.g., '1_1080ti_1_p100')."
	echo " - NODES_NUM (required): The nodes (head + worker) num of Ray cluster."
    echo " - DEVICES_NUM_PER_NODE (required): The devices num in each node."
    echo " - MODEL_NAME (required): The model to be profiled (available: 'wide_resnet', 'bert', 'moe')."
	echo " - PARAM_NUM (required) The param num of the specified model. (e.g., '500M' when given 'wide_resnet')"
    echo " - TRY_TIMES (default: 1): The repeat times for each profile configuration."
	echo " - [-o]: Force to apply data parallelism in all experiments, disabling auto-hybird search."
    echo " - [-f]: Force to apply pipeline parallelism in all experiments, disabling auto-hybird search."
    echo " - [-g]: Force to apply model (tensor) parallelism in all experiments, disabling auto-hybird search."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


########################################
#           Configurations             #
########################################
# Info
devices_name='none'
nodes_num=0
devices_num_per_node=0
model_name='none'
param_num='none'
try_times=1
is_dp_only=false
is_pp_only=false
is_mp_only=false


########################################
#               Get Args               #
########################################
while getopts "x:n:d:m:p:t:ofgh" opt
do
	case ${opt} in
        x)
        devices_name=${OPTARG};;
        n)
        nodes_num=${OPTARG};;
		d)
		devices_num_per_node=${OPTARG};;
        m)
		model_name=${OPTARG};;
		p)
		param_num=${OPTARG};;
        t)
		try_times=${OPTARG};;
		o)
        is_dp_only=true;;
        f)
        is_pp_only=true;;
        g)
        is_mp_only=true;;
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

# Check whether required vars are properly set
if [ ${devices_name} == 'none' ] || [ ${nodes_num} == 0 ] || [ ${devices_num_per_node} == 0 ] || [ ${model_name} == 'none' ] || [ ${param_num} == 'none' ];then
    echo "[E] Error: The value of required variables (devices_name, nodes_num, devices_num_per_node, model_name, param_num) are not properly set, please use 'bash ./single_node_profile.sh -h' to check."
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
. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
ray stop --force
# Sleep
sleep 30s

########################################
#          Step 2. Profiling           #
########################################
# if [ ! ${is_dp_only} ] && [ ! ${is_pp_only} ] && [ ! ${is_mp_only} ]; then
if ! ${is_dp_only} && ! ${is_pp_only} && ! ${is_mp_only}; then
	# Auto-configuration by alpa.
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
	ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats && \
	bash ./run.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -t ${try_times} -e 5
fi

if ${is_dp_only}; then
	# Force to use data parallelism only.
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
	ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats && \
	bash ./run.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -t ${try_times} -e 5 -o
fi

if ${is_pp_only}; then
	# Force to use pipeline parallelism only.
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
	ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats && \
	bash ./run.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -t ${try_times} -e 5 -f
fi

if ${is_mp_only}; then
	# Force to use model parallelism only.
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
	ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats && \
	bash ./run.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -t ${try_times} -e 5 -g
fi

# Sleep
sleep 30s
# Stop ray processes
. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
ray stop --force
