#!/bin/bash
# Author: Chunyu Xue

# @ Info: The automated script to perform crius profiling in the singularity container.


# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./crius_profiler.sh [-x DEVICES_NAME] [-n NODES_NUM] [-d DEVICES_NUM_PER_NODE] [-m MODEL_NAME] [-p PARAM_NUM] [-b BATCH_SIZE] 
                                      [-l PIPELINE_LAYER_NUM] [-c MICRO_BATCHES_NUM] [-e ITER_NUM] [-w WARMUP_NUM] [-o] [-i] [-a] [-u] [-z] [-g] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - The required arguments should be carefully review before running profiling experiments with varying configurations."
	echo "################################################"
	echo "Description:"
    echo " - DEVICES_NAME (required): The concatenated devices name (e.g., '1_1080ti_1_p100')."
	echo " - NODES_NUM (required): The nodes (head + worker) num of Ray cluster."
    echo " - DEVICES_NUM_PER_NODE (required): The devices num in each node."
    echo " - MODEL_NAME (required): The model to be profiled (available: 'wide_resnet', 'bert', 'moe')."
    echo " - PARAM_NUM (required) The param num of the specified model. (e.g., '500M' when given 'wide_resnet'. Available params are provided in './configs.py')"
    echo " - BATCH_SIZE: The batch size of the profiled model."
    echo " - MICRO_BATCHES_NUM: The micro batches num for the pipeline in alpa (default: 16)."
    echo " - PIPELINE_LAYER_NUM: The pipeline layer num for alpa, which is the unit of pipeline stages auto-slicing (default: 16)."
    echo " - ITER_NUM: The iteration num of model training."
    echo " - WARMUP_NUM: The warmup num of model training."
	echo " - [-o]: Offline profile communication operators, disabling crius profiling of hlo modules."
	echo " - [-i]: Offline inspect the hardware information."
	echo " - [-a]: Profile all candidate configurations of varying parallelism degrees."
	echo " - [-u]: Profile all candidate configurations with direct execution (as ground truth)."
	echo " - [-z]: Optimize the parallelism configuration with alpa enabled (as optimal)"
    echo " - [-g]: Generate alpa's profiling database for optimal parallelism search."
	echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


########################################
#           Configurations             #
########################################
# Info
devices_name='none'
num_hosts=0
num_devices_per_host=0
model_name='none'
param_num='none'
batch_size=256
num_pipeline_layers=16
micro_batches_num=16
iter_num=5
warmup_num=2
is_prof_comm_only=false
is_insp_hw_only=false
is_prof_all=false
is_exec_all=false
is_opt_all=false
is_gen_alpa_prof_db=false

########################################
#               Get Args               #
########################################
while getopts "x:n:d:m:p:b:l:c:e:w:oiauzgh" opt
do
	case ${opt} in
        x)
        devices_name=${OPTARG};;
        n)
        num_hosts=${OPTARG};;
		d)
		num_devices_per_host=${OPTARG};;
        m)
		model_name=${OPTARG};;
        p)
		param_num=${OPTARG};;
        b)
		batch_size=${OPTARG};;
        l)
		num_pipeline_layers=${OPTARG};;
        c)
	    micro_batches_num=${OPTARG};;
        e)
        iter_num=${OPTARG};;
        w)
        warmup_num=${OPTARG};;
		o)
        is_prof_comm_only=true;;
		i)
        is_insp_hw_only=true;;
		a)
        is_prof_all=true;;
		u)
        is_exec_all=true;;
		z)
        is_opt_all=true;;
		g)
        is_gen_alpa_prof_db=true;;
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
if [ ${devices_name} == 'none' ] || [ ${num_hosts} == 0 ] || [ ${num_devices_per_host} == 0 ] || [ ${model_name} == 'none' ] || [ ${param_num} == 'none' ];then
    echo "[E] Error: The value of required variables (devices_name, num_hosts, num_devices_per_host, model_name, param_num) are not properly set."
    exit 1
fi


########################################
#         Step 1. Preparation          #
########################################
# Activate python env and check dependencies
. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
	pip list | grep -E "jax|jaxlib|flax|alpa"
# Enter work dir
cd /app && pwd
# Build cpp backend
bash ./jaxpr/cpp/install.sh
# Mount cpp backend of the profiler
bash ./jaxpr/docker_setup.sh
# Output logfile name
OUTPUT_LOG_PATH=/app/tmp/${devices_name}_nodes_${num_devices_per_host}_devices_per_node_${model_name}_${param_num}_${batch_size}.log


########################################
#     (Optional) Inspect hardware      #
########################################
if ${is_insp_hw_only}; then
	# Offline inspect hardware information. `tee` will redirect the output log and print in .out file.
	./jaxpr/cpp/src/inspect_hardware | tee ${OUTPUT_LOG_PATH}
	exit 1
fi 


########################################
#  (Optional) Generate Alpa's prof db  #
########################################
if ${is_gen_alpa_prof_db}; then
	# Generate alpa's profiling database for optimal parallelism search
	# Rewrite output logfile name
	OUTPUT_LOG_PATH=/app/tmp/${devices_name}_nodes_${num_devices_per_host}_devices_per_node_${model_name}_${param_num}.log
	# Profiling
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats && \
		python crius_worker/jax/gen_prof_database.py --filename "/app/crius_worker/jax/prof_database/prof_database_${devices_name}_${num_devices_per_host}_d.pkl" \
									--max-comm-size-intra-node 32 --max-comm-size-inter-node 29 \
									--cache-filename "/app/tmp/hlo_op_cost_dict.pkl"
	# Sleep
	sleep 30
	# Stop ray processes
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ray stop --force

	exit 1
fi


########################################
#   (Optional) Profile communication   #
########################################
if ${is_prof_comm_only}; then
	# Offline profile communication operators. `tee` will redirect the output log and print in .out file.
	# Get default network interface
	NET_IF=$(route | grep default | grep -o "eno.")
	# Specify the network interface
	# Causes `nccl_all_to_all_thunk.cc:155: NCCL operation ncclGroupEnd() failed: unhandled cuda error` with 16 v100 gpus.
	# Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8` refer to https://stackoverflow.com/questions/68639661/nccl-operation-ncclgroupend-failed-unhandled-system-error 
	export NCCL_SOCKET_IFNAME=${NET_IF}
	# Specify the fraction of pre-allocated memory for JAX
	export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
	# Stop ray processes
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ray stop --force
	sleep 30
	
	# Init ray cluster
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats
	
	# Profile p2p communication
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && \
		python jaxpr/communication.py --profile_p2p \
									  --devices_name ${devices_name} \
									  --num_hosts ${num_hosts} \
									  --num_devices_per_host ${num_devices_per_host} | tee ${OUTPUT_LOG_PATH}
	sleep 30
	
	# Profile collective communication
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && \
		python jaxpr/communication.py --profile_collective \
									  --devices_name ${devices_name} \
									  --num_hosts ${num_hosts} \
									  --num_devices_per_host ${num_devices_per_host} | tee ${OUTPUT_LOG_PATH}
	exit 1
fi 


########################################
#          Step 2. Profiling           #
########################################
if ${is_prof_all}; then
	# Profile all configurations with estimation enabled
	export ENABLE_CRIUS_PROFILER=true
	
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && \
		python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs \
									  --devices_name ${devices_name} --num_hosts ${num_hosts} \
									  --num_devices_per_host ${num_devices_per_host} \
									  --model_name ${model_name} \
									  --param_num ${param_num} \
									  --batch_size ${batch_size} \
									  --num_micro_batches ${micro_batches_num} \
									  --num_pipeline_layers ${num_pipeline_layers} \
									  --niter ${iter_num} \
									  --warmup_num ${warmup_num} | tee ${OUTPUT_LOG_PATH}
	exit 1
fi

if ${is_exec_all}; then
	# Profile all configurations with direct execution (as ground truth)
	# Get default network interface
	NET_IF=$(route | grep default | grep -o "eno.")
	# Specify the network interface
	export NCCL_SOCKET_IFNAME=${NET_IF}
	# Specify the fraction of pre-allocated memory for JAX
	export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
	# Stop ray processes
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ray stop --force
	sleep 30

	# Init Ray cluster and execute profiling
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats && \
		bash jaxpr/measure_all_configs.sh -x ${devices_name} -n ${num_hosts} -d ${num_devices_per_host} -m ${model_name} -p ${param_num} -b ${batch_size} \
										-l ${num_pipeline_layers} -c ${micro_batches_num} -e ${iter_num} -w ${warmup_num} | tee ${OUTPUT_LOG_PATH}
	# Sleep
	sleep 30
	# Stop ray processes
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ray stop --force
	
	exit 1
fi

if ${is_opt_all}; then
	# Rewrite output logfile name
	OUTPUT_LOG_PATH=/app/tmp/${devices_name}_nodes_${num_devices_per_host}_devices_per_node_${model_name}_${param_num}.log
	# Optimize all configurations with alpa enabled (as optimal)
	# Get default network interface
	NET_IF=$(route | grep default | grep -o "eno.")
	# Specify the network interface
	export NCCL_SOCKET_IFNAME=${NET_IF}
	# Specify the fraction of pre-allocated memory for JAX
	export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
	# Stop ray processes
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ray stop --force
	sleep 30

	# Init Ray cluster and execute profiling
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats && \
		bash jaxpr/optimize_all_configs.sh -x ${devices_name} -n ${num_hosts} -d ${num_devices_per_host} -m ${model_name} -p ${param_num} \
										   -e ${iter_num} -w ${warmup_num} | tee ${OUTPUT_LOG_PATH}
	# Sleep
	sleep 30
	# Stop ray processes
	. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
		ray stop --force
	
	exit 1
fi
