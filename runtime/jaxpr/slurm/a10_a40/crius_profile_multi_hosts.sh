#!/bin/bash
# Author: Chunyu Xue

# @ Info: The automated script to establish multi-hosts Ray cluster and profile with Slurm support.


# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./crius_profile_multi_hosts.sh [-a HEAD_HOSTNAME] [-s CUR_HOSTNAME] [-x DEVICES_NAME] [-n NODES_NUM] 
                                                 [-d DEVICES_NUM_PER_NODE] [-m MODEL_NAME] [-p PARAM_NUM] [-b BATCH_SIZE] 
                                                 [-l PIPELINE_LAYER_NUM] [-c MICRO_BATCHES_NUM] [-e ITER_NUM] [-w WARMUP_NUM] [-o] [-u] [-z] [-g] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - Use SLURMD_NODENAME to get the underlying IP address of the current node. "
	echo "################################################"
	echo "Description:"
    echo " - HEAD_HOSTNAME (required): The hostname of the head node."
    echo " - CUR_HOSTNAME (required): The hostname of the current node."
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
    echo " - [-u]: Profile all candidate configurations with direct execution (as ground truth)."
    echo " - [-z]: Optimize the parallelism configuration with alpa enabled (as optimal)"
    echo " - [-g]: Generate alpa's profiling database for optimal parallelism search."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


# IP address
head_hostname='none'
cur_hostname='none'
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
is_exec_all=false
is_opt_all=false
is_gen_alpa_prof_db=false

########################################
#               Get Args               #
########################################
while getopts "a:s:x:n:d:m:p:b:l:c:e:w:ouzgh" opt
do
	case ${opt} in
        a)
        head_hostname=${OPTARG};;
        s)
        cur_hostname=${OPTARG};;
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

# Check whether ip_addr and head_hostname are properly set
if [ ${head_hostname} == 'none' ];then
    echo "[E][SHELL] Error: The value of required variables (head_hostname) is not properly set, please use '-h' to check."
    exit 1
fi

# Check whether required vars are properly set
if [ ${devices_name} == 'none' ] || [ ${num_hosts} == 0 ] || [ ${num_devices_per_host} == 0 ] || [ ${model_name} == 'none' ] || [ ${param_num} == 'none' ];then
    echo "[E] Error: The value of required variables (devices_name, num_hosts, num_devices_per_host, model_name, param_num) are not properly set."
    exit 1
fi


########################################
#         Step 1. Preparation          #
########################################
# Try activate python env and check dependencies
. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && pip list | grep -E "jax|ray|jaxlib|flax|alpa"
# Try enter work dir
cd /app && pwd
# Get default network interface
# NET_IF=$(route | grep default | grep -o "eno.")
# NET_IF=$(route | grep default | grep -o "vpapvn_........")
NET_IF=ib0.8068
echo "The network interface of this node is: ${NET_IF}"
# Specify the network interface
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Stop ray processes
ray stop --force
# Output logfile name
OUTPUT_LOG_PATH=/app/tmp/${devices_name}_nodes_${num_devices_per_host}_devices_per_node_${model_name}_${param_num}_${batch_size}.log


########################################
#          Step 2. Profiling           #
########################################
if [ -n "${SLURMD_NODENAME}" ];then
    cur_hostname=${SLURMD_NODENAME}
fi

head_ip_addr=$(getent hosts $head_hostname | awk '{print $1}')
cur_ip_addr=$(getent hosts $cur_hostname | awk '{print $1}')

if [ ${cur_hostname} == ${head_hostname} ];then
    # Head node
    echo "Initializing Ray instance from the head node (host name: ${cur_hostname}, IP address: ${cur_ip_addr})."
    . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
    ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats
    # RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats
    sleep 30
    . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
    ray status

    # Mount cpp backend of the profiler
    if [ ! -h "/app/jaxpr/crius_cupti.cpython-38-x86_64-linux-gnu.so" ];then
        bash ./jaxpr/docker_setup.sh
    fi

    if ${is_gen_alpa_prof_db}; then
        # Generate alpa's profiling database for optimal parallelism search
        # Rewrite output logfile name
	    OUTPUT_LOG_PATH=/app/tmp/${devices_name}_nodes_${num_devices_per_host}_devices_per_node_${model_name}_${param_num}.log
        # Profiling
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ulimit -c unlimited -n 65536 && \
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

    if ${is_prof_comm_only}; then
        # Profile p2p communication
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ulimit -c unlimited -n 65536 && \
            python jaxpr/communication.py --profile_p2p \
                                          --devices_name ${devices_name} \
                                          --num_hosts 2 \
                                          --num_devices_per_host 1 | tee ${OUTPUT_LOG_PATH}
        
        sleep 30
        
        # Profile collective communication
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ulimit -c unlimited -n 65536 && \
            python jaxpr/communication.py --profile_collective \
                                          --devices_name ${devices_name} \
                                          --num_hosts ${num_hosts} \
                                          --num_devices_per_host ${num_devices_per_host} | tee ${OUTPUT_LOG_PATH}
        sleep 30
        # Stop ray processes
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ray stop --force

        exit 1
    fi

    if ${is_exec_all}; then
        # Profile ground truth
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ulimit -c unlimited -n 65536 && \
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
        # Optimize all configurations with alpa enabled (as optimal)
        # Rewrite output logfile name
	    OUTPUT_LOG_PATH=/app/tmp/${devices_name}_nodes_${num_devices_per_host}_devices_per_node_${model_name}_${param_num}.log
        # Profiling
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ulimit -c unlimited -n 65536 && \
            bash jaxpr/optimize_all_configs.sh -x ${devices_name} -n ${num_hosts} -d ${num_devices_per_host} -m ${model_name} -p ${param_num} \
                                               -e ${iter_num} -w ${warmup_num} | tee ${OUTPUT_LOG_PATH}
        # Sleep
        sleep 30
        # Stop ray processes
        . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
            ray stop --force

        exit 1
    fi
else
    # Worker node
    sleep 20
    echo "Initializing Ray instance from the worker node (host name: ${cur_hostname}, IP address: ${cur_ip_addr}) to head node (host name: ${head_hostname}, IP address: ${head_ip_addr})."
    . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa && \
    ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=${head_ip_addr}:6379 --num-cpus 24 --object-store-memory 10737418240 --disable-usage-stats

    # NOTE: if the script on one node has finished execution, the raylet and gcs_server will be destroyed and thus
    #       the ray cluster is shutdown.
    sleep 1e7
fi
