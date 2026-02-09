# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: Measure performance for all configurations with alpa enabled (not optimized).

# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./measure_all_configs.sh [-x DEVICES_NAME] [-n NUM_HOSTS] [-d NUM_DEVICES_PER_HOST] [-m MODEL_NAME] [-p PARAM_NUM] [-b BATCH_SIZE] 
                                           [-l PIPELINE_LAYER_NUM] [-c MICRO_BATCHES_NUM] [-e ITER_NUM] [-w WARMUP_NUM] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - Only for traversing all parallel configurations to measure performance with alpa enabled."
	echo "################################################"
	echo "Description:"
    echo " - DEVICES_NAME (required): The concatenated devices name (e.g., '1_1080ti_1_p100')."
	echo " - NUM_HOSTS (required): The nodes (head + worker) num of Ray cluster."
    echo " - NUM_DEVICES_PER_HOST (required): The devices num in each node."
    echo " - MODEL_NAME (required): The model to be profiled (available: 'wide_resnet', 'bert', 'moe')."
    echo " - PARAM_NUM (required) The param num of the specified model. (e.g., '500M' when given 'wide_resnet'. Available params are provided in './configs.py')"
    echo " - BATCH_SIZE: The batch size of the profiled model."
    echo " - MICRO_BATCHES_NUM: The micro batches num for the pipeline in alpa (default: 16)."
    echo " - PIPELINE_LAYER_NUM: The pipeline layer num for alpa, which is the unit of pipeline stages auto-slicing (default: 16)."
    echo " - ITER_NUM: The iteration num of model training."
    echo " - WARMUP_NUM: The warmup num of model training."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}

########################################
#           Configurations             #
########################################
# Input
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

########################################
#               Get Args               #
########################################
while getopts "x:n:d:m:p:b:l:c:e:w:h" opt
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
    echo "[E] Error: The value of required variables (devices_name, num_hosts, num_devices_per_host, model_name, param_num) are not properly set, please use 'bash ./measure_all_configs.sh -h' to check."
    exit 1
fi


########################################
#               Profiling              #
########################################
num_devices=$((num_hosts * num_devices_per_host))

# Enumerate all configurations and measure
para_degrees=()
log_nd=$(python -c "import numpy as np; print(int(np.log2($num_devices)))")
for ((p_d = 0; p_d <= log_nd; p_d++)); do
    for ((d_d = 0; d_d <= log_nd - p_d; d_d++)); do
        m_d=$((log_nd - p_d - d_d))
        para_degrees+=("$(echo "2^$p_d" | bc),$(echo "2^$d_d" | bc),$(echo "2^$m_d" | bc)")
    done
done

# Configuration num
num_configs=${#para_degrees[@]}

# Traverse each configuration and profile
for idx in ${!para_degrees[@]}; do
    para_degree=${para_degrees[${idx}]}
    echo ""
    echo "------------------------------------------------------------------"
    echo "- ($((idx + 1))/${num_configs}) Profiling configurations (#PP, #DP, #MP): (${para_degree})..."
    echo "------------------------------------------------------------------"
    # Measure one configuration
    python jaxpr/runtime_profiler.py --measure_with_alpa \
                                     --devices_name ${devices_name} \
                                     --num_devices_per_host ${num_devices_per_host} \
                                     --num_hosts ${num_hosts} \
                                     --model_name ${model_name} \
                                     --param_num ${param_num} \
                                     --batch_size ${batch_size} \
                                     --num_micro_batches ${micro_batches_num} \
                                     --num_pipeline_layers ${num_pipeline_layers} \
                                     --niter ${iter_num} \
                                     --warmup_num ${warmup_num} \
                                     --parallel_degrees=${para_degree}
done
