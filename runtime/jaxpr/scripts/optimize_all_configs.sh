# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: Optimize parallelisms on candidate batch sizes with alpa enabled.

# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./optimize_all_configs.sh [-x DEVICES_NAME] [-n NUM_HOSTS] [-d NUM_DEVICES_PER_HOST] [-m MODEL_NAME] 
                                            [-p PARAM_NUM] [-e ITER_NUM] [-w WARMUP_NUM] [-r] [-s] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - Only for optimizing all parallel configurations to get the optimal performance with alpa enabled."
	echo "################################################"
	echo "Description:"
    echo " - DEVICES_NAME (required): The concatenated devices name (e.g., '1_1080ti_1_p100')."
	echo " - NUM_HOSTS (required): The nodes (head + worker) num of Ray cluster."
    echo " - NUM_DEVICES_PER_HOST (required): The devices num in each node."
    echo " - MODEL_NAME (required): The model to be profiled (available: 'wide_resnet', 'bert', 'moe')."
    echo " - PARAM_NUM (required) The param num of the specified model. (e.g., '500M' when given 'wide_resnet'. Available params are provided in './configs.py')"
    echo " - ITER_NUM: The iteration num of model training."
    echo " - WARMUP_NUM: The warmup num of model training."
    echo " - [-r]: Prune search space of alpa."
    echo " - [-s]: Disable profiling database implemented by alpa."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


########################################
#           Configurations             #
########################################
# Info
dataset_name='none'
num_hosts=0
num_devices_per_host=0
model_name='none'
param_num='none'
num_pipeline_layers=1
micro_batches_num=16
iter_num=5
warmup_num=2
is_prune_space=false
is_disable_alpa_prof_db=false
# Current path
CUR_PATH=$(cd $(dirname $0); pwd)
# Customized ray path
# RAY_PATH=$(cd ${CUR_PATH}; cd ../../ray; pwd)


########################################
#               Get Args               #
########################################
while getopts "x:n:d:m:p:e:w:rsh" opt
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
        e)
        iter_num=${OPTARG};;
        w)
        warmup_num=${OPTARG};;
        r)
        is_prune_space=true;;
        s)
        is_disable_alpa_prof_db=true;;
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
    echo "[E] Error: The value of required variables (devices_name, num_hosts, num_devices_per_host, model_name, param_num) are not properly set, please use 'bash ./run.sh -h' to check."
    exit 1
fi


########################################
#              Profiling               #
########################################
num_devices=$((num_hosts * num_devices_per_host))

if [ ${model_name} == 'wide_resnet' ]; then
    # Configurations for Wide-ResNet50
    if [ ${param_num} == '13B' ]; then
        # ResNet101
        num_pipeline_layers=33
    else
        # ResNet50
        num_pipeline_layers=16
    fi
    micro_batches_num=16
    batch_size_list=("256" "512" "1024")
    bs_num=3
    # For wres, set the range of #pp to [max(#devices // 2, #hosts), #devices]. 
    # Accordingly, set the range of #dp and #mp to [1, min(2, #devices-per-host)]
    _half_num_devices=$((num_devices / 2))
    if [ ${_half_num_devices} -gt ${num_hosts} ];then
        _min_pp=${_half_num_devices}
    else
        _min_pp=${num_hosts}
    fi
    _max_dp=$((num_devices / _min_pp))
    # Format: [l_p, h_p, l_d, h_d, l_m, h_m]
    prune_prompt="${_min_pp}_${num_devices}_1_${_max_dp}_1_${_max_dp}"
elif [ ${model_name} == 'bert' ]; then
    # Configurations for Bert
    num_pipeline_layers=16
    micro_batches_num=16
    batch_size_list=("128" "256" "512")
    bs_num=3
    # For bert, set the range of #pp to [#hosts, #hosts * 2]. 
    # Accordingly, set the range of #dp and #mp to [#devices-per-host // 2, #devices-per-host]
    _half_num_devices_per_host=$((num_devices_per_host / 2))
    if [ ${_half_num_devices_per_host} -lt 1 ];then
        _half_num_devices_per_host=1
    fi
    _double_num_hosts=$((num_hosts * 2))
    # Format: [l_p, h_p, l_d, h_d, l_m, h_m]
    prune_prompt="${num_hosts}_${_double_num_hosts}_${_half_num_devices_per_host}_${num_devices_per_host}_${_half_num_devices_per_host}_${num_devices_per_host}"
elif [ ${model_name} == 'moe' ]; then
    # Configurations for Bert
    num_pipeline_layers=16
    micro_batches_num=16
    batch_size_list=("256" "512" "1024")
    bs_num=3
    # For moe, set the range of #pp to [#hosts, #hosts * 2]. 
    # Accordingly, set the range of #dp and #mp to [#devices-per-host // 2, #devices-per-host]
    _half_num_devices_per_host=$((num_devices_per_host / 2))
    if [ ${_half_num_devices_per_host} -lt 1 ];then
        _half_num_devices_per_host=1
    fi
    _double_num_hosts=$((num_hosts * 2))
    # Format: [l_p, h_p, l_d, h_d, l_m, h_m]
    prune_prompt="${num_hosts}_${_double_num_hosts}_${_half_num_devices_per_host}_${num_devices_per_host}_${_half_num_devices_per_host}_${num_devices_per_host}"
fi


# num_devices=$((num_hosts * num_devices_per_host))
# # Reset num pipeline layers based on given gpu num
# if [ ${num_pipeline_layers} -lt ${num_devices} ] && [ ${num_devices} -gt 8 ]; then
#     num_pipeline_layers=${num_devices}
# fi


# Profile
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    echo ""
    echo "------------------------------------------------------------------"
    echo "- ($((i + 1))/${bs_num}) Profiling ${model_name}_${param_num} with batch size: ${batch_size}..."
    echo "------------------------------------------------------------------"

    if ${is_prune_space} && ${is_disable_alpa_prof_db}; then
        # Prune searching space and disable alpa's profiling databtase
        python jaxpr/runtime_profiler.py --optimize_with_alpa \
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
                                         --prune_search_space \
                                         --prune_prompt ${prune_prompt} \
                                         --disable_alpa_profiling_db
    fi

    if ${is_prune_space} && ! ${is_disable_alpa_prof_db}; then
        # Only prune searching space
        python jaxpr/runtime_profiler.py --optimize_with_alpa \
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
                                         --prune_search_space \
                                         --prune_prompt ${prune_prompt}
    fi

    if ! ${is_prune_space} && ${is_disable_alpa_prof_db}; then
        # Only disable alpa's profiling database
        python jaxpr/runtime_profiler.py --optimize_with_alpa \
                                         --overwrite_data \
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
                                         --disable_alpa_profiling_db
    fi

    if ! ${is_prune_space} && ! ${is_disable_alpa_prof_db}; then
        # Normal case
        python jaxpr/runtime_profiler.py --optimize_with_alpa \
                                         --devices_name ${devices_name} \
                                         --num_devices_per_host ${num_devices_per_host} \
                                         --num_hosts ${num_hosts} \
                                         --model_name ${model_name} \
                                         --param_num ${param_num} \
                                         --batch_size ${batch_size} \
                                         --num_micro_batches ${micro_batches_num} \
                                         --num_pipeline_layers ${num_pipeline_layers} \
                                         --niter ${iter_num} \
                                         --warmup_num ${warmup_num}
    fi
done
