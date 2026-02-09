# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: The global script of the profiling work, which recursively call the profile.sh script.


# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./run.sh [-x DEVICES_NAME] [-n NODES_NUM] [-d DEVICES_NUM_PER_NODE] [-m MODEL_NAME] [-p PARAM_NUM] [-e ITER_NUM] [-t TRY_TIMES] [-o] [-f] [-g] [-a] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - The run.sh should be put in the main directory of './jax/'."
	echo "################################################"
	echo "Description:"
    echo " - DEVICES_NAME (required): The concatenated devices name (e.g., '1_1080ti_1_p100')."
	echo " - NODES_NUM (required): The nodes (head + worker) num of Ray cluster."
    echo " - DEVICES_NUM_PER_NODE (required): The devices num in each node."
    echo " - MODEL_NAME (required): The model to be profiled (available: 'wide_resnet', 'bert', 'moe')."
    echo " - PARAM_NUM (required) The param num of the specified model. (e.g., '500M' when given 'wide_resnet'. Available params are provided in './configs.py')"
    echo " - ITER_NUM: The iteration num of model training."
    echo " - TRY_TIMES (default: 1): The repeat times for each profile configuration."
    echo " - [-o]: Force to apply data parallelism in all experiments, disabling auto-hybird search."
    echo " - [-f]: Force to apply pipeline parallelism in all experiments, disabling auto-hybird search."
    echo " - [-g]: Force to apply model (tensor) parallelism in all experiments, disabling auto-hybird search."
    echo " - [-a]: Perform test with manually specified configuration in './train.py' (default: false)."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


########################################
#           Configurations             #
########################################
# Info
dataset_name='none'
nodes_num=0
devices_num_per_node=0
model_name='none'
param_num='none'
try_times=1
num_pipeline_layers=1
micro_batches_num=1
iter_num=10
is_manual_config_test=false
is_dp_only=false
is_pp_only=false
is_mp_only=false
# Current path
CUR_PATH=$(cd $(dirname $0); pwd)
# Customized ray path
# RAY_PATH=$(cd ${CUR_PATH}; cd ../../ray; pwd)


########################################
#               Get Args               #
########################################
while getopts "x:n:d:m:p:e:t:ofgah" opt
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
        e)
        iter_num=${OPTARG};;
        t)
		try_times=${OPTARG};;
        o)
        is_dp_only=true;;
        f)
        is_pp_only=true;;
        g)
        is_mp_only=true;;
        a)
        is_manual_config_test=true;;
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
    echo "[E] Error: The value of required variables (devices_name, nodes_num, devices_num_per_node, model_name, param_num) are not properly set, please use 'bash ./run.sh -h' to check."
    exit 1
fi


########################################
#         Profiling Loop Start         #
########################################
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
elif [ ${model_name} == 'bert' ]; then
    # Configurations for Bert (1.3B)
    num_pipeline_layers=6
    micro_batches_num=16
    batch_size_list=("128" "256" "512")
    bs_num=3
elif [ ${model_name} == 'moe' ]; then
    # Configurations for Bert (1.3B)
    num_pipeline_layers=8
    micro_batches_num=16
    batch_size_list=("256" "512" "1024")
    bs_num=3
fi

# Profile
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    for((j=0;j<${try_times};j++))
    do
        try_idx=$((j+1))
        # .csv output path
        CSV_PATH=${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node/${devices_name}/${model_name}_${dataset_name}/bs_${batch_size}_nmb_${micro_batches_num}_pln_${num_pipeline_layers}_try_${try_idx}.csv

        if ! ${is_manual_config_test} || [ -f "${CSV_PATH}" ];then
            rm -rf ${CSV_PATH}
        fi

        if ! ${is_manual_config_test}; then
            # if [ ! ${is_dp_only} ] && [ ! ${is_pp_only} ] && [ ! ${is_mp_only} ]; then
            if ! ${is_dp_only} && ! ${is_pp_only} && ! ${is_mp_only}; then
                # Auto-configuration by alpa.
                bash ./profile.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -s ${dataset_name} -b ${batch_size} -l ${num_pipeline_layers} -c ${micro_batches_num} -e ${iter_num} -i ${try_idx}
            fi
            
            if ${is_dp_only}; then
                # Force to use data parallelism only.
                bash ./profile.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -s ${dataset_name} -b ${batch_size} -l ${num_pipeline_layers} -c ${micro_batches_num} -e ${iter_num} -i ${try_idx} -o
            fi

            if ${is_pp_only}; then
                # Force to use pipeline parallelism only.
                bash ./profile.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -s ${dataset_name} -b ${batch_size} -l ${num_pipeline_layers} -c ${micro_batches_num} -e ${iter_num} -i ${try_idx} -f
            fi

            if ${is_mp_only}; then
                # Force to use model parallelism only.
                bash ./profile.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -s ${dataset_name} -b ${batch_size} -l ${num_pipeline_layers} -c ${micro_batches_num} -e ${iter_num} -i ${try_idx} -g
            fi
        else
            # Manual-configuration.
            bash ./profile.sh -x ${devices_name} -n ${nodes_num} -d ${devices_num_per_node} -m ${model_name} -p ${param_num} -s ${dataset_name} -b ${batch_size} -l ${num_pipeline_layers} -c ${micro_batches_num} -e ${iter_num} -i ${try_idx} -a
            exit 1
        fi

        # Interval
        sleep 30s
    done
done


########################################
#          Profiling Loop End          #
########################################
