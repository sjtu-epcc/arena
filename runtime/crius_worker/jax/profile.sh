# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: The auto-profile script.


# Help message printer
helpMessage(){
    echo ""
	echo "################################################"
	echo "Usage:"										
	echo " - bash ./profile.sh [-x DEVICES_NAME] [-n NODES_NUM] [-d DEVICES_NUM_PER_NODE] [-m MODEL_NAME] [-p PARAM_NUM] [-s DATASET_NAME] [-b BATCH_SIZE]
                     [-l PIPELINE_LAYER_NUM] [-c MICRO_BATCHES_NUM] [-e ITER_NUM] [-i TRY_IDX] [-j JOB_ID] [-r RAY_ADDRESS] [-o] [-f] [-g] [-a] [-t] [-v] [-h]"
	echo "################################################"
	echo "Notice:"
	echo " - The profile.sh should be put in the main directory of './slice_profile/jax/'."
    echo " - The value of NODES_NUM and DEVICES_NUM_PER_NODE should be synchronized with '${RAY_PATH}/example-full.yaml'."
    echo " - Especially, check the IP address of each node in '${RAY_PATH}/example-full.yaml'."
	echo "################################################"
	echo "Description:"
    echo " - DEVICES_NAME (required): The concatenated devices name (e.g., '1_1080ti_1_p100')."
	echo " - NODES_NUM (required): The nodes (head + worker) num of Ray cluster."
    echo " - DEVICES_NUM_PER_NODE (required): The devices num in each node."
    echo " - MODEL_NAME (required): The name of the model to be profiled, available options are: ['wide_resnet', 'bert', 'moe']."
    echo " - PARAM_NUM (required) The param num of the specified model. (e.g., '500M' when given 'wide_resnet')"
    echo " - DATASET_NAME: The name of the dataset used by the profiled model, available options are: ['CIFAR10' (default), ]."
	echo " - BATCH_SIZE: The batch size of the profiled model."
    echo " - PIPELINE_LAYER_NUM: The pipeline layer num for alpa, which is the unit of pipeline stages auto-slicing (default: 16)."
    echo " - MICRO_BATCHES_NUM: The micro batches num for the pipeline in alpa (default: 16)."
    echo " - ITER_NUM: The iteration num of model training."
    echo " - TRY_IDX: The index of current try."
    echo " - JOB_ID: The job uuid of this training."
    echo " - RAY_ADDRESS: The address of the Ray cluster."
    echo " - [-o]: Force to apply data parallelism in all experiments, disabling auto-hybird search."
    echo " - [-f]: Force to apply pipeline parallelism in all experiments, disabling auto-hybird search."
    echo " - [-g]: Force to apply model (tensor) parallelism in all experiments, disabling auto-hybird search."
    echo " - [-a]: Perform test with manually specified configuration in './train.py' (default: false)."
    echo " - [-t]: Perform dummy test (default: false)."
	echo " - [-v]: Verbose mode (default: false)."
    echo " - [-h]: Help message."
	echo "################################################"
    echo ""
}


########################################
#           Configurations             #
########################################
# Input
devices_name='none'
nodes_num=0
devices_num_per_node=0
model_name='none'
param_num='none'
dataset_name='CIFAR10'
batch_size=32
num_pipeline_layers=1
micro_batches_num=1
iter_num=10
try_idx=1
job_id='default'
ray_address='auto'
is_dp_only=false
is_pp_only=false
is_mp_only=false
is_verbose=false
is_dummy_test=false
is_manual_config_test=false
# Current path
CUR_PATH=$(cd $(dirname $0); pwd)
# Customized ray path
# RAY_PATH=$(cd ${CUR_PATH}; cd ../../ray; pwd)


########################################
#               Get Args               #
########################################
while getopts "x:n:d:m:p:s:b:l:c:e:i:j:r:ofgatvh" opt
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
        s)
        dataset_name=${OPTARG};;
        b)
        batch_size=${OPTARG};;
        l)
        num_pipeline_layers=${OPTARG};;
        c)
        micro_batches_num=${OPTARG};;
        e)
        iter_num=${OPTARG};;
        i)
        try_idx=${OPTARG};;
        j)
        job_id=${OPTARG};;
        r)
        ray_address=${OPTARG};;
        o)
        is_dp_only=true;;
        f)
        is_pp_only=true;;
        g)
        is_mp_only=true;;
        a)
        is_manual_config_test=true;;
        t)
        is_dummy_test=true;;
        v)
        is_verbose=true;;
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
    echo "[E] Error: The value of required variables (devices_name, nodes_num, devices_num_per_node, model_name, param_num) are not properly set, please use 'bash ./profile.sh -h' to check."
    exit 1
fi

# Check path
if [ ! -d "${CUR_PATH}/profile_result" ];then
    mkdir ${CUR_PATH}/profile_result
fi
if [ ! -d "${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node" ];then
    mkdir ${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node
fi
if [ ! -d "${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node/${devices_name}" ];then
    mkdir ${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node/${devices_name}
fi
if [ ! -d "${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node/${devices_name}/${model_name}_${dataset_name}" ];then
    mkdir ${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node/${devices_name}/${model_name}_${dataset_name}
fi
# Console output file path
OUTPUT_PATH=${CUR_PATH}/profile_result/${nodes_num}_nodes_${devices_num_per_node}_devices_per_node/${devices_name}/${model_name}_${dataset_name}/bs_${batch_size}_nmb_${micro_batches_num}_pln_${num_pipeline_layers}_param_${param_num}_try_${try_idx}.log

# Profile
########################################
#         Alpa Profiling Start         #
########################################

echo ""
echo "########################################"
echo "#     Re-constructing Ray Cluster      #"
echo "########################################"
echo ""
echo "[I][SHELL] The path of the configuration file is: ${FILE_PATH}"
echo ""

# # Destroy existing ray cluster
# ray down ${FILE_PATH} -y

# # Construct ray cluster based on .yaml file
# ray up ${FILE_PATH} -y --no-config-cache

# # Wait
# sleep ${SLEEP_TIME}

echo ""
echo "[I][SHELL] The Ray cluster re-construction is completed. "
echo "[I][SHELL] To manually shut down the Ray cluster, use:"
echo " - (on head node) ray down [FILE_PATH] -y"
echo " - (on worker node) ray stop --force"
echo ""
echo "########################################"
echo "#       Executing Alpa Profiling       #"
echo "########################################"
echo ""
echo "########################################"
echo "Ray Cluster Info:"
echo " - Ray address: ${ray_address}"
echo " - Devices num: ${devices_name}"
echo " - Nodes num: ${nodes_num}"
echo " - Devices num per node: ${devices_num_per_node}"
echo "########################################"
echo "DL Job Info:"
echo " - Job UUID: ${job_id}"
echo " - Model name: ${model_name}"
echo " - Param num: ${param_num}"
echo " - Dataset name: ${dataset_name}"
echo " - Batch size: ${batch_size}"
echo " - Pipeline layer num: ${num_pipeline_layers}"
echo " - Miro-batches num: ${micro_batches_num}"
echo " - Iteration num: ${iter_num}"
echo " - Is DP only: ${is_dp_only}"
echo " - Is PP only: ${is_pp_only}"
echo " - Is MP only: ${is_mp_only}"
echo " - Is dummy test: ${is_dummy_test}"
echo " - Is manual config test: ${is_manual_config_test}"
echo " - Output log path: ${OUTPUT_PATH}"
echo "########################################"
echo ""

if ! ${is_manual_config_test} || [ -f "${OUTPUT_PATH}" ];then
    rm -rf ${OUTPUT_PATH}
fi


# Execute alpa profiling
if ! ${is_manual_config_test}; then
    if ${is_dp_only}; then
        # Force to apply data parallelism
        python ${CUR_PATH}/train.py --devices_name ${devices_name} \
                                    --num_devices_per_node ${devices_num_per_node} \
                                    --num_nodes ${nodes_num} \
                                    --model_name ${model_name} \
                                    --param_num ${param_num} \
                                    --dataset_name ${dataset_name} \
                                    --batch_size ${batch_size} \
                                    --resnet_layer_num 50 \
                                    --num_micro_batches ${micro_batches_num} \
                                    --num_pipeline_layers ${num_pipeline_layers} \
                                    --niter ${iter_num} \
                                    --try_idx ${try_idx} \
                                    --job_id ${job_id} \
                                    --ray_address ${ray_address} \
                                    --verbose ${is_verbose} \
                                    --is_dp_only \
                                    --is_ray_cluster_existed | tee ${OUTPUT_PATH}
    elif ${is_pp_only}; then
        # Force to apply pipeline parallelism
        python ${CUR_PATH}/train.py --devices_name ${devices_name} \
                                    --num_devices_per_node ${devices_num_per_node} \
                                    --num_nodes ${nodes_num} \
                                    --model_name ${model_name} \
                                    --param_num ${param_num} \
                                    --dataset_name ${dataset_name} \
                                    --batch_size ${batch_size} \
                                    --resnet_layer_num 50 \
                                    --num_micro_batches ${micro_batches_num} \
                                    --num_pipeline_layers ${num_pipeline_layers} \
                                    --niter ${iter_num} \
                                    --try_idx ${try_idx} \
                                    --job_id ${job_id} \
                                    --ray_address ${ray_address} \
                                    --verbose ${is_verbose} \
                                    --is_pp_only \
                                    --is_ray_cluster_existed | tee ${OUTPUT_PATH}
    elif ${is_mp_only}; then
        # Force to apply model parallelism
        python ${CUR_PATH}/train.py --devices_name ${devices_name} \
                                    --num_devices_per_node ${devices_num_per_node} \
                                    --num_nodes ${nodes_num} \
                                    --model_name ${model_name} \
                                    --param_num ${param_num} \
                                    --dataset_name ${dataset_name} \
                                    --batch_size ${batch_size} \
                                    --resnet_layer_num 50 \
                                    --num_micro_batches ${micro_batches_num} \
                                    --num_pipeline_layers ${num_pipeline_layers} \
                                    --niter ${iter_num} \
                                    --try_idx ${try_idx} \
                                    --job_id ${job_id} \
                                    --ray_address ${ray_address} \
                                    --verbose ${is_verbose} \
                                    --is_mp_only \
                                    --is_ray_cluster_existed | tee ${OUTPUT_PATH}
    else
        # Auto-configuration by alpa.
        if ! ${is_dummy_test}; then
            python ${CUR_PATH}/train.py --devices_name ${devices_name} \
                                        --num_devices_per_node ${devices_num_per_node} \
                                        --num_nodes ${nodes_num} \
                                        --model_name ${model_name} \
                                        --param_num ${param_num} \
                                        --dataset_name ${dataset_name} \
                                        --batch_size ${batch_size} \
                                        --resnet_layer_num 50 \
                                        --num_micro_batches ${micro_batches_num} \
                                        --num_pipeline_layers ${num_pipeline_layers} \
                                        --niter ${iter_num} \
                                        --try_idx ${try_idx} \
                                        --job_id ${job_id} \
                                        --ray_address ${ray_address} \
                                        --verbose ${is_verbose} \
                                        --is_ray_cluster_existed | tee ${OUTPUT_PATH}
        else
            # Dummy test.
            python ${CUR_PATH}/train.py --devices_name ${devices_name} \
                                        --num_devices_per_node ${devices_num_per_node} \
                                        --num_nodes ${nodes_num} \
                                        --model_name ${model_name} \
                                        --param_num ${param_num} \
                                        --dataset_name ${dataset_name} \
                                        --batch_size ${batch_size} \
                                        --resnet_layer_num 50 \
                                        --num_micro_batches ${micro_batches_num} \
                                        --num_pipeline_layers ${num_pipeline_layers} \
                                        --niter ${iter_num} \
                                        --is_dummy_test \
                                        --try_idx ${try_idx} \
                                        --job_id ${job_id} \
                                        --ray_address ${ray_address} \
                                        --verbose ${is_verbose} \
                                        --is_ray_cluster_existed | tee ${OUTPUT_PATH}
        fi 
    fi
else
    # Manual-configuration.
    python ${CUR_PATH}/train.py --devices_name ${devices_name} \
                                --num_devices_per_node ${devices_num_per_node} \
                                --num_nodes ${nodes_num} \
                                --model_name ${model_name} \
                                --param_num ${param_num} \
                                --dataset_name ${dataset_name} \
                                --batch_size ${batch_size} \
                                --resnet_layer_num 50 \
                                --num_micro_batches ${micro_batches_num} \
                                --num_pipeline_layers ${num_pipeline_layers} \
                                --niter ${iter_num} \
                                --try_idx ${try_idx} \
                                --job_id ${job_id} \
                                --ray_address ${ray_address} \
                                --verbose ${is_verbose} \
                                --is_manual_config_test \
                                --is_ray_cluster_existed | tee ${OUTPUT_PATH}
fi

########################################
#          Alpa Profiling End          #
########################################
