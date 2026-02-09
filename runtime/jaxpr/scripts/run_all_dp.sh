# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: Estimate performance for all parallelism configurations.


batch_size_list=("256" "512" "1024")
batch_size_list_small=("128" "256" "512")
bs_num=3

gpu_num=1


########################################
#               Get Args               #
########################################
while getopts "d:" opt
do
	case ${opt} in
        d)
        gpu_num=${OPTARG};;
		\?)
		echo ""
		echo "[E] Error: Invalid argument received..."
		exit 1;;
	esac
done


if [ ${gpu_num} == 1 ]; then
    ########################
    #      For 1 GPU       #
    ########################

    ###########################
    #   Profile Wide-ResNet   #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile Wide-ResNet (500M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 1 \
                                        --devices_name 1_a40 --model_name wide_resnet --param_num 500M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,1,1
    done

    ###########################
    #      Profile Bert       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list_small[${i}]}
        # Profile Bert (760M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 1 \
                                        --devices_name 1_a40 --model_name bert --param_num 760M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,1,1
    done

    ###########################
    #       Profile MoE       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile MoE (690M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 1 \
                                        --devices_name 1_a40 --model_name moe --param_num 690M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,1,1
    done

    exit 1
fi


if [ ${gpu_num} == 2 ]; then
    ########################
    #      For 2 GPU       #
    ########################

    ###########################
    #   Profile Wide-ResNet   #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile Wide-ResNet (500M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name wide_resnet --param_num 500M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
        # Profile Wide-ResNet (1B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name wide_resnet --param_num 1B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
        # Profile Wide-ResNet (2B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name wide_resnet --param_num 2B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
    done

    ###########################
    #      Profile Bert       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list_small[${i}]}
        # Profile Bert (760M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name bert --param_num 760M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
        # Profile Bert (1.3B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name bert --param_num 1.3B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
        # Profile Bert (2.6B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name bert --param_num 2.6B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
    done

    ###########################
    #       Profile MoE       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile MoE (690M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name moe --param_num 690M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
        # Profile MoE (1.3B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name moe --param_num 1.3B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
        # Profile MoE (2.4B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 2 \
                                        --devices_name 1_a40 --model_name moe --param_num 2.4B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,2,1
    done

    exit 1
fi


if [ ${gpu_num} == 4 ]; then
    ########################
    #      For 4 GPU       #
    ########################

    ###########################
    #   Profile Wide-ResNet   #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile Wide-ResNet (500M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name wide_resnet --param_num 500M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile Wide-ResNet (1B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name wide_resnet --param_num 1B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile Wide-ResNet (2B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name wide_resnet --param_num 2B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile Wide-ResNet (4B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name wide_resnet --param_num 4B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
    done

    ###########################
    #      Profile Bert       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list_small[${i}]}
        # Profile Bert (760M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name bert --param_num 760M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile Bert (1.3B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name bert --param_num 1.3B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile Bert (2.6B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name bert --param_num 2.6B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile Bert (6.7B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name bert --param_num 6.7B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
    done

    ###########################
    #       Profile MoE       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile MoE (690M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name moe --param_num 690M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile MoE (1.3B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name moe --param_num 1.3B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile MoE (2.4B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name moe --param_num 2.4B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
        # Profile MoE (10B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 2 --num_devices_per_host 2 \
                                        --devices_name 2_a40 --model_name moe --param_num 10B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,4,1
    done

    exit 1
fi


if [ ${gpu_num} == 8 ]; then
    ########################
    #      For 8 GPU       #
    ########################

    ###########################
    #   Profile Wide-ResNet   #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile Wide-ResNet (500M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name wide_resnet --param_num 500M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Wide-ResNet (1B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name wide_resnet --param_num 1B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Wide-ResNet (2B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name wide_resnet --param_num 2B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Wide-ResNet (4B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name wide_resnet --param_num 4B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Wide-ResNet (6.8B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name wide_resnet --param_num 6.8B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
    done

    ###########################
    #      Profile Bert       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list_small[${i}]}
        # Profile Bert (760M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name bert --param_num 760M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Bert (1.3B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name bert --param_num 1.3B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Bert (2.6B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name bert --param_num 2.6B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Bert (6.7B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name bert --param_num 6.7B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile Bert (15B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name bert --param_num 15B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
    done

    ###########################
    #       Profile MoE       #
    ###########################
    for((i=0;i<${bs_num};i++))
    do
        batch_size=${batch_size_list[${i}]}
        # Profile MoE (690M)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name moe --param_num 690M \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile MoE (1.3B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name moe --param_num 1.3B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile MoE (2.4B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name moe --param_num 2.4B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile MoE (10B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name moe --param_num 10B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
        # Profile MoE (27B)
        python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 4 --num_devices_per_host 2 \
                                        --devices_name 4_a40 --model_name moe --param_num 27B \
                                        --batch_size ${batch_size} --niter 1 --warmup_num 1 --parallel_degrees=1,8,1
    done

    exit 1
fi
