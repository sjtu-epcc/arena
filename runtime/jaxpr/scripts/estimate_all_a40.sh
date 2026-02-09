# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: Estimate performance for all parallelism configurations.


batch_size_list=("256" "512" "1024")
batch_size_list_small=("128" "256" "512")
bs_num=3


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
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 500M \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (1B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 1B \
                                  --batch_size ${batch_size}
done

###########################
#      Profile Bert       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list_small[${i}]}
    # Profile Bert (760M)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 1 --model_name bert --param_num 760M \
                                  --batch_size ${batch_size}
    # Profile Bert (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 1 --model_name bert --param_num 1.3B \
                                  --batch_size ${batch_size}
done

###########################
#       Profile MoE       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile MoE (690M)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 1 --model_name moe --param_num 690M \
                                  --batch_size ${batch_size}
    # Profile MoE (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 1 --model_name moe --param_num 1.3B \
                                  --batch_size ${batch_size}
done



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
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 500M \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 500M \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (1B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 1B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 1B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (2B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 2B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 2B \
                                  --batch_size ${batch_size}
done

###########################
#      Profile Bert       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list_small[${i}]}
    # Profile Bert (760M)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name bert --param_num 760M \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name bert --param_num 760M \
                                  --batch_size ${batch_size}
    # Profile Bert (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name bert --param_num 1.3B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name bert --param_num 1.3B \
                                  --batch_size ${batch_size}
    # Profile Bert (2.6B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name bert --param_num 2.6B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name bert --param_num 2.6B \
                                  --batch_size ${batch_size}
done

###########################
#       Profile MoE       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile MoE (690M)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name moe --param_num 690M \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name moe --param_num 690M \
                                  --batch_size ${batch_size}
    # Profile MoE (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name moe --param_num 1.3B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name moe --param_num 1.3B \
                                  --batch_size ${batch_size}
    # Profile MoE (2.4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 1_a40 --num_hosts 1 \
                                  --num_devices_per_host 2 --model_name moe --param_num 2.4B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 1 --model_name moe --param_num 2.4B \
                                  --batch_size ${batch_size}
done


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
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 500M \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 500M \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (1B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 1B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 1B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (2B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 2B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 2B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 4B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name wide_resnet --param_num 4B \
                                  --batch_size ${batch_size}
done


###########################
#      Profile Bert       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list_small[${i}]}
    # Profile Bert (760M)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name bert --param_num 760M \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name bert --param_num 760M \
                                  --batch_size ${batch_size}
    # Profile Bert (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name bert --param_num 1.3B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name bert --param_num 1.3B \
                                  --batch_size ${batch_size}
    # Profile Bert (2.6B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name bert --param_num 2.6B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name bert --param_num 2.6B \
                                  --batch_size ${batch_size}
    # Profile Bert (6.7B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name bert --param_num 6.7B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name bert --param_num 6.7B \
                                  --batch_size ${batch_size}
done


###########################
#       Profile MoE       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile MoE (690M)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name moe --param_num 690M \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name moe --param_num 690M \
                                  --batch_size ${batch_size}
    # Profile MoE (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name moe --param_num 1.3B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name moe --param_num 1.3B \
                                  --batch_size ${batch_size}
    # Profile MoE (2.4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name moe --param_num 2.4B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name moe --param_num 2.4B \
                                  --batch_size ${batch_size}
    # Profile MoE (10B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 2_a40 --num_hosts 2 \
                                  --num_devices_per_host 2 --model_name moe --param_num 10B \
                                  --batch_size ${batch_size}
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 1 --model_name moe --param_num 10B \
                                  --batch_size ${batch_size}
done


########################
#      For 8 GPU       #
########################

###########################
#   Profile Wide-ResNet   #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile Wide-ResNet (1B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 1B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (2B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 2B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 4B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (6.8B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 6.8B \
                                  --batch_size ${batch_size}
done


###########################
#      Profile Bert       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list_small[${i}]}
    # Profile Bert (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 2 --model_name bert --param_num 1.3B \
                                  --batch_size ${batch_size}
    # Profile Bert (2.6B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name bert --param_num 2.6B \
                                  --batch_size ${batch_size}
    # Profile Bert (6.7B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name bert --param_num 6.7B \
                                  --batch_size ${batch_size}
    # Profile Bert (15B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name bert --param_num 15B \
                                  --batch_size ${batch_size}
done


###########################
#       Profile MoE       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile MoE (1.3B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4  \
                                  --num_devices_per_host 2 --model_name moe --param_num 1.3B \
                                  --batch_size ${batch_size}
    # Profile MoE (2.4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name moe --param_num 2.4B \
                                  --batch_size ${batch_size}
    # Profile MoE (10B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name moe --param_num 10B \
                                  --batch_size ${batch_size}
    # Profile MoE (27B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 4_a40 --num_hosts 4 \
                                  --num_devices_per_host 2 --model_name moe --param_num 27B \
                                  --batch_size ${batch_size}
done


########################
#      For 16 GPU      #
########################

###########################
#   Profile Wide-ResNet   #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile Wide-ResNet (2B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 2B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 4B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (6.8B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 6.8B \
                                  --batch_size ${batch_size}
    # Profile Wide-ResNet (13B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name wide_resnet --param_num 13B \
                                  --batch_size ${batch_size}
done


###########################
#      Profile Bert       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list_small[${i}]}
    # Profile Bert (2.6B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name bert --param_num 2.6B \
                                  --batch_size ${batch_size}
    # Profile Bert (6.7B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name bert --param_num 6.7B \
                                  --batch_size ${batch_size}
    # Profile Bert (15B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name bert --param_num 15B \
                                  --batch_size ${batch_size}
    # Profile Bert (39B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name bert --param_num 39B \
                                  --batch_size ${batch_size}
done


###########################
#       Profile MoE       #
###########################
for((i=0;i<${bs_num};i++))
do
    batch_size=${batch_size_list[${i}]}
    # Profile MoE (2.4B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name moe --param_num 2.4B \
                                  --batch_size ${batch_size}
    # Profile MoE (10B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name moe --param_num 10B \
                                  --batch_size ${batch_size}
    # Profile MoE (27B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name moe --param_num 27B \
                                  --batch_size ${batch_size}
    # Profile MoE (70B)
    python jaxpr/crius_profile.py --estimate_e2e --profile_all_configs --devices_name 8_a40 --num_hosts 8 \
                                  --num_devices_per_host 2 --model_name moe --param_num 70B \
                                  --batch_size ${batch_size}
done
