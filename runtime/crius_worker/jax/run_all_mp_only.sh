# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: Run multiple profiling with varying configurations within forcing model parallelism.
#         Note that this script should be executed after all preparation work is done.
#         (VITAL) This script is temporary and should be modified based on runtime hardware environment.

###########################
#   Expected for 2 GPUs   #
###########################
# Profile Wide-ResNet (500M)
bash ./run.sh -x 2_1080ti -n 1 -d 2 -m wide_resnet -p 500M -t 1 -g
# Profile Bert (760M)
bash ./run.sh -x 2_1080ti -n 1 -d 2 -m bert -p 760M -t 1 -g
# Profile MoE (690M)
bash ./run.sh -x 2_1080ti -n 1 -d 2 -m moe -p 690M -t 1 -g


# ###########################
# #   Expected for 4 GPUs   #
# ###########################
# # Profile Wide-ResNet (1B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m wide_resnet -p 1B -t 1 -g
# # Profile Bert (1.3B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m bert -p 1.3B -t 1 -g
# # Profile MoE (1.3B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m moe -p 1.3B -t 1 -g


# ###########################
# #   Expected for 8 GPUs   #
# ###########################
# # Profile Wide-ResNet (2B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m wide_resnet -p 2B -t 1 -g
# # Profile Bert (2.6B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m bert -p 2.6B -t 1 -g
# # Profile MoE (2.4B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m moe -p 2.4B -t 1 -g


# ###########################
# #   Expected for 16 GPUs  #
# ###########################
# # Profile Wide-ResNet (4B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m wide_resnet -p 4B -t 1 -g
# # Profile Bert (6.7B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m bert -p 6.7B -t 1 -g
# # Profile MoE (10B)
# bash ./run.sh -x 2_1080ti -n 2 -d 2 -m moe -p 10B -t 1 -g
