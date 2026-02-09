# !/usr/bin/env bash
# Author: Chunyu Xue

# @ Info: Run multiple profiling with varying configurations through Nsight Systems tool.
#         Note that this script should be executed after all preparation work is done.
#         (VITAL) This script is temporary and should be modified based on runtime hardware environment.

export DEVICE_NAME=1_a40
export NODE_NUM=1
export DEVICE_NUM=2

# ###########################
# #    Wide-ResNet (500M)   #
# ###########################
# # Data parallelism
# export OUTPUT_FILE=nsys-rep/wrn_500m_dp_output.qdrep
# nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m wide_resnet -p 500M -b 256 -l 16 -c 16 -o
# # Pipeline
# export OUTPUT_FILE=nsys-rep/wrn_500m_pp_output.qdrep
# nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m wide_resnet -p 500M -b 256 -l 16 -c 16 -f
# # Model parallelism
# export OUTPUT_FILE=nsys-rep/wrn_500m_mp_output.qdrep
# nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m wide_resnet -p 500M -b 256 -l 16 -c 16 -g

###########################
#     Wide-ResNet (1B)    #
###########################
# # Data parallelism
# export OUTPUT_FILE=nsys-rep/wrn_1b_dp_output.qdrep
# nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m wide_resnet -p 1B -b 256 -l 16 -c 16 -o
# Pipeline
export OUTPUT_FILE=nsys-rep/wrn_1b_pp_output.qdrep
nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m wide_resnet -p 1B -b 256 -l 16 -c 16 -f
# # Model parallelism
# export OUTPUT_FILE=nsys-rep/wrn_1b_mp_output.qdrep
# nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m wide_resnet -p 1B -b 256 -l 16 -c 16 -g

###########################
#       Bert (760M)       #
###########################
# # Data parallelism
# export OUTPUT_FILE=nsys-rep/bert_760m_dp_output.qdrep
# nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m bert -p 760M -b 256 -l 16 -c 16 -o
# Pipeline
export OUTPUT_FILE=nsys-rep/bert_760m_pp_output.qdrep
nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m bert -p 760M -b 256 -l 16 -c 16 -f
# Model parallelism
export OUTPUT_FILE=nsys-rep/bert_760m_mp_output.qdrep
nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m bert -p 760M -b 256 -l 16 -c 16 -g

###########################
#        MoE (690M)       #
###########################
# Data parallelism
export OUTPUT_FILE=nsys-rep/moe_690m_dp_output.qdrep
nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m moe -p 690M -b 256 -l 16 -c 16 -o
# Pipeline
export OUTPUT_FILE=nsys-rep/moe_690m_pp_output.qdrep
nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m moe -p 690M -b 256 -l 16 -c 16 -f
# Model parallelism
export OUTPUT_FILE=nsys-rep/moe_690m_mp_output.qdrep
nsys profile -o ${NSYS_PATH}/${OUTPUT_FILE} --gpu-metrics-device=all -t cuda,nvtx,osrt,cudnn,cublas -f true bash ./profile.sh -x ${DEVICE_NAME} -n ${NODE_NUM} -d ${DEVICE_NUM} -m moe -p 690M -b 256 -l 16 -c 16 -g
