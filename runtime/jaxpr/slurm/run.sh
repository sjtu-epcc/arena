
# salloc -p gpu-a40 -N 1 -n 1 --cpus-per-task 48 --gres=gpu:2 --exclusive
# salloc -p gpu-a40-bigdata -N 1 -n 1 --cpus-per-task 48 --gres=gpu:2 --exclusive

# salloc -p gpu-a10 -N 1 -n 1 --cpus-per-task 32 --gres=gpu:2 --exclusive
# salloc -p gpu-a10-bigdata -N 1 -n 1 --cpus-per-task 32 --gres=gpu:2 --exclusive

# singularity run --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp:/app/jaxpr/tmp" --bind "./profile_result:/app/crius_worker/jax/profile_result" --bind "./tmp:/app/crius_worker/jax/tmp" --bind "./tmp_res:/app/crius_worker/jax/tmp_res" --network host --nv -B /etc/libibverbs.d crius-profiler.sif

# singularity run --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp:/app/jaxpr/tmp" --bind "./profile_result:/app/crius_worker/jax/profile_result" --bind "./tmp:/app/crius_worker/jax/tmp" --bind "./tmp_res:/app/crius_worker/jax/tmp_res" --network host --nv -B /etc/libibverbs.d crius-profiler-cu112.sif

# singularity run --writable-tmpfs --bind "./prof_database:/app/crius_worker/jax/prof_database" --bind "./prof_log:/app/jaxpr/prof_log" --bind "./comm_data:/app/jaxpr/comm_data" --bind "./console_log/optimal:/app/tmp" --bind "./tmp:/app/jaxpr/tmp" --bind "./profile_result:/app/crius_worker/jax/profile_result" --bind "./tmp:/app/crius_worker/jax/tmp" --bind "./tmp_res:/app/crius_worker/jax/tmp_res" --network host --nv -B /etc/libibverbs.d crius-profiler-v2.sif

# . /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa
# ifconfig
# export NCCL_SOCKET_IFNAME=ib0.8068; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8; export NCCL_IB_HCA=mlx5,ibp; export NCCL_DEBUG=INFO

# # Reduce communication bottleneck in ib5 with high communication workload
# export NCCL_SOCKET_NTHREADS=8; export NCCL_NSOCKS_PERTHREAD=8

# ray stop --force

# export ENABLE_CRIUS_PROFILER=true
# bash jaxpr/cpp/install.sh; bash jaxpr/docker_setup.sh
# cp /home/bigdata/cyxue/crius_profile.py /app/jaxpr/crius_profile.py; cp /home/bigdata/cyxue/runtime_profiler.py /app/jaxpr/runtime_profiler.py
# cp /home/bigdata/cyxue/hlo_profiler.py /app/jaxpr/hlo_profiler.py; cp /home/bigdata/cyxue/estimate_all_a10.sh /app/jaxpr/estimate_all_a10.sh

# (A10)
# (Head) ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --node-ip-address 192.168.1.62 --port=6379 --num-cpus 16 --num-gpus 1 --object-store-memory 10737418240 --disable-usage-stats
# ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --node-ip-address 192.168.1.66 --port=6379 --num-cpus 8 --object-store-memory 21474836480 --disable-usage-stats

# (Worker) ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=192.168.1.62:6379 --node-ip-address 192.168.1.x --num-cpus 16 --num-gpus 1 --object-store-memory 10737418240 --disable-usage-stats
# ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=192.168.1.66:6379 --node-ip-address 192.168.1.x --num-cpus 8 --object-store-memory 21474836480 --disable-usage-stats

# ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=192.168.1.66:6379 --node-ip-address 192.168.1.59 --num-cpus 8 --object-store-memory 21474836480 --disable-usage-stats


# To evaluate tuner
# python jaxpr/runtime_profiler.py --optimize_with_alpa --overwrite_data --devices_name 4_a40 --num_devices_per_host 2 --num_hosts 4 --model_name wide_resnet --param_num 1B --batch_size 256 --num_micro_batches 16 --num_pipeline_layers 16 --disable_alpa_profiling_db --prune_search_space --prune_prompt 4_8_1_2_1_2

# python jaxpr/runtime_profiler.py --optimize_with_alpa --overwrite_data --devices_name 4_a40 --num_devices_per_host 2 --num_hosts 4 --model_name bert --param_num 1.3B --batch_size 128 --num_micro_batches 16 --num_pipeline_layers 16 --disable_alpa_profiling_db --prune_search_space --prune_prompt 1_4_2_8_2_8

# python jaxpr/runtime_profiler.py --optimize_with_alpa --overwrite_data --devices_name 8_a40 --num_devices_per_host 2 --num_hosts 8 --model_name wide_resnet --param_num 4B --batch_size 256 --num_micro_batches 16 --num_pipeline_layers 16 --disable_alpa_profiling_db --prune_search_space --prune_prompt 8_16_1_2_1_2 --overwrite_coarsened_layer_num none

# python jaxpr/runtime_profiler.py --optimize_with_alpa --overwrite_data --devices_name 8_a40 --num_devices_per_host 2 --num_hosts 8 --model_name bert --param_num 6.7B --batch_size 256 --num_micro_batches 16 --num_pipeline_layers 16 --disable_alpa_profiling_db --prune_search_space --prune_prompt 4_8_2_4_2_4 --overwrite_coarsened_layer_num none




# (A40)
# (Head) ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --node-ip-address 192.168.1.15 --port=6379 --num-cpus 16 --num-gpus 2 --object-store-memory 21474836480 --disable-usage-stats

# (Worker) ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=192.168.1.21:6379 --node-ip-address 192.168.1.22 --num-cpus 16 --num-gpus 2 --object-store-memory 21474836480 --disable-usage-stats

# (Head) ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --node-ip-address 192.168.1.30 --port=6379 --num-cpus 8 --num-gpus 2 --object-store-memory 21474836480 --disable-usage-stats

# (Worker) ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=192.168.1.30:6379 --node-ip-address 192.168.1.x --num-cpus 8 --num-gpus 2 --object-store-memory 21474836480 --disable-usage-stats


python jaxpr/runtime_profiler.py --measure_with_alpa --devices_name 1_a40 --num_hosts 1 --num_devices_per_host 1 --model_name moe --param_num 690M --batch_size 256 --num_micro_batches 16 --num_pipeline_layers 16 --niter 1 --warmup_num 1 --parallel_degrees=1,1,1


python jaxpr/runtime_profiler.py --measure_with_alpa --devices_name 2_a40 --num_hosts 2 --num_devices_per_host 2 --model_name bert --param_num 1.3B --batch_size 128 --num_micro_batches 16 --num_pipeline_layers 16 --niter 1 --warmup_num 1 --parallel_degrees=2,2,1


python jaxpr/runtime_profiler.py --measure_with_alpa --devices_name 8_a40 --num_hosts 8 --num_devices_per_host 2 --model_name moe --param_num 2.4B --batch_size 256 --num_micro_batches 16 --num_pipeline_layers 16 --niter 1 --warmup_num 1 --parallel_degrees=8,2,1


python jaxpr/runtime_profiler.py --estimate_e2e --num_hosts 8 --num_devices_per_host 2 --devices_name 8_a40 --model_name wide_resnet --param_num 2B --batch_size 1024 --niter 1 --warmup_num 1 --parallel_degrees=16,1,1


python jaxpr/runtime_profiler.py --estimate_e2e --num_hosts 8 --num_devices_per_host 2 --devices_name 8_a40 --model_name bert --param_num 2.6B --batch_size 512 --niter 1 --warmup_num 1 --parallel_degrees=8,2,1


python jaxpr/runtime_profiler.py --estimate_e2e --num_hosts 8 --num_devices_per_host 2 --devices_name 8_a40 --model_name moe --param_num 2.4B --batch_size 1024 --niter 1 --warmup_num 1 --parallel_degrees=8,2,1



# python crius_worker/jax/gen_prof_database.py --filename "/app/crius_worker/jax/prof_database/prof_database_4_a10_2_d.pkl" --max-comm-size-intra-node 32 --max-comm-size-inter-node 29 --cache-filename "/app/tmp/hlo_op_cost_dict.pkl"


# Compile error (even the smallest model) -> no overlay space in `df -h`? 
# Restart singularity container on head and all worker nodes solve the problem.
# Delete `profile-results.npy` can release overlay space.

# Compile error after getting optimal parallelism before lanuching executatbles <- too small layer num.

# Heavy bottleneck with ib5 with high workload (long_t / short_t > 10)? 
# The reason might be traffic blocking with ib5 switch board. Adding thread num in socket communication can
# alleviate this phenomena.




# # For 1-node A40 
# # 1 GPU
# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 1 -m wide_resnet -p 500M -e 5 -w 2 | tee /app/tmp/1_a40_nodes_1_devices_per_node_wide_resnet_500M.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 1 -m bert -p 760M -e 5 -w 2 | tee /app/tmp/1_a40_nodes_1_devices_per_node_bert_760M.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 1 -m moe -p 690M -e 5 -w 2 | tee /app/tmp/1_a40_nodes_1_devices_per_node_moe_690M.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 1 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_1_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 1 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_1_devices_per_node_bert_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 1 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_1_devices_per_node_moe_1.3B.log


# # 2 GPUs
# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m wide_resnet -p 500M -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_wide_resnet_500M.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m bert -p 760M -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_bert_760M.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m moe -p 690M -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_moe_690M.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_bert_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_moe_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 1_a40 -n 1 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/1_a40_nodes_2_devices_per_node_moe_2.4B.log


# For 2-node A40 (4 GPUs)
# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m wide_resnet -p 500M -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_wide_resnet_500M.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m bert -p 760M -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_bert_760M.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m moe -p 690M -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_moe_690M.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_bert_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_moe_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_moe_2.4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m wide_resnet -p 4B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_wide_resnet_4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m bert -p 6.7B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_bert_6.7B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a40 -n 2 -d 2 -m moe -p 10B -e 5 -w 2 | tee /app/tmp/2_a40_nodes_2_devices_per_node_moe_10B.log


# For 4-node A40 (8 GPUs)
# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_bert_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_moe_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_moe_2.4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m wide_resnet -p 4B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_wide_resnet_4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m bert -p 6.7B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_bert_6.7B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m moe -p 10B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_moe_10B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m wide_resnet -p 6.8B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_wide_resnet_6.8B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m bert -p 15B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_bert_15B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a40 -n 4 -d 2 -m moe -p 27B -e 5 -w 2 | tee /app/tmp/4_a40_nodes_2_devices_per_node_moe_27B.log


# For 8-node A40 (16 GPUs)
# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_moe_2.4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m wide_resnet -p 4B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_wide_resnet_4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m bert -p 6.7B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_bert_6.7B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m moe -p 10B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_moe_10B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m wide_resnet -p 6.8B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_wide_resnet_6.8B.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m bert -p 15B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_bert_15B.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m moe -p 27B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_moe_27B.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m wide_resnet -p 13B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_wide_resnet_13B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m bert -p 39B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_bert_39B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a40 -n 8 -d 2 -m moe -p 70B -e 5 -w 2 | tee /app/tmp/8_a40_nodes_2_devices_per_node_moe_70B.log



# # For 2-node A10 (4 GPUs)

# bash ./jaxpr/optimize_all_configs.sh -x 2_a10 -n 2 -d 2 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/2_a10_nodes_2_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a10 -n 2 -d 2 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/2_a10_nodes_2_devices_per_node_bert_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a10 -n 2 -d 2 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/2_a10_nodes_2_devices_per_node_moe_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a10 -n 2 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/2_a10_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a10 -n 2 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/2_a10_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 2_a10 -n 2 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/2_a10_nodes_2_devices_per_node_moe_2.4B.log


# For 4-node A10 (8 GPUs)

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m wide_resnet -p 500M -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_wide_resnet_500M.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m bert -p 760M -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_bert_760M.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m moe -p 690M -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_moe_690M.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_bert_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_moe_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_moe_2.4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m wide_resnet -p 4B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_wide_resnet_4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m bert -p 6.7B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_bert_6.7B.log

# bash ./jaxpr/optimize_all_configs.sh -x 4_a10 -n 4 -d 2 -m moe -p 10B -e 5 -w 2 | tee /app/tmp/4_a10_nodes_2_devices_per_node_moe_10B.log


# For 8-node A10 (16 GPUs)

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m wide_resnet -p 500M -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_wide_resnet_500M.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m bert -p 760M -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_bert_760M.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m moe -p 690M -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_moe_690M.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m wide_resnet -p 1B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_wide_resnet_1B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m bert -p 1.3B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_bert_1.3B.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m moe -p 1.3B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_moe_1.3B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m wide_resnet -p 2B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_wide_resnet_2B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m bert -p 2.6B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_bert_2.6B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m moe -p 2.4B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_moe_2.4B.log

# bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m wide_resnet -p 4B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_wide_resnet_4B.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m bert -p 6.7B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_bert_6.7B.log

# !!! bash ./jaxpr/optimize_all_configs.sh -x 8_a10 -n 8 -d 2 -m moe -p 10B -e 5 -w 2 | tee /app/tmp/8_a10_nodes_2_devices_per_node_moe_10B.log
