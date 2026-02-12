# Arena Artifact Evaluation for EuroSys'26

Arena is a large model training system to dynamically schedule and efficiently execute large models with adaptive parallelism in GPU clusters.
We provide the source codes and the benchmarking scripts to reproduce the major experimental results of Arena. 

The major claims of Arena system include:

1. Arena's disaggregated profiler achieves average error
rates of 4.4%, 5.1%, 3.1%, 4.6%, and 8.3% for 1, 2, 4, 8, and 16 GPU cases; Arena reduces the GPU time (i.e., elapsed time × occupied GPU count) by 18.1x on average (2.55x at least), as compared to direct measurement.
2. In Arena's parallelism planner, the best proxy plan (used for scheduling) among grids achieves average 93.4% performance of the AP searched optimal plan, thus is accurate enough to achieve AP-aware cluster scheduling.
3. With AP-aware scheduling, Arena scheduler reduces average job completion time (JCT) by 81.3% (FCFS), 80.5% (ElasticFlow-LS), 76.6% (Gavel) and 75.2% (Sia), completing up to 1.45x more jobs. From the cluster perspective, Arena outperforms baselines with up to 1.55x higher average throughput and 1.58x higher peak throughput.

Since the full-fleet evaluation involves tens to hundreds of GPUs, for reproducibility, the artifact mainly uses 4 A40 GPUs (see hardware dependencies below) unless specified.


## 1. Dependencies & Installation

### 1.1. Hardware dependencies.

The artifact requires a Linux system equipped with at least 192 GB of system memory, 256 GB of available disk storage, and 4 NVIDIA A40 GPUs (48GB, connected via PCIe or NVLink). 

### 1.2. Software dependencies.

The artifact requires Conda for package management. The software stack includes CUDA 11.8 and Python3.8. All software dependencies are automatically installed in our provided Dockerfile. 

### 1.3. How to access.

The Arena system is available at GitHub: https://github.com/sjtu-epcc/arena/tree/ae-eurosys#. Users can clone the GitHub repository.

```bash
git clone --recursive -b ae-eurosys https://github.com/sjtu-epcc/arena.git
cd arena
git checkout ae-eurosys
```

### 1.4. How to install.

We provide a Dockerfile to prepare the software dependencies.

```bash
cd runtime
docker build -t arena/arena:ae-eurosys -f ./profile_cu118.Dockerfile .
docker run --runtime=nvidia -it --rm --gpus all --shm-size 64g --network=host --privileged --volume [USER_DIR]/.cache:/root/.cache --env NVIDIA_DISABLE_REQUIRE=1 --name arena arena/arena:ae-eurosys
conda activate alpa         # Alpa env
bash jaxpr/cpp/install.sh   # Build kernel-level profiler
```

## 2. Evaluation Workflow

### 2.1. Efficiency of disaggregated profiling.

We provide the instructions to run the single-device profiler and the multi-device direct execution to evaluate the accuracy and profiling cost reduction.

Before running single-device profiling, users should offline profile the communication latency data (may take dozens of minutes or a few hours):

```bash
# Get default network interface (e.g., eno1)
export NET_IF=$(route | grep default | grep -o "eno.")
# Specify the network interface
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Stop ray processes (optional)
ray stop --force

# Start ray process on head node
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats
# Start ray process on worker node(s)
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=[HEAD_NODE_IP]:6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Offline profile P2P communication (only between 2 GPUs)
ulimit -c unlimited -n 65536 && python jaxpr/communication.py --profile_p2p --devices_name 1_a40 --num_hosts 1 --num_devices_per_host 2 --overwrite_data
# Offline profile collective communication
ulimit -c unlimited -n 65536 && python jaxpr/communication.py --profile_collective --devices_name 1_a40 --num_hosts 1 --num_devices_per_host 4 --overwrite_data --only_best_locality
```

To run single-device profiling with (1) specified model configurations (layers are uniformly clustered into stages), (2) specified device assignment, (3) specified parallelism (`--parallel_degrees={pp}_{dp}_{tp}`, GPUs are correspondingly sharded).
Taking vanilla pipeline parallelism (1F1B) on 4 GPUs as an example:

```bash
# (Optional) Envs
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
# Enable arena profiler
export ENABLE_CRIUS_PROFILER=true
# Specify one GPU
export CUDA_VISIBLE_DEVICES=0
# Profile
python jaxpr/runtime_profiler.py --estimate_e2e --num_hosts 1 --num_devices_per_host 4 --devices_name 1_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --num_micro_batches 16 --niter 5 --warmup_num 2 --parallel_degrees=4,1,1

# NOTE: Other argument descriptions are given and described in `./runtime/jaxpr/runtime_profiler.py`.
```

To measure the end-to-end iteration time (rather than estimating it) with the specified parallel plan, use `--measure_with_alpa` instead of `--estimate_e2e`. Notably, the auto parallelizing techniques of Alpa are not actually used here, instead this step only uses the most basic training functions with specified parallelism. Users should first establish a Ray cluster, then execute the following commands on the head node to perform vanilla pipeline parallelism training with 4 GPUs on 1 node:

```bash
# Specify all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Get default network interface (e.g., eno1)
export NET_IF=$(route | grep default | grep -o "eno.")
# Specify the network interface
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Stop ray processes (optional)
ray stop --force

# Start ray process on head node
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Start ray process on worker node(s)
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=[HEAD_NODE_IP]:6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Disable arena profiler
export ENABLE_CRIUS_PROFILER=false
# Profile
python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 4 --devices_name 1_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --niter 5 --warmup_num 2 --parallel_degrees=4,1,1
```

In our 1x4 A40 node, the estimated/measured e2e iteration time is 15.893s/16.121s with the profiling cost of 63.475/613.93 #GPU x time(s). This could be slightly different compared to Figure 16's results due to the usage of different hardware. 
As listed in Table 2 and `./runtime/crius_worker/jax/configs.py`, users can flexibly set the model, number of parameters, batch size, and parallelism degrees to reproduce the results in Figure 16.


### 2.2. Effectiveness of parallelism planning.

We then provide the instructions to run Arena's parallelism planner to evaluate the performance of the best proxy plan (used for cluster scheduling) among all grids, compared to the AP searched optimal plan. 

Within a grid (with fixed resources and number of pipeline stages), the parallelism planning process includes the following steps: (1) Cluster layers with best-effort balance on layer FLOPs and minimal inter-stages communication; (2) Shard GPUs with GPU fraction allocated to each layer; (3) Enumerate the Pareto-optimal parallelism plans within the grid:

```bash
# Enable arena profiler
export ENABLE_CRIUS_PROFILER=true
# Specify one GPU
export CUDA_VISIBLE_DEVICES=0
# Profile
python jaxpr/runtime_profiler.py --estimate_e2e --num_hosts 1 --num_devices_per_host 4 --devices_name 1_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --niter 5 --warmup_num 2 --enable_cell_profile --num_pipeline_stages=4 --cell_prof_strategy=auto

# Optional strategies of grid (i.e., cell) profiling: 
# - "auto" (default): Use the parallel plan hints generated by the pipeline planner to select the theoratically best-performing plan as the target parallel plan.
# - "minimal": Only profile two parallel plans (vanilla data and tensor parallelism). 
# - "uniform": Profile all uniform (symmetric) parallel plans (e.g., `(4,1,1)`, `(1,2,2)`) with contraint of #stages, including hybrid ones.
```

To evaluate the parallelism planner among all grids, i.e., the best proxy plan (used for scheduling this job) among all grids:

```bash
# Enable arena profiler
export ENABLE_CRIUS_PROFILER=true
# Specify one GPU
export CUDA_VISIBLE_DEVICES=0
# Profile
python jaxpr/crius_cell_profile.py --num_hosts 1 --num_devices_per_host 4 --devices_name 1_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --num_micro_batches 16 --niter 5 --warmup_num 2 --cell_prof_strategy=auto
```

To measure the performance of the optimal parallelism plan searched by Alpa, use `--optimize_with_alpa`:

```bash
# Specify all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Get default network interface (e.g., eno1)
export NET_IF=$(route | grep default | grep -o "eno.")
# Specify the network interface
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Stop ray processes (optional)
ray stop --force

# Start ray process on head node
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Start ray process on worker node(s)
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=[HEAD_NODE_IP]:6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Disable arena profiler
export ENABLE_CRIUS_PROFILER=false
# Profile
python jaxpr/runtime_profiler.py --optimize_with_alpa --num_hosts 1 --num_devices_per_host 4 --devices_name 1_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --niter 5 --warmup_num 2
```

In our 1x4 A40 node, the arena-estimated/alpa-optimized e2e iteration time is 10.445s/10.633s. Similar to 2.1, users can flexibly set model configurations and hyperparameters.



### 2.3. Large-scale simulated scheduling experiments.

The simulation requires offline profiling all training jobs by enumerating possible combinations of models (e.g., GPT-1.3B), hyperparameters (e.g., global batch size 256), and allocated hardware (e.g., 1x4 A40 GPUs) as listed in Table 1 and 2. 
Here, to avoid extensive offline profiling for artifact evaluation, we have provided our profiling data in `./database/prof_database.pkl` (raw data in `./runtime/jaxpr/prof_log/`) that includes Arena's estimated data, data profiled via data parallelism, and Alpa's searched data. 

To run large-scale simulated scheduling with 1,280 GPUs and Philly trace (Figure 11, Figure 12), use the following instructions (`[POLICY]` includes `fcfs`, `elasticflow-l`, `gavel`, and `sia`):

```bash
cd ./
# For Arena
python simulator.py --policy=crius --trace_type=philly --sched_with_opt --max_sched_round=2000 --enable_alpa --result_dir=./plot
# For other baselines
python simulator.py --policy=[POLICY] --trace_type=philly --max_sched_round=2000 --enable_alpa --result_dir=./plot
```

We also provide scripts to visualize the results (`[METRIC]` includes `thr`, `jct`, and `queuing_time`):

```bash
python simulator.py --visual --visualized_metric=[METRIC] --result_dir=./plot --out_dir=./figures --trace_type=philly
```

For throughput metric, users can both inspect average/maximum values in console logs and visualized curves (Figure 11) in `./figures/cluster_thr.pdf`.
For JCT, number of finished jobs, and queuing time metrics, users can inspect results in console logs.


