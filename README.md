# Arena Artifact Evaluation for EuroSys'26

We provide the source code and the benchmarking scripts to reproduce the major experimental results of Arena. Notably, these codes are only for artifact evaluation and have not been organized as the official open-source version of Arena. 

The major claims of Arena system include:

- Arena's disaggregated profiler achieves average error
rates of 4.4%, 5.1%, 3.1%, 4.6%, and 8.3% for 1, 2, 4, 8, and 16 GPU cases; Arena reduces the GPU time (i.e., elapsed time × occupied GPU count) by 18.1× on average (2.55× at least), as compared to direct measurement.
- XXX

Since the full-fleet evaluation involves tens to hundreds of GPUs, for reproducibility, the artifact provides the following benchmarking tests:

- Disaggregated profiling experiment (Figure 16);
- Pareto frontier deduction (Figure 14);
- Large-scale simulated scheduling experiment with 1,280 GPUs and Philly trace (Figure 11, Figure 12);
- Large-scale simulated scheduling experiment with 1,280 GPUs and Helios/PAI traces (Figure 13).


## 1. Dependencies & Installation

### 1.1. Hardware dependencies.

The artifact requires a Linux system equipped with at least 192 GB of system memory, 256 GB of available disk storage, and 4 NVIDIA A40 GPUs. 

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
docker run --runtime=nvidia -it --rm --gpus all --shm-size 64g --network=host --privileged --volume [USER_DIR]/.cache:/root/.cache --env NVIDIA_DISABLE_REQUIRE=1 --name arena dicardo/arena:ae-eurosys
conda activate alpa         # Alpa env
bash jaxpr/cpp/install.sh   # Build kernel-level profiler
```

## 2. Benchmarking Workflow

### 2.1. Efficiency of disaggregated profiling.

We provide the instructions to run the single-device profiler and the multi-device direct execution to evaluate the accracy and profiling cost reduction.

To run single-device profiling with (1) specified model configurations (layers are uniformly clustered into stages), (2) specified device assignment, (3) specified parallelism (`--parallel_degrees={pp}_{dp}_{tp}`, GPUs are correspondingly sharded).

For example, to profile vanilla pipeline parallelism (1F1B) on 4 GPUs:

```bash
# (Optional) Envs
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
# Enable crius profiler
export ENABLE_CRIUS_PROFILER=true
# Specify one GPU
export CUDA_VISIBLE_DEVICES=0
# Profile
python jaxpr/runtime_profiler.py --estimate_e2e --num_hosts 1 --num_devices_per_host 4 --devices_name 2_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --num_micro_batches 16 --niter 5 --warmup_num 2 --parallel_degrees=4,1,1

# NOTE: Other argument descriptions are given and described in `./runtime/jaxpr/runtime_profiler.py`.
```

To measure the end-to-end iteration time (rather than estimating it) with the specified parallel plan, use `--measure_with_alpa` instead of `--estimate_e2e`. It should be noted that the auto parallelizing techniques of alpa are not actually used here, instead we only use the most basic executing function to manually profile the model with specified parallel plan.

Since Crius profiler is disabled, we must execute training on `N` GPUs, where `N` equals to the user-required resource quota.

Before profiling, we first need to establish a Ray cluster for potential multi-hosts training scenarios:

```bash
# Specify one GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Get default network interface (e.g., eno1)
export NET_IF=$(route | grep default | grep -o "eno.")
# Specify the network interface
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for jax
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Start ray process on head node
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Start ray process on worker node(s)
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --address=[HEAD_NODE_IP]:6379 --num-gpus 4 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats

# Stop ray processes (optional)
ray stop --force
```

Then, we execute the following commands on the head node to perform vanilla pipeline parallelism training with 4 GPUs on 1 node:

```bash
# Disable crius profiler
export ENABLE_CRIUS_PROFILER=false
# Profile
python jaxpr/runtime_profiler.py --measure_with_alpa --num_hosts 1 --num_devices_per_host 4 --devices_name 1_a40 --model_name wide_resnet --param_num 1B --batch_size 256 --niter 5 --warmup_num 2 --parallel_degrees=4,1,1
```

In our 1x4 A40 node, the estimated/measured e2e iteration time is 15.893s/16.121s with the profiling cost of 22.859/613.93 #GPU x time(s). This could be slightly different compared to Figure 16's results due to the usage of different hardware. 
As listed in Table 2 and `./runtime/crius_worker/jax/configs.py`, users can flexibly set the model, number of parameters, batch size, and parallelism degrees to reproduce the results in Figure 16.


### 2.2. Effectiveness of parallelism planning.

TBD

### 2.3. Large-scale simulated scheduling experiments.

TBD

