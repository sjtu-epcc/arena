# Documentation for Runtime Orchestration

## 0. Build Your Docker Container Locally

Execute following commands to build your customized Docker image locally:

```bash
# Build image from dockerfile (in `./Crius` dir, as the `/app` dir)
docker build --network host -t dicardo/crius-runtime:v1.1 -f ./runtime/client_cu112.Dockerfile .

# Modify alpa source code 
docker run --runtime=nvidia -it --rm --gpus all --shm-size=10.24gb --network=host --privileged dicardo/crius-runtime:v1.1

# Path of alpa dir: `/opt/conda/envs/alpa/lib/python3.8/site-packages/alpa`
# Github link of crius-customized alpa: git clone --recursive https://github.com/DicardoX/custom_alpa.git
# 1. Replace `alpa/global_env.py` with `custom_alpa/global_env.py`
# 2. Replace `alpa/pipeline_parallel/stage_profiling.py` with `custom_alpa/pipeline_parallel/stage_profiling.py`
# 3. Replace `alpa/pipeline_parallel/stage_construction.py` with `custom_alpa/pipeline_parallel/stage_construction.py`

# Commit modified container as the docker image
docker commit -a "dicardo" -m "test" "[CONTAINER_ID]" dicardo/crius-runtime:v1.1

# Push to dockerhub
docker push dicardo/crius-runtime:v1.1

```



## 1. Deploy as Singularity Container

### 1.1 Setup Container Environment

```bash
# Pull docker image
singularity pull --force crius-runtime.sif docker://dicardo/crius-runtime:v1.1

# Startup singularity container
cd cyxue; singularity run --writable-tmpfs --bind "./prof_database:/app/runtime/crius_worker/jax/prof_database" --bind "./prof_log:/app/runtime/jaxpr/prof_log" --bind "./comm_data:/app/runtime/jaxpr/comm_data" --bind "./console_log/optimal:/app/runtime/tmp" --bind "./runtime_res:/app/runtime/plot" --bind "./tmp:/app/runtime/jaxpr/tmp" --bind "./profile_result:/app/runtime/crius_worker/jax/profile_result" --bind "./tmp:/app/runtime/crius_worker/jax/tmp" --bind "./tmp_res:/app/runtime/crius_worker/jax/tmp_res" --network host --nv -B /etc/libibverbs.d crius-runtime.sif

# Activate conda env
. /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa; export NCCL_SOCKET_IFNAME=ib0.8068; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8; export NCCL_IB_HCA=mlx5,ibp; export NCCL_SOCKET_NTHREADS=8; export NCCL_NSOCKS_PERTHREAD=8

# Stop ray cluster
ray stop --force

# Check ifconfig
ifconfig

# Export necessary envs
# `ib0.8068` is the network interface of inifiband connection
export NCCL_SOCKET_IFNAME=ib0.8068; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8; export NCCL_IB_HCA=mlx5,ibp; export NCCL_SOCKET_NTHREADS=8; export NCCL_NSOCKS_PERTHREAD=8; ifconfig

```


### 1.2 Startup Daemon as Head/Worker Runtime

On head container, we execute the following commands:

```bash
# Customize the json file of cluster spec based on available resources. 
cd /app
vim runtime/cluster_spec.json

# Copy newly modified files into container
cp /home/bigdata/cyxue/crius_runtime.py /app/runtime/crius_runtime.py; cp /home/bigdata/cyxue/worker_runtime.py /app/runtime/crius_worker/worker_runtime.py; cp /home/bigdata/cyxue/runtime_profiler.py /app/runtime/jaxpr/runtime_profiler.py; cp /home/bigdata/cyxue/cluster_spec.json /app/runtime/cluster_spec.json; cp /home/bigdata/cyxue/db_querier.py /app/db_querier.py; cp /home/bigdata/cyxue/dp_mem.json /app/database/dp_mem.json; cp /home/bigdata/cyxue/gavel_sched.py /app/baselines/gavel_sched.py



# (After all worker daemons are startup) Execute runtime orchestration
python runtime/crius_runtime.py --policy crius --sched_with_opt --max_sched_round 72 --overwrite_net_if ib0.8068 --trace_name "runtime_trace.csv" | tee /home/bigdata/cyxue/output.log

python runtime/crius_runtime.py --policy gavel --sched_with_opt --max_sched_round 72 --overwrite_net_if ib0.8068 --trace_name "runtime_trace.csv" | tee /home/bigdata/cyxue/output_gavel.log

python runtime/crius_runtime.py --policy fcfs --sched_with_opt --max_sched_round 72 --overwrite_net_if ib0.8068 --trace_name "runtime_trace.csv" | tee /home/bigdata/cyxue/output_fcfs.log


# For dummy test
python runtime/crius_runtime.py --policy crius --sched_with_opt --max_sched_round 10 --overwrite_net_if ib0.8068 --trace_name "dummy_runtime_trace.csv"

```


On worker container (one for each host), we execute the following commands:

```bash
# Startup worker daemon
cd /app; python runtime/crius_worker/worker_runtime.py

```
