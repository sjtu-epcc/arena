# Documentation of Crius Worker

> We use Docker/Container methods in the implementation of Crius worker to be fast deployed on cluster nodes.

--------------

## 1. Run Docker Container on Each Node

First, we enter the work directory of the node and create the following folders:

```bash
# In work dir 
mkdir prof_database         # To store the profiling database of basic operators
chmod 777 prof_database
mkdir profile_result        # To store the profiling result of auto parallelism
chmod 777 profile_result
```
These folders will be mounted into the container.

Then, we run a container on each node (without interactive mode) with the following commands:

```bash
docker run --name crius-worker --runtime=nvidia -it --rm --gpus all --shm-size=11.00gb --network=host --volume ./profile_result:/app/jax/profile_result --volume ./prof_database:/app/jax/prof_database --privileged dicardo/crius-worker:v1
# Absolute path version
docker run --name crius-worker --runtime=nvidia -it --rm --gpus all --shm-size=11.00gb --network=host --volume /home/cyxue/Projects/playground/crius_worker/profile_result:/app/jax/profile_result --volume /home/cyxue/Projects/playground/crius_worker/prof_database:/app/jax/prof_database  --privileged dicardo/crius-worker:v1
```

Thus, after entering the docker images, we first apply the following commands on each node:

```bash
# Activate alpa env, if come to `CommandNotFoundError`, use `source activate` instead.
conda activate alpa
# Work dir
cd jax
# Get default network interface
NET_IF=$(route | grep default | grep -o "eno.")
echo "${NET_IF}"
# Specify the network interface based on the using situation in `ifconfig`
export NCCL_SOCKET_IFNAME=${NET_IF}
# Specify the fraction of pre-allocated memory for JAX
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

then, we run `ray start` command on each node:

- On head node: 

```bash
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats
```

Then, we can use `ctrl + P + Q` to detach from the container (still running background).

------------

## 2. Run Multiple Ray Clusters within the Same Host



```bash
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6379 --object-manager-port 6380 --node-manager-port 6381 --ray-client-server-port 10001 --min-worker-port 10002 --max-worker-port 11001 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats
```

```bash
ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 ray start --head --port=6389 --object-manager-port 6390 --node-manager-port 6391 --ray-client-server-port 11002 --min-worker-port 11003 --max-worker-port 12002 --num-cpus 60 --object-store-memory 10737418240 --disable-usage-stats
```

## x. (Deprecated) Execute Job Training Outside the Container

We can use the following command to execute the training script outside the Docker container:

```bash
docker exec crius-worker bash -c "cd /app/jax; ln -s /opt/conda/envs/alpa/bin/python /usr/bin/python; bash ./profile.sh -x 1_1080ti -n 1 -d 2 -m wide_resnet -p 500M -b 32 -l 16 -c 2 -o"
```

To execute a series of training operations of multiple jobs, we can formulate a bash script, which is dispatched from the central server to each node. A naive example is as follows:

```bash
# Pass 1.
docker exec crius-worker bash -c "cd /app/jax; ln -s /opt/conda/envs/alpa/bin/python /usr/bin/python; bash ./profile.sh -x 1_1080ti -n 1 -d 2 -m wide_resnet -p 500M -b 32 -l 16 -c 2 -o"

# Pass 2.
echo "[I][SHELL] Sleeping..."
sleep 30
echo "[I][SHELL] Sleep ends."

# Pass 3.
docker exec crius-worker bash -c "cd /app/jax; ln -s /opt/conda/envs/alpa/bin/python /usr/bin/python; bash ./profile.sh -x 1_1080ti -n 1 -d 2 -m wide_resnet -p 500M -b 64 -l 16 -c 2 -o"
```
