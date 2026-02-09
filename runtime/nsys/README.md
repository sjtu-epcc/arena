# Documentation of Exploiting Nsight System Toolkit

- Ref: [Nsys Doc](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)


-------------------

## 1. Common commands

### 1.0 Pre-check

To check the available CUDA devices, use:

```bash
nsys profile --gpu-metrics-device=help
```

### 1.1 Profile within Non-interactive Mode

We use the following commands to profile and store the result of GPU metrics in non-interactive mode (fixed command):

```bash
# Global path of Nsys workspace
export NSYS_PATH=/home/cyxue/Projects/playground/nsys/nsys_toolkit
# Profiling command warpped with Nsys profiling, only need to profile CUDA related APIs
# To restrict the time cost of database generation, we limit the sampling frequency of GPU to 100/s.
nsys profile -o ${NSYS_PATH}/output.qdrep --gpu-metrics-device all --gpu-metrics-frequency 100 -t cuda,cudnn,cublas -f true ${CMD}
# Note: When cross-nodes profiling stucks with NCCL related error or without any error, check whether the ENV `NCCL_SOCKET_IFNAME` is properly set.
```

> An optional visualizing method is to download `output.qdrep` and open with Nvidia Nsight System Client.

To parse the profiling result, we need to generate a temporary SQLite database and extract needed metrics from it:

```bash
# Export the profiling result as SQLite database
nsys export -t sqlite -o ${DB_PATH} -f true ${NSYS_REP_PATH}
# Parse result in database-querying style
export METRIC="PCIe RX Throughput"
sqlite3 ${DB_PATH} "SELECT rawTimestamp, CAST(JSON_EXTRACT(data, '$.\"${METRIC}\"') as INTEGER) as value FROM GENERIC_EVENTS WHERE value != 0 LIMIT 10"
```

<!-- ### 1.2 Profile within Interactive Mode

To profile GPU metrics for other worker hosts, we use the following command to start & stop Nsys profiling in interactive mode (when launching service):

```bash
# Global path of Nsys workspace
export NSYS_PATH=/home/cyxue/Projects/playground/nsys/nsys_toolkit
# Launch session of Nsys profiling, only need to profile CUDA related APIs
# To restrict the time cost of database generation, we limit the sampling frequency of GPU to 100/s.
nsys launch -t cuda,cudnn,cublas --show-output true --session nsys-prof ${SERV_LAUNCH_CMD}
# Start profiling the service
nsys start -o ${NSYS_PATH}/output_2.qdrep --gpu-metrics-device all --gpu-metrics-frequency 100 -f true --session nsys-prof
# Stop profiling
nsys stop --session nsys-prof -->
```

-------------------