#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Chunyu Xue

"""
The worker runtime, implemented as a daemon process of the Docker container.
-----------------------------------------------------
Functions: Run a flask server in the container to: 
    - Listen to the specified port of the container.
    - When receiving the request of executing a specified training job, 
      create a new process to execute the training job.
      specified by the scheduler, recording the PID of this process.
    - When receiving the request of suspending a specified training job, read 
      the iteration num recorded in a tmp file by the training job and return 
      the executed iteration num to the scheduler.
    - When the training job ends, return a ending flag of the job to the scheduler.
Refs:
    - Subprocess: https://www.qiniu.com/qfans/qnso-60482639
                  https://www.runoob.com/w3cnote/python3-subprocess.html
                  https://cloud.tencent.com/developer/article/1445388
                  (kill process) https://blog.tonyseek.com/post/kill-the-descendants-of-subprocess/
    - Flask: https://www.jianshu.com/p/4a83aca1ec9f
             https://www.w3cschool.cn/article/47726613.html
"""

import os
import argparse
import json
import traceback
# import threading
import subprocess
from time import sleep
from flask import Flask, request, jsonify
from typing import Any

# Args 
parser = argparse.ArgumentParser()
# parser.add_argument("--devices_name", default="default_device", type=str)
args = parser.parse_args()

# Instantiate flask
app = Flask(__name__)

# Max timeout threshold
MAX_TIMEOUT = 20


class WorkerRuntime:
    """ 
    The class of worker runtime running in the Docker container as the daemon process. 
    """
    def __init__(self):
        self.process_pool = dict()
        # Passed time from the last check
        self.checked_timer = 0
    
    def register(self, job_id: str, process: Any):
        """ Register the job uuid and the process of the training process. """
        self.process_pool[job_id] = process
    
    def query(self, job_id: str):
        """ Query the process of the target job uuid. """
        assert job_id in self.process_pool, \
            "Job id not found in the local process pool."
        return self.process_pool[job_id]

    def clear(self, job_id: str):
        """ Clear the record of the target job uuid. """
        _ = self.process_pool.pop(job_id)
    
    # def run(self):
    #     """ Update the checked timer and kill all training processes if the timer exceeds the timeout threshold. """
    #     while True:
    #         sleep(1)
    #         self.checked_timer += 1
    #         if self.checked_timer > MAX_TIMEOUT:
    #             # Kill all training processes
    #             for _job_id in self.process_pool.keys():
    #                 print("[I][RT] The periodical check from Crius runtime has been timeout, kill all training process...")
    #                 # Kill the training process and related sub-processes
    #                 os.killpg(os.getpgid(self.process_pool[_job_id].pid), 9)
    #                 # Clear record
    #                 worker_runtime.clear(job_id=job_id)

# Instantiate worker runtime
worker_runtime = WorkerRuntime()

# Current absolute path of this script
CUR_PATH = os.path.dirname(os.path.abspath(__file__))


@app.route("/query/check", methods=['POST'])
def check():
    """
    Periodically called to get json data and check whether the underlying training jobs are ended. 
    """
    # Return msg
    return_msg = {
        "executed_iter_num": 0,
        "avg_iter_time": 0.0,
        "last_iter_time": 0.0,
        "compilation_time": 0.0,
        "is_ended": False,
        "cmd": None,
        "job_id": None,
        "pid": None,
        "debug_msg": None
    }
    # Get data
    _data = request.get_data()
    # Trasform from byte to json
    json_data = json.loads(_data)
    if json_data is None:
        return_msg["debug_msg"] = "Input json data is None."
        return jsonify(return_msg)
    
    # Parse to get the training configurations
    job_id = json_data["job_id"]
    try_idx = json_data["try_idx"]
    # Training process
    process = worker_runtime.process_pool[job_id]
    return_msg["job_id"] = job_id
    return_msg["pid"] = process.pid
    
    # Get the executed iter num
    iter_cnt_file = f"{job_id}_{str(try_idx)}.txt"
    file_path = os.path.join(f"{CUR_PATH}/jax/tmp_res", iter_cnt_file)
    if os.path.exists(file_path):
        # Record executed iter num
        with open(file_path, 'r') as f:
            lines = f.readlines()
            _avg_iter_time = 0.0
            _iter_num = len(lines)
            for line in lines:
                _avg_iter_time += float(line)
            _avg_iter_time = round(_avg_iter_time / _iter_num, 3)
            return_msg["executed_iter_num"] = _iter_num
            return_msg["last_iter_time"] = round(float(lines[-1]), 3)
            return_msg["avg_iter_time"] = _avg_iter_time
    else:
        # File not found
        return_msg["debug_msg"] = f"The target training job is running, but " + \
                                  f"iter num tmp file with path `{file_path}` " + \
                                  f"is not found." if not process.poll() \
                                    else f"Job process is ended and iter num " + \
                                         f"tmp file `{file_path}` is not found."
        return_msg["is_ended"] = True if process.poll() is not None else False

        return jsonify(return_msg)
        
    
    # Get the compilation time
    comp_time_file = f"compile_time_{job_id}_{str(try_idx)}.txt"
    file_path = os.path.join(f"{CUR_PATH}/jax/tmp_res", comp_time_file)
    if os.path.exists(file_path):
        # Record compilation time
        with open(file_path, 'r') as f:
            line = f.readline()
            return_msg["compilation_time"] = round(float(line), 3)
    else:
        # File not found
        return_msg["debug_msg"] = f"The target training job is running, but " + \
                                  f"compile time tmp file with path `{file_path}` " + \
                                  f"is not found." if not process.poll() \
                                    else f"Job process is ended and compile num " + \
                                         f"tmp file `{file_path}` is not found."
        return_msg["is_ended"] = True if process.poll() is not None else False

        return jsonify(return_msg)

    if process.poll() is None:
        # Not ended
        return_msg["debug_msg"] = "The target training job is running."
        return jsonify(return_msg)
    
    # Ended
    return_msg["is_ended"] = True
    # Clear record in worker client
    worker_runtime.clear(job_id)
    
    if not os.path.exists(file_path):
        return_msg["executed_iter_num"] = 0
        return_msg["avg_iter_time"] = 0.0
        return_msg["last_iter_time"] = 0.0
        return_msg["compilation_time"] = 0.0
        return_msg["debug_msg"] = f"The iter cntr and compile time file is " + \
                                  f"not created (process is killed before entering the iterating)."
        return jsonify(return_msg)
    else:
        return_msg["debug_msg"] = "The target training job is ended."

    return jsonify(return_msg)


@app.route("/query/suspend", methods=['POST'])
def suspend():
    """ Get json data, suspend the target training job and return its executed iteration num. """
    # Return msg
    return_msg = {
        "executed_iter_num": 0,
        "avg_iter_time": 0.0,
        "compilation_time": 0.0,
        "cmd": None,
        "job_id": None,
        "pid": None,
        "debug_msg": None
    }
    # Get data
    _data = request.get_data()
    # Trasform from byte to json
    json_data = json.loads(_data)
    if json_data is None:
        return_msg["debug_msg"] = "Input json data is None."
        return jsonify(return_msg)
    
    # Parse to get the training configurations
    job_id = json_data["job_id"]
    try_idx = json_data["try_idx"]
    is_head = bool(json_data["is_head"])

    # The subprocess in the worker nodes will be automatically deleted when the training process
    # on the head node is deleted.
    if job_id in worker_runtime.process_pool:
        # Get the training process
        process = worker_runtime.process_pool[job_id]
        return_msg["job_id"] = job_id
        return_msg["pid"] = process.pid
        
        # Kill the training process and related sub-processes
        # process.kill()
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except Exception as e:
            print(f"[I][RT] Fail to kill training process, returned: {e}")
            traceback.print_exc()
            return_msg["debug_msg"] = "Fail to kill training process, probably the target process is already killed due to runtime failure."
            return jsonify(return_msg)

        # Clear record in worker client
        worker_runtime.clear(job_id=job_id)

        # Block until the process is killed
        is_killed = False
        while not is_killed:
            sleep(0.1)
            is_killed = (process.poll() is not None)
    
    if is_head:
        # Get the compilation time
        comp_time_file = f"compile_time_{job_id}_{str(try_idx)}.txt"
        file_path = os.path.join(f"{CUR_PATH}/jax/tmp_res", comp_time_file)
        if os.path.exists(file_path):
            # Record compilation time
            with open(file_path, 'r') as f:
                line = f.readline()
                return_msg["compilation_time"] = round(float(line), 3)
        
        # Get the executed iter num
        iter_cnt_file = f"{job_id}_{str(try_idx)}.txt"
        file_path = os.path.join(f"{CUR_PATH}/jax/tmp_res", iter_cnt_file)
        if not os.path.exists(file_path):
            return_msg["executed_iter_num"] = 0
            return_msg["avg_iter_time"] = 0.0
            return_msg["debug_msg"] = f"The iter cntr and compile time file is " + \
                                      f"not created (process is killed before entering the iterating)."
            return jsonify(return_msg)
        # Record executed iter num
        with open(file_path, 'r') as f:
            lines = f.readlines()
            _avg_iter_time = 0.0
            _iter_num = len(lines)
            for line in lines:
                _avg_iter_time += float(line)
            _avg_iter_time = round(_avg_iter_time / _iter_num, 3)
            return_msg["executed_iter_num"] = _iter_num
            return_msg["avg_iter_time"] = _avg_iter_time

    # Add msg
    return_msg["debug_msg"] = "The training process has been suspended."

    # Sleep for 1 sec to wait for the suspend of all sub-processes
    sleep(1)
    
    return jsonify(return_msg)


@app.route("/query/train", methods=['POST'])
def train():
    """ Get json data and create a new process to execute the training job. """
    # Return msg
    return_msg = {
        "cmd": None,
        "worker_pid_table": None,
        "debug_msg": None
    }
    # Get data
    _data = request.get_data()
    # Trasform from byte to json
    json_data = json.loads(_data)
    if json_data is None:
        return_msg["debug_msg"] = "Input json data is None."
        return jsonify(return_msg)
    
    # Parse to get the training configurations
    job_id = json_data["job_id"]
    try_idx = json_data["try_idx"]
    is_head = bool(json_data["is_head"])
    # Ip addr and daemon port
    overwrite_net_if = json_data["overwrite_network_interface"]
    head_node_ip_addr = json_data["head_node_ip_addr"]
    head_node_daemon_port = json_data["head_node_daemon_port"]
    head_node_port = json_data["head_node_port"]
    worker_node_ip_addr = json_data["worker_node_ip_addr"]
    worker_node_daemon_port = json_data["worker_node_daemon_port"]
    # Device info
    devices_name = json_data["devices_name"]
    num_devices_per_node = json_data["num_devices_per_node"]
    node_num = json_data["node_num"]
    # Job info
    model_name = json_data["model_name"]
    param_num = json_data["param_num"]
    batch_size = json_data["batch_size"]
    iter_num = json_data["iter_num"]
    prune_prompt = json_data["prune_prompt"]
    # Visible GPU rank
    gpu_rank_list = list(json_data["gpu_rank_list"])
    gpu_visible_str = str(gpu_rank_list[0])
    for _idx in range(1, len(gpu_rank_list), 1):
        gpu_visible_str = gpu_visible_str + f",{gpu_rank_list[_idx]}"
    # Port info of the ray cluster
    port = json_data["port"]
    object_manager_port = json_data["object_manager_port"]
    node_manager_port = json_data["node_manager_port"]
    ray_client_server_port = json_data["ray_client_server_port"]
    min_worker_port = json_data["min_worker_port"]
    max_worker_port = json_data["max_worker_port"]

    # Remove '-r' if exists in devices_name
    if '-' in devices_name:
        devices_name = devices_name.split('-')[0]
    
    # Make sure the target tmp iter cnt file is new generated 
    # in the training process.
    iter_cnt_file = f"{job_id}_{str(try_idx)}.txt"
    file_path = os.path.join(f"{CUR_PATH}/jax/tmp_res", iter_cnt_file)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Compilation time file
    comp_time_file = f"compile_time_{job_id}_{str(try_idx)}.txt"
    file_path = os.path.join(f"{CUR_PATH}/jax/tmp_res", comp_time_file)
    # Remove if existed
    if os.path.exists(file_path):
        os.remove(file_path)

    # Ray start cmd
    if is_head:
        ray_cmd = f"/bin/bash -c 'ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 " + \
                  f"ray start --head --node-ip-address {head_node_ip_addr} " + \
                  f"--port {port} --object-manager-port {object_manager_port} " + \
                  f"--node-manager-port {node_manager_port} " + \
                  f"--ray-client-server-port {ray_client_server_port} " + \
                  f"--min-worker-port {min_worker_port} " + \
                  f"--max-worker-port {max_worker_port} " + \
                  f"--num-cpus 8 --num-gpus {num_devices_per_node} " + \
                  f"--object-store-memory 10737418240 --disable-usage-stats'; sleep 15s" 
    else:
        # Extra sleep to wait for the establishment of the head node
        ray_cmd = f"sleep 3s; /bin/bash -c 'ulimit -c unlimited -n 65536 && RAY_DISABLE_MEMORY_MONITOR=1 " + \
                  f"ray start --address={head_node_ip_addr}:{head_node_port} " + \
                  f"--node-ip-address {worker_node_ip_addr} " + \
                  f"--object-manager-port {object_manager_port} " + \
                  f"--node-manager-port {node_manager_port} " + \
                  f" --ray-client-server-port {ray_client_server_port} " + \
                  f"--min-worker-port {min_worker_port} --max-worker-port {max_worker_port} " + \
                  f"--num-cpus 8 --num-gpus {num_devices_per_node} " + \
                  f"--object-store-memory 10737418240 --disable-usage-stats'; sleep 5s"

    net_if_cmd = "NET_IF=$(/bin/bash -c 'route' | grep default | grep -o 'eno.'); export NCCL_SOCKET_IFNAME=${NET_IF}" \
                    if overwrite_net_if == "none" else f"export NCCL_SOCKET_IFNAME={overwrite_net_if}"

    # Formulate prepare command of job training
    cmd_list = [
        # TODO(chunyu): Check whether executed in container environment.
        # ". /opt/conda/etc/profile.d/conda.sh && conda activate /opt/conda/envs/alpa", 
        "conda activate /opt/conda/envs/alpa", 
        # "cd /app",
        "cd /home/cyxue/Projects/crius/Crius/",
        net_if_cmd,
        "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8",
        f"export CUDA_VISIBLE_DEVICES={gpu_visible_str}",
        "export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1",
        ray_cmd,
        "sleep 0.5s", 
        "ray status",
    ]
    pre_cmd = f"{cmd_list[0]}; {cmd_list[1]}; {cmd_list[2]}; {cmd_list[3]}; {cmd_list[4]}; " + \
              f"{cmd_list[5]}; {cmd_list[6]}; {cmd_list[7]}; {cmd_list[8]}"

    # Formulate process clear command after job training, clear all sub-processes 
    # under this training process, therefore clear the target Ray cluster without 
    # affecting other ray clusters.
    clr_cmd = "python -c 'import os; os.killpg(os.getpgid(os.getpid()), 9)'; sleep 5s"

    # Ip addr of the ray cluster called by `ray.init()`
    ray_head_init_addr = f"{head_node_ip_addr}:{port}"
    
    # Prune cmd
    prune_cmd = f"--prune_search_space --prune_prompt {prune_prompt}" \
                    if prune_prompt != "none" else ""
    
    # Profile cmd
    prof_cmd = f"python {CUR_PATH}/../jaxpr/runtime_profiler.py --optimize_with_alpa " + \
               f"--overwrite_data --overwrite_coarsened_layer_num none " + \
               f"--rt_job_id {job_id} --try_idx {try_idx} " + \
               f"--devices_name {devices_name} " + \
               f"--num_devices_per_host {num_devices_per_node} " + \
               f"--num_hosts {node_num} " + \
               f"--ray_address {ray_head_init_addr} " + \
               f"--model_name {model_name} " + \
               f"--param_num {param_num} " + \
               f"--batch_size {batch_size} " + \
               f"--num_micro_batches 16 " + \
               f"--num_pipeline_layers 16 " + \
               f"--niter {iter_num} " + \
               f"--warmup_num 0 " + \
               f"{prune_cmd}"
    
    exec_cmd = f"{pre_cmd}; {prof_cmd}; {clr_cmd}" if is_head else f"{pre_cmd}"
    
    # Add msg
    return_msg["cmd"] = exec_cmd

    # Create a new process to execute the training job
    process = subprocess.Popen(exec_cmd, shell=True, preexec_fn=os.setsid)
    # Register job id and the pid
    worker_runtime.register(job_id, process)
    # Add msg
    return_msg["pid"] = str(process.pid)
    return_msg["debug_msg"] = f"The training process has been started, please periodically " + \
                              f"check whether it is ended until you want to suspend it."

    return jsonify(return_msg)


if __name__ == '__main__':
    # "host='0.0.0.0'" allows the server to be visied in the public network 
    # "port=4160" identifies the exposed port
    app.run(host="0.0.0.0", port=4160, debug=True)

    # # Run the worker runtime
    # worker_runtime.run()
