# @ Info: Profiler imange.

# NOTE: There are some modifications should be made in the source code of pip-installed alpa:
#   - In some hosts with old Nvidia driver and without `nvidia-container-toolkit`, 
#     `nvidia-smi` might leads to error. In this case, after docker build, the user
#     needs to docker run the container and modify `self.has_cuda` to `True` rather 
#     than obtained from `nvidia-smi` in `/opt/conda/envs/alpa/lib/python3.8/
#     site-packages/alpa/global_env.py`.
#   - Also, since Alpa produces `profile-results-xxx.npy` as logs by default, which 
#     consumes memory in singularity overlay. We need to comment out the `np.save()` 
#     in line 1295 of `/opt/conda/envs/alpa/lib/python3.8/site-packages/alpa/pipeline_parallel/stage_profiling.py`.
#   - In `/opt/conda/envs/alpa/lib/python3.8/site-packages/alpa/device_mesh.py`, we
#     need to modify to `env_vars["NCCL_DEBUG"] = os.environ["NCCL_DEBUG"]` in line 1113
#     and add `if "NCCL_IB_HCA" in os.environ: env_vars["NCCL_IB_HCA"] = os.environ["NCCL_IB_HCA"]` in line 1111.
#   - In `/opt/conda/envs/alpa/lib/python3.8/site-packages/alpa/pipeline_parallel/stage_profiling.py`, we
#     add:
#       - In get_compute_cost() line 1234: 
#           all_pruned_stages_indices = list()
#       - In get_compute_cost() line 1249: 
#           # To skip legacy error in tensorflow xla
#           if sliced_virtual_meshes[0].num_hosts > 1 and sliced_virtual_meshes[0].num_hosts < len(ray.nodes()):
#               print(f"[TMP] Skip profiling of {sliced_virtual_meshes[0].num_hosts} due to legacy error in tensorflow...")
#               profile_results, pruned_stages_indices = crius_prune_mesh_num_devices(
#                      mesh_num_devices=sliced_virtual_meshes[0].num_devices,
#                      cluster_size=cluster_size, mesh_id=mesh_id, 
#                      num_all_layers=len(layers), layer_flops_prefix_sum=layer_flops_prefix_sum,
#                      autosharding_configs=autosharding_configs[mesh_id], 
#                      profile_results=profile_results, 
#                      imba_tolerance=auto_stage_option.stage_imbalance_tolerance)
#               all_pruned_stages_indices.extend(pruned_stages_indices)
#               # Skip compiling and profiling
#               continue
#       - In get_compute_cost() line 1260: 
#           Add one argument to generate_training_stages_2d(): all_pruned_stages_indices
#       - In generate_training_stages_2d() line 659: 
#           Add one argument at last: pruned_stages_indices: Sequence[Tuple[int]] = None
#       - In generate_training_stages_2d() line 680: 
#            stage_idx = (start, end, mesh_id, 0)
#            if stage_idx in pruned_stages_indices:
#                # Skip compiling and profiling for this stage
#                for _i, _cfg in enumerate(autosharding_configs):
#                    if _cfg is not None:
#                        _stage_idx = (start, end, mesh_id, _i)
#                        stages.append(
#                            (_stage_idx, None, _cfg))
#                continue
#       - Add new functions: crius_prune_mesh_num_devices() and _crius_forge_profile_results()
#       - Import Any and os.

# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# Metadata
LABEL maintainer="dicardo@sjtu.edu.cn"
LABEL version="1.0"
LABEL description="Image for building the profiler runtime container."

#########################################
#        Alpa Installation Begin        # 
#########################################
# init workdir
RUN mkdir -p /build
WORKDIR /build

# Envs
# ENV CUDA_PATH=/usr/local/cuda
# ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/compat:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive

# Install common tool & conda
RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install vim -y && \
    apt install bc && \
    apt-get install net-tools -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/alpa && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Install conda alpa env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name alpa python=3.8 -y && \
    conda activate alpa && \
    pip3 install cmake==3.24.1 && \
    apt install coinor-cbc -y && \
    pip3 install pandas && \
    pip3 install --upgrade pip && \
    pip3 install cupy-cuda11x && \
    pip3 install alpa && \
    pip3 install jaxlib==0.3.22+cuda113.cudnn820 -f https://alpa-projects.github.io/wheels.html && \
    pip3 install numpy==1.23.5 && \
    pip3 install flask && \
    pip3 install deprecated && \
    pip3 install cvxpy

# Install other dependencies
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate alpa && \
    pip3 install orjson && \
    pip3 install statsmodels && \
    pip3 install cplex && \
    pip3 install cvxopt

#########################################
#         Alpa Installation End         # 
#########################################

#########################################
#      Profiling Preparation Begin      # 
#########################################

# Envs
RUN mkdir -p /app
ENV WORK_DIR=/app

# Work dir
WORKDIR ${WORK_DIR}

# Copy files to work dir
COPY . ${WORK_DIR}
# Add all permission to all files.
RUN chmod -R 777 ${WORK_DIR}

# Expose port of the container (not public, use `docker run -p [HOST_PORT]:4160` to handle public).
EXPOSE 4160 6379

# # (Deprecated) Env to solve `NVIDIA-SMI couldn't find libnvidia-ml.so library in your system`.
# ENV LD_PRELOAD=/usr/local/cuda/targets/x86_64-linux/lib/stubs/libnvidia-ml.so

#########################################
#       Profiling Preparation End       # 
#########################################

# Enterpoint for bash shell
ENTRYPOINT ["/bin/bash"]
