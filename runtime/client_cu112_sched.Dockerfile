# @ Info: Image for building the profiler runtime container.
# FUNC: 
#   - Install common tools and conda toolkit.
#   - Install customized Alpa.
#   - Copy scripts into workdir.

# Base image
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# Metadata
LABEL maintainer="dicardo@sjtu.edu.cn"
LABEL version="1.0"
LABEL description="Image for building the crius client container."

#########################################
#        Alpa Installation Begin        # 
#########################################
# init workdir
RUN mkdir -p /build
WORKDIR /build

# Envs
# If leads to not found error of `/usr/bin/gcc-7` in this base image, apt install in the dockerfile
# ENV CC=/usr/bin/gcc-7
# ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc-7
ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/compat:/usr/lib/x86_64-linux-gnu:$CUDA_PATH/targets/x86_64-linux/lib/stubs/:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive
# Set proxy for go modules
# ENV GOPROXY="https://gocenter.io"
# ENV GO111MODULE=on

# install common tool & conda
RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install vim -y && \
    apt install gcc-7 g++-7 -y && \
    apt install bc && \
    apt-get install net-tools -y && \
    apt install ssh -y && \
    wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    mkdir -p /opt/conda/envs/alpa && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# install conda alpa env
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --name alpa python=3.8 -y && \
    conda activate alpa && \
    pip3 install cmake==3.24.1 && \
    apt install coinor-cbc -y && \
    pip3 install orjson && \
    pip3 install cvxpy && \
    pip3 install cplex && \
    pip3 install pandas && \
    pip3 install --upgrade pip && \
    # pip3 install cupy-cuda113 && \
    pip3 install alpa && \
    pip3 install cupy-cuda112 && \
    # pip3 install jaxlib==0.3.22+cuda113.cudnn820 -f https://alpa-projects.github.io/wheels.html && \
    pip3 install jaxlib==0.3.22+cuda112.cudnn810 -f https://alpa-projects.github.io/wheels.html && \
    pip3 install numpy==1.23.5 && \
    pip3 install flask && \
    pip3 install deprecated

# To solve `libcuda.so.1 cannot be found` error in Docker build jaxlib
# Ref: https://github.com/tensorflow/tensorflow/issues/10776#issuecomment-309128975
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH
# RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Install customized alpa from the source 
# Need to run `rm -rf ~/.cache/bazel` before each re-compilation
# RUN . /opt/conda/etc/profile.d/conda.sh && \
#     conda activate alpa && \
#     # git clone --recursive https://github.com/alpa-projects/alpa.git && \
#     # cd alpa && \
#     git clone --recursive https://github.com/DicardoX/custom_alpa.git && \
#     cd custom_alpa && \
#     pip3 install -e ".[dev]" && \
#     cd build_jaxlib && \
#     python3 build/build.py --enable_cuda --dev_install --bazel_options=--override_repository=org_tensorflow=$(pwd)/../third_party/tensorflow-alpa && \
#     cd dist && \
#     pip3 install -e .

# To solve `libcuda.so.1 cannot be found` error in Docker build jaxlib
# Ref: https://github.com/tensorflow/tensorflow/issues/10776#issuecomment-309128975
# RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

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
EXPOSE 4161 6379

#########################################
#       Profiling Preparation End       # 
#########################################

# Enterpoint for bash shell
ENTRYPOINT ["/bin/bash"]
