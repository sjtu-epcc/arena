# @ Info: Customized Alpa image as the worker engine for Crius runtime.
# FUNC: 
#   - Install common tools and conda toolkit.
#   - Install Alpa using pip toolkits.
#   - Copy profiling scripts into workdir.

# Base image
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

# Metadata
LABEL maintainer="dicardo@sjtu.edu.cn"
LABEL version="1.0"
LABEL description="Customized Alpa image as the worker engine for Crius runtime."

#########################################
#        Alpa Installation Begin        # 
#########################################
# init workdir
RUN mkdir -p /build
WORKDIR /build

# install common tool & conda
RUN apt update && \
    apt install wget -y && \
    apt install git -y && \
    apt install vim -y && \
    apt-get install net-tools -y && \
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
    apt install coinor-cbc -y && \
    pip3 install pandas && \
    pip3 install --upgrade pip && \
    pip3 install cupy-cuda113 && \
    pip3 install alpa && \
    pip3 install jaxlib==0.3.22+cuda113.cudnn820 -f https://alpa-projects.github.io/wheels.html && \
    pip3 install numpy==1.23.5 && \
    pip3 install flask


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

#########################################
#       Profiling Preparation End       # 
#########################################

# Enterpoint for bash shell
ENTRYPOINT ["/bin/bash"]
