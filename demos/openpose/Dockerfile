#
# Based on https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1786#issuecomment-1180161510
# but changing base image.
#
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# https://forums.developer.nvidia.com/t/issues-building-docker-image-from-ngc-container-nvcr-io-nvidia-pytorch-22-py3/209034/5
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

# Install dependencies for OpenPose and its installation
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --yes && apt-get install --yes --auto-remove \
    libopencv-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libboost-all-dev \
    libhdf5-dev \
    libatlas-base-dev \
    git \
    cmake \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Pip dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
    numpy \
    opencv-python

# Clone the OpenPose repository and compile it
WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git ./ && \
    git submodule update --init --recursive --remote
WORKDIR  /openpose/build/
RUN cmake \
    -D BUILD_PYTHON=ON \
    -D BUILD_CAFFE=ON \
    -D BUILD_EXAMPLES=ON \
    -D GPU_MODE=CUDA \
    -D USE_CUDNN=OFF \
    -D CMAKE_BUILD_TYPE=Release \
    -D DOWNLOAD_BODY_COCO_MODEL=ON \
    -D DOWNLOAD_BODY_MPI_MODEL=ON \
    -D DOWNLOAD_HAND_MODEL=OFF \
    -D DOWNLOAD_FACE_MODEL=OFF \
    ..
RUN make --jobs `nproc`

RUN pip install git+https://github.com/tryolabs/norfair.git@master

WORKDIR /demo/src
