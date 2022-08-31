ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

#
# MMDetection snippet start
#
# Based on https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile
# but with slight modifications around versions to install.
#
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV MMDET_MMCV_VERSION="1.6.1"
ENV MMDET_CUDA="cu113"
ENV MMDET_PYTORCH="1.11.0"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==${MMDET_MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/${MMDET_CUDA}/torch${MMDET_PYTORCH}/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .

#
# MMDetection snippet end
#
WORKDIR /mmdetection
ADD https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

WORKDIR /demo
RUN pip install git+https://github.com/tryolabs/norfair.git@master

WORKDIR /demo/src
