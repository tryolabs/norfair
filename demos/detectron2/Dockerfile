# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 gcc libglib2.0-0 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install Detectron2 building from source
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# For some reason it doesn't work with latest version of OpenCV
RUN pip install opencv-python==4.5.5.64

RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair

WORKDIR /demo/src/

# Download model weights
ADD https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl /model/model_final_f10217.pkl