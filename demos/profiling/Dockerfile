# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags
FROM nvcr.io/nvidia/tensorrt:20.12-py3

RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /demo
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# install torch2trt and trt-pose and norfair
WORKDIR /
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    git clone https://github.com/NVIDIA-AI-IOT/trt_pose

WORKDIR /torch2trt/
RUN python setup.py install --plugins

WORKDIR /trt_pose/
RUN python setup.py install && \
    gdown --id 13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU && \
    gdown --id 1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd

WORKDIR /
RUN pip install git+https://github.com/tryolabs/norfair.git@master

WORKDIR /demo/src/
