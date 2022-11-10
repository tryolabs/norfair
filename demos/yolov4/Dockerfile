ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git /pytorch-YOLOv4
ENV PYTHONPATH "${PYTHONPATH}:/pytorch-YOLOv4"
WORKDIR /pytorch-YOLOv4

ADD https://github.com/tryolabs/norfair/releases/download/v0.4.0/yolov4_fixed_layer_names.pth /checkpoint/yolov4_fixed_layer_names.pth

WORKDIR /demo
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/tryolabs/norfair.git@master

WORKDIR /demo/src
