FROM nvcr.io/nvidia/tensorrt:20.12-py3

RUN apt update && \
apt install -y libgl1 && \
apt-get update && \
apt-get -y install sudo

WORKDIR /demo
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
pip install -r requirements.txt

# install torch2trt and trt-pose and norfair
WORKDIR /
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
git clone https://github.com/NVIDIA-AI-IOT/trt_pose

WORKDIR /torch2trt/
RUN sudo python3 setup.py install --plugins

WORKDIR /trt_pose/
RUN sudo python3 setup.py install && \
gdown --id 13FkJkx7evQ1WwP54UmdiDXWyFMY1OxDU && \
gdown --id 1XYDdCUdiF2xxx4rznmLb62SdOUZuoNbd

WORKDIR /
RUN git clone https://github.com/tryolabs/norfair.git ./norfair/ && \
pip3 install ./norfair/

WORKDIR /demo/src/