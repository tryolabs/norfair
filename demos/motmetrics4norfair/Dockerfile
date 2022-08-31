# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download the MOT17 dataset
WORKDIR /
# To download Detections + Ground Truth (9.7 MB)
RUN curl -O https://motchallenge.net/data/MOT17Labels.zip && \
    unzip MOT17Labels.zip -d MOT17

WORKDIR /demo
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN git clone https://github.com/ifzhang/ByteTrack /ByteTrack
WORKDIR /ByteTrack
RUN pip install -r requirements.txt && \
    python setup.py develop && \
    pip install cython_bbox

WORKDIR /
RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair[metrics,video]

WORKDIR /demo/src/
