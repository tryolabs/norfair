# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /
# Download the MOT17 dataset
RUN curl -O https://motchallenge.net/data/MOT17.zip && \
    unzip MOT17.zip
RUN rm -rf MOT17/test

# Download the MOT20 dataset
RUN curl -O https://motchallenge.net/data/MOT20.zip && \
    unzip MOT20.zip
RUN rm -rf MOT20/test

# Replace MOT Challenge's detections for ByteTrack's detections
RUN pip install gdown && \
    gdown --folder https://drive.google.com/drive/folders/1m8alcbgt3e3XAWCq-iOQknGFKqFwSwt0?usp=sharing
RUN cp -r bytetrack_detections/MOT17/train/ /MOT17/
RUN cp -r bytetrack_detections/MOT20/train/ /MOT20/

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
