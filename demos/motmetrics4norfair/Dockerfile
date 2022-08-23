FROM python:3.8-slim-buster

RUN apt update && \
    apt install -y libgl1 sudo gcc libglib2.0-0 git curl unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download the MOT17 dataset
WORKDIR /
# To download Detections + Ground Truth (9.7 MB)
RUN curl -O https://motchallenge.net/data/MOT17Labels.zip && \
    unzip MOT17Labels.zip -d MOT17

WORKDIR /demo
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

WORKDIR /
RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair[metrics,video]

WORKDIR /demo/src/
