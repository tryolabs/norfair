# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt update && \
    apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /sahi
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/tryolabs/norfair.git

WORKDIR /demo/src/
