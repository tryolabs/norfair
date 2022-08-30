# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update && \
    apt-get install -y libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install -r https://raw.githubusercontent.com/WongKinYiu/yolov7/main/requirements.txt && \
    # For some reason it doesn't work with latest version of OpenCV \
    pip install opencv-python==4.5.5.64

RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair

WORKDIR /demo/src/
