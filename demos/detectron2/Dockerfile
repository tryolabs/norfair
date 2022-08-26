FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y libgl1 sudo gcc libglib2.0-0 git && \
    pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# Download model weights
ADD https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl /model/model_final_f10217.pkl

WORKDIR /demo
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair

WORKDIR /demo/src/
