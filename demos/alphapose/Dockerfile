# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
FROM nvcr.io/nvidia/pytorch:22.08-py3

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && \
    apt-get install -y \
    sudo libyaml-dev git gcc build-essential wget unzip \
    libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev \
    locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install alphapose
# https://pythontechworld.com/issue/mvig-sjtu/alphapose/1067
# https://github.com/MVIG-SJTU/AlphaPose/issues/1072
# https://github.com/MVIG-SJTU/AlphaPose/issues/572
RUN git clone https://github.com/MVIG-SJTU/AlphaPose.git ./AlphaPose/ && \
    pip3 install cython && \
    ln -s /usr/bin/ninja /usr/bin/ninja-build && \
    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip && \
    sudo unzip ninja-linux.zip -d /usr/local/bin/ && \
    sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force && \
    git clone https://gitlab.freedesktop.org/mesa/glu ./glu/ && \
    pip3 install meson && \
    cd ./glu/ && mkdir build && cd build && meson --prefix=$XORG_PREFIX -Dgl_provider=gl --buildtype=release --prefix '/' .. && ninja && sudo ninja install && rm -vf /usr/lib/libGLU.a

COPY setup.py /root/AlphaPose/setup.py

# Our own flavor of DataWriter, which includes Norfair tracking
COPY writer.py /root/AlphaPose/alphapose/utils/writer.py

RUN cd /root/AlphaPose/ && python3 setup.py build develop --user

RUN pip3 install gdown==4.4.0 && \
    mkdir /root/AlphaPose/detector/yolo/data/ && \
    gdown --id 1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC -O /root/AlphaPose/detector/yolo/data/yolov3-spp.weights && \
    gdown --id 1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA -O /root/AlphaPose/pretrained_models/jde.1088x608.uncertainty.pt && \
    gdown --id 1kQhnMRURFiy7NsdS8EFL-8vtqEXOgECn -O /root/AlphaPose/pretrained_models/fast_res50_256x192.pth

# if you want to use yolox-x as the detector
# RUN wget -P /root/AlphaPose/detector/yolox/data/ https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_x.pth

RUN pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair

WORKDIR /root/AlphaPose/
