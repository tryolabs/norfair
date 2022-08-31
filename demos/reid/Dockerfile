FROM python:3.8-slim-buster

RUN apt update && \
    apt install -y libgl1 libglib2.0-0 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /reid
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install git+https://github.com/tryolabs/norfair.git@master

WORKDIR /demo/src/
