#!/usr/bin/env bash
docker build . -t norfair-openpose

# docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-openpose bash

docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-openpose \
           bash

           # python3 demo.py /demo/traffic.mp4

