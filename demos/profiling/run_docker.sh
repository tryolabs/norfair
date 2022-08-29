#!/usr/bin/env bash
docker build . -t trt-video
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo trt-video 
