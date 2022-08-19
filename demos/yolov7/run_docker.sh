#!/usr/bin/env bash
docker build . -f Dockerfile.local -t norfair-yolov7
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-yolov7 bash
