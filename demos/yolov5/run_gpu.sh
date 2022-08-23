#!/usr/bin/env -S bash -e
docker build . -t norfair-yolov5
docker run -t --rm \
           --gpus all \
           -v `realpath .`:/demo \
           norfair-yolov5 \
           python demo.py /demo/traffic.mp4
