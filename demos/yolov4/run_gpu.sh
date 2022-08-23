#!/usr/bin/env -S bash -e
docker build . -t norfair-yolov4
docker run -t --rm \
           --gpus all \
           -v `realpath .`:/demo \
           norfair-yolov4 \
           python src/demo.py /demo/traffic.mp4
