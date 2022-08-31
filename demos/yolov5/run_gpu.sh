#!/usr/bin/env -S bash -e
docker build . -t norfair-yolov5
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-yolov5 \
           bash
