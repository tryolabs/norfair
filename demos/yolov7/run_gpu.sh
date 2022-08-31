#!/usr/bin/env -S bash -e
docker build . -t norfair-yolov7
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-yolov7 \
           bash
