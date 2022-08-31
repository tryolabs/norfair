#!/usr/bin/env -S bash -e
docker build . -t norfair-yolov4
docker run -it --rm \
           --gpus all \
           -v `realpath .`:/demo \
           norfair-yolov4 \
           bash
