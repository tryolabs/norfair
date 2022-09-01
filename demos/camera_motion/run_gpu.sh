#!/usr/bin/env -S bash -e
docker build . -t norfair-camera-motion
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-camera-motion \
           bash
