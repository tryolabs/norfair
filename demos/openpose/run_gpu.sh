#!/usr/bin/env bash
docker build . -t norfair-openpose
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-openpose \
           bash
