#!/usr/bin/env -S bash -e
docker build . -t norfair-trt-profiling
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-trt-profiling \
           bash
