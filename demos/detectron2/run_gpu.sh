#!/usr/bin/env -S bash -e
docker build . -t norfair-detectron
docker run -t --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-detectron \
           python demo.py /demo/traffic.mp4
