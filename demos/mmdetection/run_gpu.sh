#!/usr/bin/env bash
docker build . -t norfair-mmdetection
docker run -t --rm \
           --gpus all \
           -v `realpath .`:/demo \
           norfair-mmdetection \
           python src/demo.py /demo/traffic.mp4
