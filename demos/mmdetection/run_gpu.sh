#!/usr/bin/env bash
docker build . -t norfair-mmdetection
docker run -it --rm \
           --gpus all \
           -v `realpath .`:/demo \
           norfair-mmdetection \
           bash
