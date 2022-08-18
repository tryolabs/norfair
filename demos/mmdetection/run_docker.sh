#!/usr/bin/env bash
docker build . -f Dockerfile.local -t norfair-mmdetection
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-mmdetection bash
