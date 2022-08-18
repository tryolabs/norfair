#!/usr/bin/env bash
docker build . -f Dockerfile.local -t norfair-bbx-kp
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-bbx-kp bash
