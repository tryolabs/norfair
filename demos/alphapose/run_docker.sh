#!/usr/bin/env bash
docker build . -f Dockerfile -t norfair-alphapose
nvidia-docker run --gpus all -it --shm-size=1gb --rm --net=host --runtime nvidia -v `realpath .`:/demo norfair-alphapose bash