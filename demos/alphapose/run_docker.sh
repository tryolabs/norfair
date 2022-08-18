#!/usr/bin/env bash
docker build . -f Dockerfile.local -t norfair-alphapose
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-alphapose bash
