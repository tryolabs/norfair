#!/usr/bin/env -S bash -e
docker build . -t norfair-alphapose
docker run -it --rm \
           --gpus all \
           --shm-size=5gb \
           -v `realpath .`:/demo \
           norfair-alphapose \
           bash
