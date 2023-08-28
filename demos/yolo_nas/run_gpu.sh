#!/usr/bin/env -S bash -e
docker build . -t norfair-yolonas
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-yolonas \
           bash
