#!/usr/bin/env -S bash -e
docker build . -t norfair-multicamera
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-multicamera \
           bash
