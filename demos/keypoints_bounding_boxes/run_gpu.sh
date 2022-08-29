#!/usr/bin/env -S bash -e
docker build . -t norfair-bbx-kp
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-bbx-kp \
           bash
