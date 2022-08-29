#!/usr/bin/env -S bash -e
docker build . -t norfair-detectron
docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
           norfair-detectron \
           bash
