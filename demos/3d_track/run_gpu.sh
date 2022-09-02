#!/usr/bin/env -S bash -e
docker build . -t norfair-3d
docker run -it --rm \
           --gpus all \
           -v `realpath .`:/demo \
           norfair-3d \
           bash
