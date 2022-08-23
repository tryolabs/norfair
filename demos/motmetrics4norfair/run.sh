#!/usr/bin/env -S bash -e
docker build . -t norfair-motmetrics
docker run -t --rm \
           -v `realpath .`:/demo \
           norfair-motmetrics \
           python motmetrics4norfair.py /MOT17/train/
