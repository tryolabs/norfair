#!/usr/bin/env -S bash -e
docker build . -t norfair-motmetrics
docker run -it --rm \
           -v `realpath .`:/demo \
           norfair-motmetrics \
           bash
