#!/usr/bin/env -S bash -e
docker build . -t norfair-reid
docker run -it --rm \
           -v `realpath .`:/demo \
           norfair-reid \
           bash
