#!/usr/bin/env -S bash -e
docker build . -t norfair-degirum
docker run -it --rm \
           -v `realpath .`:/demo \
           norfair-degirum \
           bash
