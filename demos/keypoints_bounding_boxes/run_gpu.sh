#!/usr/bin/env -S bash -e
docker build . -t norfair-bbx-kp

docker run -it --rm \
           --gpus all \
           --shm-size=1gb \
           -v `realpath .`:/demo \
            --env DISPLAY=unix$DISPLAY \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
           norfair-bbx-kp \
           bash



           python3 demo.py /demo/traffic.mp4 /demo/out.mp4



           
./build/examples/openpose/openpose.bin --video examples/media/video.avi