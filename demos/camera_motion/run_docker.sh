
#!/usr/bin/env bash
docker build . -f Dockerfile.local -t norfair-camera-motion
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-camera-motion bash