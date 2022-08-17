docker build . -f Dockerfile.local -t trt-video
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo trt-video 
