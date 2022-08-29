docker build . -f Dockerfile.local -t norfair-reid
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/reid norfair-reid bash
