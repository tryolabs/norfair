docker build . -f Dockerfile.local -t norfair-something
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/generic_container norfair-something bash
