docker build . -f Dockerfile.local -t norfair-openpose
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-openpose bash
