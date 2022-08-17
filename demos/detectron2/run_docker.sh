docker build . -f Dockerfile.local -t norfair-detectron
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-detectron bash
