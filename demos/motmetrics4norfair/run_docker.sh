docker build . -f Dockerfile.local -t norfair-motmetrics
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-motmetrics bash
