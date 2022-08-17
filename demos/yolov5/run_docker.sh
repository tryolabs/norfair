docker build . -f Dockerfile.local -t norfair-yolov5
docker run --gpus all -it --shm-size=1gb --rm -v `realpath .`:/demo norfair-yolov5 bash
