# Detectron2 example

Simplest possible example of tracking. Based on [Detectron2](https://github.com/facebookresearch/detectron2).

## Instructions


1. Build and run the docker container with:
    ```bash
        ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes returned by Detectron2.

![Norfair Detectron2 demo](../../docs/traffic.gif)