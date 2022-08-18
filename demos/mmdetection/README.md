# MMdetection example

Simplest possible example of tracking using [MMDetection](https://github.com/open-mmlab/mmdetection).

## Instructions


1. Build and run the Docker container with:
    ```bash
        ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes returned by MMDetection.

![Norfair MMDetection demo](../../docs/traffic_mmdet.gif)