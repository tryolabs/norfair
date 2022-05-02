# Filter setups and distances profiling

Measure elapsed time while updating the tracker and using a [TRT pose estimator](https://github.com/NVIDIA-AI-IOT/trt_pose) model, for different filter setups and distance functions.

## Instructions

1. Build and run the docker container with:
    ```bash
        cd profiling_container && ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python demo.py --help 
    ``` 


