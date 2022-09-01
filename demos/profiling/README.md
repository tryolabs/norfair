# Filter and distance function profiling

Measure inference speed for different filters and distance functions using a [TRT pose estimator](https://github.com/NVIDIA-AI-IOT/trt_pose) model.

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.
