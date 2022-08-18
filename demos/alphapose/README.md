# Tracking pedestrians with AlphaPose

An example of how to integrate Norfair into the video inference loop of a pre existing repository. This example uses Norfair to try out custom trackers on [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).

## Instructions


1. Build and run the Docker container with:
    ```bash
        ./run_docker.sh
    ``` 

4. In the container, display the demo instructions: 
    ```bash
        python3 video_demo.py --help 
    ``` 
    In the container, use the `/demo` folder as a volume to share files with the container.
    ```bash
        python3 video_demo.py --video /demo/video.mp4 --outdir /demo/ --save_video
    ``` 

## Explanation

With Norfair, you can try out your own custom tracker on the very accurate poses produced by AlphaPose by just integrating it into AlphaPose itself, and therefore avoiding the difficult job of decoupling the model from the code base.

This produces the following results:

![Norfair AlphaPose demo](../../docs/alphapose.gif)