# Tracking pedestrians with AlphaPose

An example of how to integrate Norfair into the video inference loop of a pre-existing solution. This example uses Norfair to try out custom trackers on [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), which has a non-trivial inference loop.

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.

2. In the container, display the demo instructions:

   ```bash
    python3 scripts/demo_inference.py --help
   ```

   In the container, use the `/demo` folder as a volume to share files with the container.

   ```bash
   python3 scripts/demo_inference.py --detector yolo --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video /demo/video.mp4 --save_video --outdir /demo/
   ```

## Explanation

With Norfair, you can try out your own custom tracker on the very accurate poses produced by AlphaPose by just integrating it into AlphaPose itself, and therefore avoiding the difficult job of decoupling the model from the code base.

This example modifies AlphaPose's original `writer.py` file, integrating a few lines that add Norfair tracking over the existing codebase.

This produces the following results:

https://user-images.githubusercontent.com/3588715/189143865-c22630d4-d60a-422b-b21e-9b8637e83ec2.mp4
