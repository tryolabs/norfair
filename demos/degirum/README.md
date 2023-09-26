# DeGirum Example

Example of tracking with DeGirum PySDK. Based on [YOLOv5 Example](https://github.com/tryolabs/norfair/tree/9b315b4cfa5f9cf145f068a21a2b7673703ac9e3/demos/yolov5).

## Instructions

1. Build and run the Docker container with `./run.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example can track objects using any detection model from DeGirum's cloud platform.

