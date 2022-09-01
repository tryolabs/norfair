# YOLOv5 Example

Simplest possible example of tracking. Based on [YOLOv5](https://github.com/ultralytics/yolov5).

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This example tracks objects using a single or double point per detection: the centroid or the two corners of the bounding boxes around objects returned by YOLOv5.

## Tracking cars

![Norfair tracking cars using YOLOv5](../../docs/yolov5_cars.gif)

## Tracking pedestrians

![Norfair tracking pedestrians using YOLOv5](../../docs/yolov5_pedestrian.gif)

## Tracking all

![Norfair tracking cars and pedestrians using YOLOv5](../../docs/yolov5_all.gif)
