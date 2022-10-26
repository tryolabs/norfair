# YOLOPv2 Example

Tracking objects with Norfair based on the [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) model.

YOLOPv2 is a multi-task learning network, this model does Panoptic Driving Perception to simultaneously perform the task of traffic object detection, drivable road area segmentation, and lane line detection.

This model was trained over [BDD100K](https://doc.bdd100k.com/index.html) dataset using [these](https://doc.bdd100k.com/format.html) labels for each task.

## Instructions

1. Build and run the Docker container with `./run_gpu.sh`.
2. Copy a video to the `src` folder.
3. Within the container, run with the default parameters:

   ```bash
   python demo.py <video>.mp4
   ```

For additional settings, you may display the instructions using `python demo.py --help`.

## Explanation

This demo uses YOLOPv2 capabilities for object detection, drivable area segmentation, and lane line detection. Norfair is used to add tracking capabilities for the detected objects.

https://user-images.githubusercontent.com/67343574/195704838-eee83fd3-652b-4b27-a670-6e7929d64c00.mp4
