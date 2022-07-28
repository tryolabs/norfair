# MMdetection example

Simplest possible example of tracking using [MMDetection](https://github.com/open-mmlab/mmdetection).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. [Follow the instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation) to install mmdet using mim and cloning the repo inside `demo/mmdetection`.
3. Download the [checkpoint](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth) inside `demo/mmdetection/checkpoints`
3. Run `python mmdetection_cars.py`. For the demo, we are using [this traffic footage](https://www.youtube.com/watch?v=aio9g9_xVio).

## Explanation

This example tracks objects using a single point per detection: the centroid of the bounding boxes returned by MMDetection.

![Norfair MMDetection demo](../../docs/traffic_mmdet.gif)
