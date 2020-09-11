![Norfair by Tryolabs logo](docs/logo.png)

![Build status](https://github.com/tryolabs/norfair/workflows/CI/badge.svg?branch=master)

Norfair is a customizable lightweight Python library for real-time 2D object tracking.

Using Norfair, you can add tracking capabilities to any detector with just a few lines of code.

## Features

- Any detector expressing its detections as a series of `(x, y)` coordinates can be used with Norfair. This includes detectors performing object detection, pose estimation, and instance segmentation.

- The function used to calculate the distance between tracked objects and detections is defined by the user, making the tracker extremely customizable. This function can make use of any extra information, such as appearance embeddings, which can heavily improve tracking performance.

- Modular. It can easily be inserted into complex video processing pipelines to add tracking to existing projects. At the same time it is possible to build a video inference loop from scratch using just Norfair and a detector.

- Fast. The only thing bounding inference speed will be the detection network feeding detections to Norfair.

Norfair is built, used and maintained by [Tryolabs](https://tryolabs.com).

## Installation

Norfair currently supports Python 3.6+.

```bash
pip install norfair
```

## How it works

Norfair works by estimating the future position of each point based on its past positions. It then tries to match these estimated positions with newly detected points provided by the detector. For this matching to occur, Norfair can rely on any distance function specified by the user of the library. Therefore, each object tracker can be made as simple or as complex as needed.

The following is an example of a particularly simple distance function calculating the Euclidean distance between tracked objects and detections. This is possibly the simplest distance function you could use in Norfair, as it uses just one single point per detection/object.

```python
 def euclidean_distance(detection, tracked_object):
     return np.linalg.norm(detection.points - tracked_object.estimate)
```

As an example we use [Detectron2](https://github.com/facebookresearch/detectron2) to get the single point detections to use with this distance function. We just use the centroids of the bounding boxes it produces around cars as our detections, and get the following results.

![Tracking cars with Norfair](docs/traffic.gif)

On the left you can see the points we get from Detectron2, and on the right how Norfair tracks them assigning a unique identifier through time. Even a straightforward distance function like this one can work when the tracking needed is simple.

Norfair also provides several useful tools for creating a video inference loop. Here is what the full code for creating the previous example looks like, including the code needed to set up Detectron2:

```python
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from norfair import Detection, Tracker, Video, draw_tracked_objects

# Set up Detectron2 object detector
cfg = get_cfg()
cfg.merge_from_file("demos/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
detector = DefaultPredictor(cfg)

# Norfair
video = Video(input_path="video.mp4")
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=20)

for frame in video:
    detections = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = [Detection(p) for p in detections['instances'].pred_boxes.get_centers().cpu().numpy()]
    tracked_objects = tracker.update(detections=detections)
    draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
```

The video and drawing tools use OpenCV frames, so they are compatible with most Python video code available online. The point tracking is based on [SORT](https://arxiv.org/pdf/1602.00763.pdf) generalized to detections consisting of a dynamically changing amount of points per detection.

## Motivation

Trying out the latest state of the art detectors normally requires running repositories which weren't intended to be easy to use. These tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are therefore intended to be run as a one-off evaluation script to get some result metric to publish on a particular research paper. This explains why they tend to not be easy to run as inference scripts, or why extracting the core model to use in another standalone script isn't always trivial.

Norfair was born out of the need to quickly add a simple layer of tracking over a wide range of newly released SOTA detectors. It was designed to seamlessly be plugged into a complex, highly coupled code base, with minimum effort. Norfair provides a series of modular but compatible tools, which you can pick and chose to use in your project.

## Documentation

You can find the documentation for Norfair's API [here](docs/README.md).

## Examples

We provide several examples of how Norfair can be used to add tracking capabilities to several different detectors.

1. [Simple tracking of cars](demos/detectron2/README.md) using [Detectron2](https://github.com/facebookresearch/detectron2).
2. [Simple tracking of cars](demos/yolov4/README.md) using [YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4/tree/master).
3. [Simple tracking pedestrians](demos/alphapose/README.md) using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).
4. [Speed up pose estimation by extrapolating detections](demos/openpose/README.md) using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

![Norfair OpenPose Demo](docs/openpose_skip_3_frames.gif)

## Commercial support

Tryolabs can provide commercial support, implement new features in Norfair or build video analytics tools for solving your challenging problems. Norfair powers several video analytics applications, such as the [face mask detection](https://tryolabs.com/blog/2020/07/09/face-mask-detection-in-street-camera-video-streams-using-ai-behind-the-curtain/) tool.

If you are interested, please [contact us](https://tryolabs.com/#contact).

## License

Copyright © 2020, [Tryolabs](https://tryolabs.com). Released under the [BSD 3-Clause](LICENSE).
