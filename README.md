![Norfair by Tryolabs logo](docs/logo.png)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/norfair/blob/master/demos/yolov7/src/yolov7_demo.ipynb) [![Documentation](https://img.shields.io/badge/api-reference-blue?logo=readthedocs)](https://github.com/tryolabs/norfair/blob/master/docs/README.md) [![Board](https://img.shields.io/badge/project-board-blue?logo=github)](https://github.com/tryolabs/norfair/projects/1) ![Build status](https://github.com/tryolabs/norfair/workflows/CI/badge.svg?branch=master) [![DOI](https://zenodo.org/badge/276473370.svg)](https://zenodo.org/badge/latestdoi/276473370)

Norfair is a customizable lightweight Python library for real-time 2D object tracking.

Using Norfair, you can add tracking capabilities to any detector with just a few lines of code.

<img src="/docs/soccer.gif" alt="Tracking soccer players with Norfair and a moving camera." width="500px" />

## Features

- Any detector expressing its detections as a series of `(x, y)` coordinates can be used with Norfair. This includes detectors performing object detection, pose estimation, and keypoint detection (see [examples](#examples--demos)).

- The function used to calculate the distance between tracked objects and detections is defined by the user, making the tracker extremely customizable. This function can make use of any extra information, such as appearance embeddings, which can heavily improve tracking performance.

- Modular. It can easily be inserted into complex video processing pipelines to add tracking to existing projects. At the same time, it is possible to build a video inference loop from scratch using just Norfair and a detector.

- Fast. The only thing bounding inference speed will be the detection network feeding detections to Norfair.

Norfair is built, used and maintained by [Tryolabs](https://tryolabs.com).

## Installation

Norfair currently supports Python 3.6+.

For the minimal version, install as:

```bash
pip install norfair
```

To make Norfair install the dependencies to support more features, install as:

```bash
pip install norfair[video]  # Adds several video helper features running on OpenCV
pip install norfair[metrics]  # Supports running MOT metrics evaluation
pip install norfair[metrics,video]  # Everything included
```

If the needed dependencies are already present in the system, installing the minimal version of Norfair is enough for enabling the extra features. This is particuarly useful for embedded devices, where installing compiled dependencies can be difficult, but they can sometimes come preinstalled with the system.

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

The video and drawing tools use OpenCV frames, so they are compatible with most Python video code available online. The point tracking is based on [SORT](https://arxiv.org/pdf/1602.00763.pdf) generalized to detections consisting of a dynamically changing number of points per detection.

## Examples & demos

We provide several examples of how Norfair can be used to add tracking capabilities to different detectors, and also showcase more advanced features.

> Note: for ease of reproducibility, we provide Dockerfiles for all the demos. Even though Norfair does not need a GPU, the default configuration of most demos requires a GPU to be able to run the detectors. For this, make sure you install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so that you GPU can be shared with Docker.
>
> It is possible to run several demos with a CPU, but you will have to modify the scripts or tinker with the installation of their dependencies.

### Adding tracking to different detectors

Most tracking demos are showcased with vehicles and pedestrians, but the detectors are generally trained with many more classes from the [COCO dataset](https://cocodataset.org/).

1. [YOLOv7](demos/yolov7): tracking object centroids.
2. [YOLOv5](demos/yolov5): tracking object centroids.
3. [YOLOv4](demos/yolov4): tracking object centroids.
4. [Detectron2](demos/detectron2): tracking object centroids.
5. [AlphaPose](demos/alphapose): tracking human keypoints (pose estimation) and inserting Norfair into a complex existing pipeline using.
6. [OpenPose](demos/openpose): tracking human keypoints.

### Advanced features

1. [Speed up pose estimation by extrapolating detections](demos/openpose) using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
2. [Track both bounding boxes and human keypoints](demos/keypoints_bounding_boxes) (multi-class), unifying the detections from a YOLO model and OpenPose.
3. [Re-identification (ReID)](demos/reid) of tracked objects using appearance embeddings. This is a good starting point for scenarios with a lot of occlusion, in which the Kalman filter alone would struggle.
4. [Accurately track objects even if the camera is moving](demos/camera_motion), by estimating camera motion potentially accounting for pan, tilt, rotation, movement in any direction, and zoom.

### Benchmarking and profiling

1. [Kalman filter and distance function profiling](demos/profiling) using [TRT pose estimator](https://github.com/NVIDIA-AI-IOT/trt_pose).
2. Computation of [MOT17](https://motchallenge.net/data/MOT17/) scores using [motmetrics4norfair](demos/motmetrics4norfair).

![Norfair OpenPose Demo](docs/openpose_skip_3_frames.gif)

## Motivation

Trying out the latest state-of-the-art detectors normally requires running repositories that weren't intended to be easy to use. These tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are therefore intended to be run as a one-off evaluation script to get some result metric to publish on a particular research paper. This explains why they tend to not be easy to run as inference scripts, or why extracting the core model to use in another standalone script isn't always trivial.

Norfair was born out of the need to quickly add a simple layer of tracking over a wide range of newly released SOTA detectors. It was designed to seamlessly be plugged into a complex, highly coupled code base, with minimum effort. Norfair provides a series of modular but compatible tools, which you can pick and choose to use in your project.

## Comparison to other trackers

Norfair's contribution to Python's object tracker library repertoire is its ability to work with any object detector by being able to work with a variable number of points per detection, and the ability for the user to heavily customize the tracker by creating their own distance function.

If you are looking for a tracker, here are some other projects worth noting:

- [**OpenCV**](https://opencv.org) includes several tracking solutions like [KCF Tracker](https://docs.opencv.org/3.4/d2/dff/classcv_1_1TrackerKCF.html) and [MedianFlow Tracker](https://docs.opencv.org/3.4/d7/d86/classcv_1_1TrackerMedianFlow.html) which are run by making the user select a part of the frame to track, and then letting the tracker follow that area. They tend not to be run on top of a detector and are not very robust.
- [**dlib**](http://dlib.net) includes a correlation single object tracker. You have to create your own multiple object tracker on top of it yourself if you want to track multiple objects with it.
- [**AlphaPose**](https://github.com/MVIG-SJTU/AlphaPose) just released a new version of their human pose tracker. This tracker is tightly integrated into their code base, and to the task of tracking human poses.
- [**SORT**](https://github.com/abewley/sort) and [**Deep SORT**](https://github.com/nwojke/deep_sort) are similar to this repo in that they use Kalman filters (and a deep embedding for Deep SORT), but they are hardcoded to a fixed distance function and to tracking boxes. Norfair also adds some filtering when matching tracked objects with detections, and changes the Hungarian Algorithm for its own distance minimizer. Both these repos are also released under the GPL license, which might be an issue for some individuals or companies because the source code of derivative works needs to be published.

## Benchmarks

[MOT17](https://motchallenge.net/data/MOT17/) results obtained using [motmetrics4norfair](https://github.com/tryolabs/norfair/tree/master/demos/motmetrics4norfair) demo script. Hyperparameters were tuned for reaching a high `MOTA` on this dataset. A more balanced set of hyperparameters, like the default ones used in the other demos, is recommended for production.

|                | Rcll  | Prcn  |   GT MT PT ML    |      FP FN IDs FM      |  MOTA MOTP  |
| :------------: | :---: | :---: | :--------------: | :--------------------: | :---------: |
|  MOT17-13-DPM  | 18.5% | 85.8% |   110 6 31 73    |    355 9490 90 100     | 14.7% 26.7% |
| MOT17-04-FRCNN | 56.3% | 93.2% |   83 18 43 22    |   1959 20775 90 104    | 52.0% 10.7% |
| MOT17-11-FRCNN | 61.4% | 92.9% |   75 18 34 23    |     443 3639 65 62     | 56.1% 10.2% |
|  MOT17-04-SDP  | 77.5% | 97.4% |    83 49 25 9    |   1003 10680 232 257   | 74.9% 13.2% |
|  MOT17-13-SDP  | 57.6% | 83.3% |   110 46 26 38   |   1349 4934 164 163    | 44.6% 20.0% |
|  MOT17-05-DPM  | 38.1% | 82.9% |   133 11 58 64   |    544 4281 90 100     | 28.9% 24.3% |
|  MOT17-09-DPM  | 59.7% | 75.1% |    26 4 17 5     |   1052 2147 120 117    | 37.7% 26.3% |
|  MOT17-10-DPM  | 37.5% | 84.9% |    57 6 19 32    |    856 8024 127 153    | 29.8% 24.9% |
|  MOT17-02-SDP  | 50.9% | 75.9% |   62 11 38 13    |   3003 9122 272 290    | 33.3% 18.4% |
|  MOT17-11-DPM  | 54.2% | 84.7% |   75 12 24 39    |     927 4321 87 63     | 43.5% 21.7% |
| MOT17-09-FRCNN | 58.5% | 98.5% |    26 7 17 2     |     49 2209 40 39      | 56.8% 9.6%  |
|  MOT17-11-SDP  | 75.7% | 90.9% |   75 34 30 11    |    719 2297 112 105    | 66.9% 14.1% |
| MOT17-02-FRCNN | 36.5% | 79.5% |    62 7 26 29    |   1751 11796 124 136   | 26.4% 13.4% |
| MOT17-05-FRCNN | 54.9% | 90.0% |   133 23 69 41   |     420 3123 78 84     | 47.7% 18.0% |
|  MOT17-04-DPM  | 42.5% | 83.5% |    83 7 44 32    |   3985 27335 406 432   | 33.2% 21.1% |
|  MOT17-10-SDP  | 74.1% | 88.0% |    57 30 24 3    |   1295 3323 283 286    | 61.8% 19.8% |
| MOT17-10-FRCNN | 60.9% | 75.7% |    57 14 37 6    |   2507 5020 310 306    | 39.0% 17.2% |
|  MOT17-09-SDP  | 67.7% | 94.6% |    26 12 14 0    |     204 1722 54 56     | 62.8% 13.0% |
|  MOT17-02-DPM  | 20.2% | 81.4% |    62 5 14 43    |   856 14833 113 109    | 15.0% 24.6% |
| MOT17-13-FRCNN | 59.3% | 74.7% |   110 33 53 24   |   2334 4735 342 331    | 36.3% 18.4% |
|  MOT17-05-SDP  | 66.8% | 87.9% |   133 32 81 20   |    637 2299 134 133    | 55.6% 16.5% |
|    OVERALL     | 53.7% | 87.3% | 1638 385 724 529 | 26248 156135 3333 3426 | 44.9% 16.4% |

## Commercial support

Tryolabs can provide commercial support, implement new features in Norfair or build video analytics tools for solving your challenging problems. Norfair powers several video analytics applications, such as the [face mask detection](https://tryolabs.com/blog/2020/07/09/face-mask-detection-in-street-camera-video-streams-using-ai-behind-the-curtain/) tool.

If you are interested, please [contact us](mailto:hello@tryolabs.com).

## Citing Norfair

For citations in academic publications, please export your desired citation format (BibTeX or other) from [Zenodo](https://doi.org/10.5281/zenodo.5146253).

## License

Copyright © 2022, [Tryolabs](https://tryolabs.com). Released under the [BSD 3-Clause](LICENSE).
