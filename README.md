![Logo](docs/logo.png)

Norfair is a lightweight Python library for adding object tracking to any type of detector. As long as the detector returns points expressed as (x, y) coordinates, such as the 2 points describing the bounding box of an object detector, or the 17 keypoints forming the poses in a pose estimator, Norfair can handle it. Also, the injection of additional information to each object, such as embeddings produced by the detector, or any other appearance information that may improve tracking, is supported.

## Usage

The reason Norfair is so lax with its requirements, is because the decision of what distance function to use for matching detections with tracked objects is left to the user of the library.

Internally, Norfair estimates the future position of each point based on its past position, and then tries to match these estimated positions with the newly detected points provided by the detector. The interesting part is that the way in which the distance between tracked objects and new detections is calculated is left for the user to decide. This keeps Norfair small internally, while at the same time making it extremelly flexible; the user can build their object tracker to be as complex as they need.

Having said this, the distance function can also be extremelly simple. The simplest case would probably be tracking one point per object and using the Euclidean distance as our distance function. This is how that would look like:

```python
 def euclidean_distance(detection, tracked_object):
     return np.linalg.norm(detection.points - tracked_object.estimate)
```

And this is how this distance function looks working on a traffic video when using [Detectron2](https://detectron2.readthedocs.io/tutorials/install.html) as our detector, and only using the centroids of the bounding boxes:

![](docs/traffic.gif)

On the left you can see the points we get from Detectron, and on the right how Norfair tracks them. Even straightforward distance functions like this one can work when the tracking needed is simple. Click here to see a longer version of this video.

Norfair is built to be modular. This way the user can chose between inserting only a couple of Norfair's tools inside an already running video detection loop to add tracking to it, or create a new inference loop from scratch using just the building blocks provided by Norfair. [ Remove ?]

Following this last way of doing things, this is how the full code for the previous car tracking example looks like, including the code needed to set up Detectron2:

```python
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from norfair import Detection, Tracker, Video, draw_tracked_objects, draw_points

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
    # Convert Detectron2 detections into Norfair's Detection objects
    detections = [Detection(p) for p in detections['instances'].pred_boxes.get_centers().cpu().numpy()]
    tracked_objects = tracker.update(detections=detections)
    draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
```

## Motivation [ Remove ? ]

Trying out the latest state of the art detectors normally requires running repositiories which weren't intended to be easy to use. These tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are therefore intended to be run as a one-off evaluation script to get some result metric to publish on their particular research paper. This explains why they tend to not be easy to run as inference scripts, or why extracting the core model to use in a stand alone way isn't always trivial.

Norfair was born out of the need to quickly add a simple layer of tracking over a wide range of newly released SOTA detectors. For this reason it was designed to seamlessly be plugged into a complex, highly coupled code base, with minium effort. Norfair provides a series of independent but compatible tools, which you can pick and chose to use in your project.

```This is the diff that shows how adding Norfair's tracking to the AlphaPose repository looks like: [Maybe use a more researchy paper? Or maybe alpha pose is better because we can compare the trackings?]```

## Installation

```bash
pip install norfair
```

## Documentation

You can find the documentation for Norfair's API [here](docs/API.md).

## Examples

[Example of using Norfair to skip frames on another detector, maybe instance segmentation? Use this to show off progress bar, maybe showing how it's fps is running X times faster than if we inferred all frames or whatever. Maybe create a function to move not tracked points based on how tracked points moved?]

## Disclaimers

Norfair's point prediction is done using KalmanFilters in a similar way to [SORT](https://arxiv.org/pdf/1602.00763.pdf). For this reason, Norfair works best with static cameras, like security cameras, though the addition of support for moving cameras is being discussed by the team.

