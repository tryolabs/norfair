![Logo](docs/logo.png)

Norfair is a lightweight Python library for adding object tracking to any type of detector. As long as the detector returns points expressed as (x, y) coordinates, such as the 2 points describing the bounding box of an object detector, or the 17 keypoints forming the poses in a pose estimator, Norfair can handle it. Also, using additional information such as the embeddings produced by the detector, or any other appearance information that may improve tracking, is supported.

## Usage

The reason Norfair is so lax with its requirements, is because the decision of what distance function to use for matching detections with tracked objects is left to the user of the library.

Internally, Norfair estimates the future position of each point based on its past position. It then tries to match these estimated positions with newly detected points provided by the detector. The interesting part is that the function which calculates the distance between tracked objects and new detections is left for the user of the library to define. Each object tracker can therefore be made as simple or complex as needed.

The following is an example of a particularly simple distance function calculating the Euclidean distance between objects and detections. This is possibly the simplest distance function you could use in Norfair, as it uses just one single point per detection/object.

```python
 def euclidean_distance(detection, tracked_object):
     return np.linalg.norm(detection.points - tracked_object.estimate)
```

To get the single point detections to use with this distance function we use [Detectron2](https://detectron2.readthedocs.io/tutorials/install.html), and just use the centroids of the bounding boxes it produces around cars.

![](docs/traffic.gif)

On the left you can see the points we get from Detectron2, and on the right how Norfair tracks them. Even a straightforward distance function like this one can work when the tracking needed is simple.

Norfair also provides several useful tools for creating a video inference loop. Here is what the full code for creating the previous example looks like, including the code needed to set up Detectron2:

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

The video and drawing tools use OpenCV frames, so they are compatible with most Python video code available online.

## Motivation

Trying out the latest state of the art detectors normally requires running repositiories which weren't intended to be easy to use. These tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are therefore intended to be run as a one-off evaluation script to get some result metric to publish on their particular research paper. This explains why they tend to not be easy to run as inference scripts, or why extracting the core model to use in a stand alone way isn't always trivial.

Norfair was born out of the need to quickly add a simple layer of tracking over a wide range of newly released SOTA detectors. For this reason it was designed to seamlessly be plugged into a complex, highly coupled code base, with minium effort. Norfair provides a series of modular but compatible tools, which you can pick and chose to use in your project.

## Installation

```bash
pip install norfair
```

## Documentation

You can find the documentation for Norfair's API [here](docs/API.md).

## Examples

### AlphaPose

Sometimes detection models are deeply coupled to their code base, and extracting them for using in your own video loop is very hard. This is very common in repositories based on research papers, which are usually meant to execute a one-off evaluation job for getting the metrics for their associated paper. [ Delete I think ]

An example of a model which is very deeply coupled to its code base is AlphaPose. With Norfair you can try out your own custom tracker on the very accurate poses produced by AlphaPose by just applying this git diff (with regards to this [commit](https://github.com/MVIG-SJTU/AlphaPose/commit/ded84d450faf56227680f0527ff7e24ab7268754)) into AlphaPose itself, and therefore avoiding the difficult job of decoupling the model from the code base. Just apply this diff and use their `video_demo.py` script on your video.

```diff
diff --git a/dataloader.py b/dataloader.py
index ed6ee90..a7dedb0 100644
--- a/dataloader.py
+++ b/dataloader.py
@@ -17,6 +17,8 @@ import cv2
 import json
 import numpy as np
 import sys
+sys.path.append("/home/lalo/norfair")
+import norfair
 import time
 import torch.multiprocessing as mp
 from multiprocessing import Process
@@ -606,6 +608,17 @@ class WebcamLoader:
         # indicate that the thread should be stopped
         self.stopped = True
 
+detection_threshold = 0.2
+keypoint_dist_threshold = None
+def keypoints_distance(detected_pose, tracked_pose):
+    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
+    match_num = np.count_nonzero(
+        (distances < keypoint_dist_threshold)
+        * (detected_pose.scores > detection_threshold)
+        * (tracked_pose.last_detection.scores > detection_threshold)
+    )
+    return 1 / (1 + match_num)
+
 class DataWriter:
     def __init__(self, save_video=False,
                 savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
@@ -624,6 +637,11 @@ class DataWriter:
         if opt.save_img:
             if not os.path.exists(opt.outputpath + '/vis'):
                 os.mkdir(opt.outputpath + '/vis')
+        self.tracker = norfair.Tracker(
+            distance_function=keypoints_distance,
+            distance_threshold=0.3,
+            detection_threshold=0.2
+        )
 
     def start(self):
         # start a thread to read frames from the file video stream
@@ -672,7 +690,15 @@ class DataWriter:
                     }
                     self.final_result.append(result)
                     if opt.save_img or opt.save_video or opt.vis:
-                        img = vis_frame(orig_img, result)
+                        img = orig_img.copy()
+                        global keypoint_dist_threshold
+                        keypoint_dist_threshold = img.shape[0] / 30
+                        detections = [
+                            norfair.Detection(p['keypoints'].numpy(), scores=p['kp_score'].squeeze().numpy())
+                            for p in result['result']
+                        ]
+                        tracked_objects = self.tracker.update(detections=detections)
+                        norfair.draw_tracked_objects(img, tracked_objects)
                         if opt.vis:
                             cv2.imshow("AlphaPose Demo", img)
                             cv2.waitKey(30)
```

This produces the following results:

![alphapose](docs/alphapose.gif)



### OpenPose

If you just want to accelerate your pose detection, you can use Norfair to only run one out of every 3 frames through your detector and let Norfair interpolate the detections through the rest of the frames:

```python
import norfair
from norfair import Detection, Tracker, Video

import numpy as np
import yaml
import sys

frame_skip_period = 3
detection_threshold = 0.2
distance_threshold = 0.3

class OpenposeDetector():
    def __init__(self):
        with open("./demos/openpose_config.yml", 'r') as stream:
            open_pose_config = yaml.safe_load(stream)["openpose"]
        openpose_dir = open_pose_config['dir']
        sys.path.append(openpose_dir + "/build/python/openpose")
        from openpose import OpenPose  # noqa
        open_pose_config["default_model_folder"] = openpose_dir + "/models/"
        self.detector = OpenPose(open_pose_config)

    def __call__(self, image):
        return self.detector.forward(image, False)

def keypoints_distance(detected_pose, tracked_pose):
    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
    match_num = np.count_nonzero(
        (distances < keypoint_dist_threshold)
        * (detected_pose.scores > detection_threshold)
        * (tracked_pose.last_detection.scores > detection_threshold)
    )
    return 1 / (1 + match_num)

pose_detector = OpenposeDetector()
video = Video(input_path="video.mp4")
tracker = Tracker(distance_function=keypoints_distance,
                  distance_threshold=distance_threshold,
                  detection_threshold=detection_threshold)
keypoint_dist_threshold = video.input_height / 30

for i, frame in enumerate(video):
    if i % frame_skip_period == 0:
        detected_poses = pose_detector(frame)
        detections = [] if not detected_poses.any() else [
            Detection(p, scores=s)
            for (p, s) in zip(detected_poses[:, :, :2], detected_poses[:, :, 2])
        ]
        tracked_objects = tracker.update(detections=detections, period=frame_skip_period)
        norfair.draw_points(frame, detections)
    else:
        tracked_objects = tracker.update()
    norfair.draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
```

We are skipping 2 out of every 3 frames, which should make your video process 3 times faster, as the time added by running the tracker itself is negligible when compared to not having to run 2 inferences on a deep neural network.

This is how the results look like:

![openpose_skip_3_frames](docs/openpose_skip_3_frames.gif)

