![Logo](docs/logo.png)

Norfair is a lightweight Python library for adding object tracking to any type of detector. As long as the detector returns points expressed as (x, y) coordinates grouped into detections, such as the 2 points describing the bounding box of an object detector, or the 17 keypoints forming the poses in a pose estimator, Norfair can handle it. The injection of additional useful information to each detection, such as any embedding produced by the detector, or any appearance information that may improve tracking, is also supported.

[Video showing points -> tracked obects and explaining exactly what norfair does, and saying something like "norfair working with openpose 2 or whatever", and maybe a code snippet bellow if we add default distance funcitons to norfair?]

## Design

The reason Norfair is so lax with its requirements, is because the decision of what distance function to use for matching detections with tracked objects is left to the user of the library.

Internally, Norfair estimates the future position of each point based on their past positions, and then tries to match these estimated positions with the newly detected points provided by the detector. The interesting part is that the way this distance between tracked objects and new detections is calculated is left for the user to decide. This keeps Norfair simple internally, while at the same time making it extremelly flexible; the user can build their object tracker to be as complex as they need.

Having said this, the distance function can also be extremelly simple. This is how the distance function for tracking cars by tracking the centroids of the bounding boxes returned by [FasterRCNN50?] from Detectron2 looks like:

```python
 def ceintroid_distance(detection, tracked_object):
     return np.linalg.norm(detection.points - tracked_object.estimate)
```

and this is how this distance function looks working on a traffic video:

![](docs/traffic.gif)

[Make video smaller and longer with some of those video to gif sites, maybe put detections and tracking video side by side, and link to longer video]

Norfair is meant to be as modular and simple to use as possible, so the user can chose between using only a couple of Norfair's tools inside an already running video inference loop, or create a new inference loop from scratch using just Norfair and a detector. Following the latter way of doing this, this is how the code for the previous car tracking example looks like:

```python
from norfair import Detection, Tracker, Video, draw_tracked_objects

# Setting up Detectron2
detector = Detectron2FasterRCNN()

# Norfair
video = Video(input_path="video.mp4")
tracker = Tracker(distance_function=ceintroid_distance)

for frame in video:
    detections = detector(frame)
    # Convert detections from Detectron2 format into Norfair point format
    detections = [Detection(p) for p in detections.pred_boxes.get_centers().numpy()]
    tracked_objects = tracker.update(detections=detections)

    draw_tracked_objects(frame, tracked_objects)
    video.write(frame)
    
# Make this code run? Make detectron2 the standard code example? Colab?
```

## Motivation

Trying out the latest state of the art detectors normally requires running repositiories which weren't originally intended to be easy to use. These usually tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are therefore intended to be run as a one-off evaluation script to get some result metric for their particular research paper. This explains why they tend to not be easy to use as inference scripts, or why extracting the core model to use in a stand alone way isn't always trivial.

Norfair was born out of the need to quickly add a simple layer of tracking over a wide range of newly released SOTA detectors. For this reason it was designed to seamlessly be plugged into a complex, highly coupled code base, with minium effort. Norfair provides a series of independent but compatible tools, which you can pick and chose to use in your project

```This is the diff that shows how adding Norfair's tracking to the AlphaPose repository looks like: [Maybe use a more researchy paper? Or maybe alpha pose is better because we can compare the trackings?]```

## Examples

Example of using Norfair to skip frames on another detector, maybe instance segmentation? Use this to show off progress bar, maybe showing how it's fps is running X times faster than if we inferred all frames or whatever.

## Installation

```bash
pip install norfair
```

## Documentation

You can find the documentation for Norfair's API here.

## Comparisons with other trackers

## Video

Norfair provides a `Video` class to provide a simple and pythonic api to interact with video. It returns regular OpenCV frames which allows you to use the huge number of tools OpenCV provides to modify images.

You can get a simple video inference loop with just:

```
video = Video(input_path="video.mp4")
for frame in video:
    # Your modifications to the frame
    video.write(frame)
```

It makes sense to use Norfair even if you don't need tracking, as its lightweight enough to warrant being used just for its video management features.

![camera](file:///Users/joaqo/camera.gif?lastModify=1598507860)

[Sacar esto de la camara es ridiculo, hacer alarde de progress bars es medio lame]

Or if you use you set `Video(camera=0)` to consume a video stream from your webcam:

![file](file:///Users/joaqo/file.gif?lastModify=1598507860)

we think the api is a bit more pythonic than the standard OpenCV api, provides a lot of good defaults and provides ammenities such as progress bars and more.

## Disclaimers

At the moment norfair works best with static cameras, like security cameras, though we are discussing adding support for cameras in movement.

