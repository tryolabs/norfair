# Getting Started

Norfair's goal is to easily track multiple objects in videos based on the frame-by-frame detections of a user-defined model.

## Model or Detector

We recommend first deciding and setting up the model and then adding Norfair on top of it.
Models trained for any form of [object detection](https://paperswithcode.com/task/object-detection) or [keypoint detection](https://paperswithcode.com/task/keypoint-detection) (including [pose estimation](https://paperswithcode.com/task/pose-estimation)) are all supported. You can check some of the integrations we have as examples:

- [Yolov7](https://github.com/tryolabs/norfair/tree/master/demos/yolov7), [Yolov5](https://github.com/tryolabs/norfair/tree/master/demos/yolov5) and [Yolov4](https://github.com/tryolabs/norfair/tree/master/demos/yolov4)
- [Detectron2](https://github.com/tryolabs/norfair/tree/master/demos/detectron2)
- [Alphapose](https://github.com/tryolabs/norfair/tree/master/demos/alphapose)
- [Openpose](https://github.com/tryolabs/norfair/tree/master/demos/openpose)
- [MMDetection](https://github.com/tryolabs/norfair/tree/master/demos/mmdetection)

Any other model trained on one of the supported tasks is also supported and should be easy to integrate with Norfair, regardless of whether it uses Pytorch, TensorFlow, or other.

If you are unsure of which model to use, [Yolov7](https://github.com/WongKinYiu/yolov7) is a good starting point since it's easy to set up and offers models of different sizes pre-trained on object detection and pose estimation.

!!! Note
    Norfair is a Detection-Based-Tracker (DBT) and as such, its performance is highly dependent on the performance of the model of choice.

The detections from the model will need to be wrapped in an instance of [Detection][norfair.tracker.Detection] before passing them to Norfair.

## Install

Installing Norfair is extremely easy, simply run `pip install norfair` to install the latest version from [PyPI](https://pypi.org/project/norfair/).

You can also install the latest version from the master branch using `pip install git+https://github.com/tryolabs/norfair.git@master#egg=norfair`

## Video

Norfair offers optional functionality to process videos (mp4 and mov formats are supported) or capture a live feed from a camera.
To use this functionality you need to install Norfair with the `video` extra using this command: `pip install norfair[video]`.

Check the [Video class][norfair.video.Video] for more info on how to use it.

## Tracking

Let's dive right into a simple example in the following snippet:

``` python
from norfair import Detection, Tracker, Video, draw_tracked_objects

detector = MyDetector()  # Set up a detector
video = Video(input_path="video.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=100)

for frame in video:
   detections = detector(frame)
   norfair_detections = [Detection(points) for points in detections]
   tracked_objects = tracker.update(detections=norfair_detections)
   draw_tracked_objects(frame, tracked_objects)
   video.write(frame)
```

The tracker is created and then the detections are fed to it one frame at a time in order. This method is called _online tracking_ and allows Norfair to be used in live feeds and real-time scenarios where future frames are not available.

Norfair includes functionality for creating an output video with drawings which is useful for evaluating and debugging. We usually start with this simple setup and move from there.

### Next Steps

The next steps depend a lot on your goal and the result of evaluating the output videos, nevertheless here are some pointers that might help you solve common problems

#### Detection Issues

Most common problem is that the tracking has errors or is not precise enough. In this case, the first thing to check is whether this is a detection error or a tracking error. As mentioned above if the detector fails the tracking will suffer.

To debug this use [`draw_points`][norfair.drawing.draw_points] or [`draw_boxes`][norfair.drawing.draw_boxes] to inspect the detections and analyze if they are precise enough. If you are filtering the detections based on scores, this is a good time to tweak the threshold. If you decide that the detections are not good enough you can try a different architecture, a bigger version of the model, or consider fine-tuning the model on your domain.


#### Tracking Issues

After inspecting the detections you might find issues with the tracking, several things can go wrong with tracking but here is a list of common errors and things to try:

- Objects take **too long to start**, this can have multiple causes:
    - `initialization_delay` is too big on the Tracker. Makes the TrackedObject stay on initializing for too long, `3` is usually a good value to start with.
    - `distance_threshold` is too small on the Tracker. Prevents the Detections to be matched with the correct TrackedObject. The best value depends on the distance used.
    - Incorrect `distance_function` on the Tracker. Some distances might not be valid in some cases, for instance, if using IoU but the objects in your video move so quickly that there is never an overlap between the detections of consecutive frames. Try different distances, `euclidean` or `create_normalized_mean_euclidean_distance` are good starting points.
- Objects take **too long to disappear**. Lower `hit_counter_max` on the Tracker.
- Points or bounding boxes **jitter too much**. Increase `R` (measurement error) or lower `Q` (estimate or process error) on the `OptimizedKalmanFilterFactory` or `FilterPyKalmanFilterFactory`. This makes the Kalman Filter put less weight on the measurements and trust more on the estimate, stabilizing the result.
- **Camera motion** confuses the Tracker. If the camera moves, the apparent movement of objects can become too erratic for the Tracker. Use `MotionEstimator`.
- **Incorrect matches** between Detections and TrackedObjects, a couple of scenarios can cause this:
    - `distance_threshold` is too big so the Tracker matches Detections to TrackedObjects that are simply too far. Lower the threshold until you fix the error, the correct value will depend on the distance function that you're using.
    - Mismatches when objects overlap. In this case, tracking becomes more challenging, usually, the quality of the detection degrades causing one of the objects to be missed or creating a single big detection that includes both objects. On top of the detection issues, the tracker needs to decide which detection should be matched to which TrackedObject which can be error-prone if only considering spatial information. The solution is not easy but incorporating the notion of the appearance similarity based on some kind of embedding to your distance_function can help.
- Can't **recover** an object **after occlusions**. Use ReID distance, see [this demo](https://github.com/tryolabs/norfair/tree/master/demos/reid) for an example but for real-world use you will need a good ReID model that can provide good embeddings.
