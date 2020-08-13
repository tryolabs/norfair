![Logo](/Users/joaqo/code/norfair/Norfair Logotipo.png)

Norfair is a Python library that adds object tracking to any type of detector. As long as the detector returns (x, y) points, such as the 2 points describing the bounding box of an object detector, or the 17 keypoints of each pose in a pose estimator, Norfair can handle it. Also, the user can customize how the tracking itself is done, so you can add additional information to it, like visual embeddings produced by the detector, or any appearance information that you think may help with tracking.

## Design

The reason Norfair allows for this customization of its tracker, and for a variable amount of tracked points per object, is because the decision of what distance function to use for matching detections with tracked objects is left for the user of the library.

Internally Norfair assigns a Kalman Filter to each point on each tracked object to estimate their future positions on each frame of the video, and does the matching between detections and their estimated future positions according to their distance, but the distance function that determines how this all works is left entirelly to the user. This keeps Norfair simple internally, and at the same time making it extremelly flexible, letting the user make the final object tracker they build be as complex as they see fit.

The distance function can be extremelly simple though, this is how the distance function for tracking the centroids of the objects returned by [FasterRCNN50?] from Detectron2 looks like:

```python
 def distance(detection, tracked_object):
     return np.linalg.norm(detection.points - tracked_object.estimate)
```

and this is how this distance function on Detectron2 only tracking cars looks like:

![](/Users/joaqo/traffic.gif)

[Make video smaller and longer with some of those video to gif sites, maybe put detections and tracking video side by side, and maybe link to longer video]

## Motivation

Trying out the latest state of the art detectors normally requires running repositiories which weren't intended to be easy to use. These usually tend to be repositories associated with a research paper describing a novel new way of doing detection, and they are intended to run a one-off evaluation script to get some metric for that particular research paper, which explains why they aren't usually easy to use as inference scripts, or why extracting the core model to use in external code isn't trivial. We wanted to see how the huge progress being made in object detectors / pose estimators / instance segmentation models could affect object tracking, so we built a simple object tracker named Norfair.

The issues mentioned in the above paragraph motivated the library to be extremelly modular, with 2 overall of usage:

1. If your detector can be extracted as a stand alone model, and you want to build code around it, Norfair provides a set of stand alone tools to allow for this. As an example of this use case, this is how the above traffic example with Detectron2 looks like in code:

   ```python
   from norfair import Detection, Tracker, Video, draw_tracked_objects
   
   # Your detector
   detector = Detectron2FasterRCNN()
   
   # Norfair
   video = Video(input_path="video.mp4")
   tracker = Tracker(distance_function=distance_fn)
   
   for frame in video:
       detections = detector(frame)
       # Convert detections from Detectron2 format into Norfair point format
       detections = [Detection(p) for p in detections.pred_boxes.get_centers().numpy()]
       tracked_objects = tracker.update(detections=detections)
   
       draw_tracked_objects(frame, tracked_objects)
       video.write(frame)
       
   # Make this code run? Make detectron2 the standard code example? Colab?
   ```

   On top of the object tracker, Norfair provides very useful drawing and video managing tools built on top of OpenCV, which are useful on their own even if you don't need the object tracking and want to avoid some of the boilerplate OpenCV often needs.

2. If your detector can't be extracted from its codebase, which tends to be the case of repositories associated to research papers, Norfair is simple enought that you can insert it into the repository. Therefore you can just look for the part of the code which returns the Detections, write a short conversion function to convert from their detection format into Norfair's, and you are up and running. If the researchers provide video processing or drawing code already, you don't need to use Norfair's video manager or drawing utils! In our tests this has sometimes made trying SOTA detectors as trackers go from taking days to minutes.

   ```python
   # Insert example of a diff or something for inserting norfair into alpha pose or something like that?
   ```

--- [Delete rest?]

## Use cases

A typical use case would be any software already using a detector (such as object detection, instance segmentantion, pose estimation, etc) on video, seeking to easily add tracking to it. Another way we've seen it used is to speed up video inference, by only running the detector every x frames, and using Norfair to interpolate the detections on the skipped frames [Think of more examples? Remove paragraph?]

## Instalation

```bash
pip install norfair
```

## API (Add links between refernces and definitions of things)

### Tracker

In charge of performing the tracking and storing the tracked objects as you iterate over the video. The `Tracker` class first needs to get instantiated as an object, and then continuously updated inside your video processing loop as you get new detections using its `update` method.

Arguments:

- `distance_function`: Function used by the tracker to determine the distance between newly detected objects and the tracked objects the tracker is tracking. This function should take 2 arguments, the first being a detection of type `Detection` and the second a tracked object of type `TrackedObject`, and should return a number (####check if it could be and Int and a Float!)

- `hit_inertia_min (optional)`: Each tracked objects keeps an internal hit inertia counter which tracks how often its getting matched to a detection, if it doesn't get any match for a certain amount of frames and it gets below the value set by this argument, the object is destroyed. Defaults to `10`.
- `hit_inertia_max (optional)`: Each tracked objects keeps an internal hit inertia counter which tracks how often its getting matched to a detection, this argument defines how large this inertia can grow. Defaults to `25`.
- `match_distance_threshold (optional)`: A detection and a matched object whose distance is below this number can get matched, the opposite is true for larger numbers. Defaults to `1`.
- `detection_threshold (optional)`: Sets the threshold at which a point's score must dip below to be ignored by the tracker. Defaults to 0.

### Tracker.update

The function to which the detections must be passed to as you iterate through your video.

Arguments:

- `detections (optional)`: The detections in the current frame being processed. A list of `Detection`s. If there aren't any detections, or you are skipping frames on purpose and the current frame is getting framed, the update function should still be run so the tracker keeps advancing, but with this argument set to None, or without any arguments.
- `period`: You can chose not to run your detector on all frames, so as to process the video faster. This parameter sets how many frames have been ignored by the detector since the last unignored frame. It can change as you process a video, which is useful if you are dynamically changing how many frames your detector is skipping on a video when working in real-time.

Returns:

- A list of `TrackedObjects`.

### Detection

The object which encodes how Norfair understands detections.

Arguments:

- `points`: A numpy array of shape `(number of points per object, 2)`, with each point being an `x, y` coordinate in the image, which the previous `2` references. The number of points per object must be the same for every detection fed into a particular tracker (###########test this).
- `scores`: An array of length `number of points per object` which assigns a score to each of the points defined in `points`. This is used to tell the tracker which points to ignore. This way we can use still use detections for which we don't haven't detected the position of every point.
- `data`: The argument to which any extra data you want to use in your distance function can be stored. Anything you store here will be available to you in your distance function so you can do interesting things like use embeddings for taking into account the detection's appearance and not just position when tracking objects.

### TrackedObject

The object returned by the tracker's `update` function on each iteration.

Properties:

- `estimate`: where the tracker predicts the point will be in the current frame based on past detections. A numpy array with the same shape as the detections being fed to the tracker that produced it.
- `id`: The unique identifier assigned to this object by the tracker.
- `last_detection`: The last detection that matched with this tracked object.
- `last_distance`: The distance the tracker had with the last object it matched with.
- `age`: the age of this object measured in number of frames.

### Video

In charge of processing video. Wraps around OpenCV with a much cleaner interface, but at the same time allowing you to use all of OpenCV's tools on the frames it lets you iterate over. Instantiate an object of class `Video` and then iterate over it as it outputs its frames.

Arguments:

- `camera=None`: An integer representing the device Id of the camera you want to get your video from. Webcams tend to have an Id of `0`. `camera` and `input_path` can't be used at the same time, you have to chose one.
- `input_path=None`: A string representing the path to your input video file. `camera` and `input_path` can't be used at the same time, you have to chose one.
- `output_path="."`: The output path to the video where you want to save your modified frames.
- `output_fps=None`: Allows you to chose the fps to encode your output file at. If not provided Norfair infers it from your video input source. This is useful for video cameras, where you may know the input fps, but where latency added to your video loop by your detector or other code can make your real fps be much lower than your input fps.
- `label=""`: Label to add to progress bar when processing the current video.
- `codec_fourcc=None`: Encoding for output file.

### Video.write

Function that writes a frame (usually a frame modified by the user in some way) to an output video file.

Arguments:

- `frame`: The OpenCV frame to be write to file.

### Video.show

Function that shows a frame (usually a frame modified by the user in some way) through a GUI video stream.

Arguments:

- `frame`: The OpenCV frame to be shown.
- `downsample_ratio=1`: How much to downsample the frame being show. Useful when looking at the GUI video through a slow internet connectioin using something like X11 forwarding on an ssh connection.

### Video.get_output_file_path

Function which returns the output path being used in case you are writing your frames to a video file. Usefull if you didn't set `output_path`, and want to know what the autogenerated output file path by Norfair will be.

### draw_points

Function that draws the points in a list of detections on a frame.

Arguments:

- `frame`: OpenCV frame to draw on. Modified in place.
- `detections`: List of `Detection`s to be drawn.
- `radius=None`: radius of the circles representing the detections.
- `thickness=None`: Thickness of the circles representing the detections.
- `color=None`: `Color` of the circles representing the detections.

### draw_tracked_objects

Function that draws a list of tracked objects on a frame.

Arguments:

- `frame`: OpenCV frame to draw on. Modified in place.
- `objects`: List of `TrackedObject`s to be drawn.
- `radius=None`: radius of the circles representing the tracked objects.
- `color=None`: `Color` of the circles representing the tracked objects.
- `id_size=None`: Size of the identifying number being drawn on each tracked object. The id wont get drawn if `id_size` is set to 0.
- `id_thickness=None`: Thickness of the identifying number being drawn on each tracked object.
- `draw_points=True`: Boolean determining if the function should draw the points estimated by the tracker. If set to true the points get drawn, if set to false only the id number gets drawn.

###  draw_debug_metrics

Function that draws debug information of tracked objects on a frame. Usefull while developing your distance function.

Arguments:

- `frame`: OpenCV frame to draw on. Modified in place.
- `objects`: List of `TrackedObject`s to be drawn.
- `text_size=None`: Size of the text displaying the debug information.
- `text_thickness=None`: Thickness of the text displaying the debug information.
- `color=None`: `Color` of the text displaying the debug information.
- `only_ids=None`: List of Ids that determine which objects to display the debug information of. Only objects whose id is in this list will display their debug information.
- `only_initializing_ids=None`: Objects have an internal id called initializing id which is used by the tracker to manage objects which may never be instantiated into a full object, it may be useful to filter by this id when debugging. List of Ids that determine which objects to display the debug information of. Only objects whose initializing id is in this list will display their debug information.

### Color

Object which represents an OpenCV color. Its properties are the colors which it can represent. For example, just set `Color.blue` to get the OpenCV tuple representing the color blue.

Properties:

- green
- white
- olive
- black
- navy
- red
- maroon
- grey
- purple
- yellow
- lime
- fuchsia
- aqua
- blue
- teal
- silver

### print_objects_as_table

Function that prints a list of objects and their debug information as a table to console. Useful for debugging.

Arguments:

- `tracked_objects`: List of `Object`s to print as a table.

---

## Tracker

Norfair is able to provide tracking for detectors of all types for two reasons:

First, it can track any number of points per object, so detections can range from consisting of a single point representing the centroid of the bounding box returned by an object detector, the 17 keypoints returned by a pose estimator, or even more. It will track all the keypoints, and it accepts a dynamically varying number of keypoints per object.

The second reason is that Norfair makes you chose your own function for determining the distance between detections and tracked objects. This usually adds a small amount of extra work for the user, but it adds a huge amount of diversity to what the tracker can do.

For example, you can define a very simple distance function such as:

```python
def distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)
```

Which just computes the vector distance between the points in each detection and the estimated points on each tracked object. Still, you can be more creative and do things like adding an embedding parameter to your detections to leverage the appearance of your objects into your distance function, or use they keypoints in a human pose to weigh your distances by each person's size, so the distance between people scales with how close they are to the video camera.

Norfair expects you to provide the points to track in a `Detection` object, which has to have a `.points` property with the keypoints in a numpy array with shape

## Video

Norfair provides a `Video` class to provide a simple and pythonic api to interact with video. It returns regular OpenCV frames which allows you to use the huge number of tools OpenCV provides to modify images.

You can get a simple video inference loop with just:
```python
video = Video(input_path="video.mp4")
for frame in video:
    # Your modifications to the frame
    video.write(frame)
```
we think the api is a bit more pythonic than the standard OpenCV api, provides a lot of good defaults and provides ammenities such as progress bars and more.

![file](/Users/joaqo/file.gif)

Or if you use you set `Video(camera=0)` to consume a video stream from your webcam:

[Sacar esto de la camara es ridiculo, hacer alarde de progress bars es medio lame]

![camera](/Users/joaqo/camera.gif)

It makes sense to use Norfair even if you don't need tracking, as its lightweight enough to warrant being used just for its video management features.

## Disclaimers

At the moment norfair works best with static cameras, like security cameras, though we are discussing adding support for cameras in movement.

