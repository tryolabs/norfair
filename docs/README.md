# API

## Tracker

The class in charge of performing the tracking of the detections produced by the detector. The `Tracker` class first needs to get instantiated as an object, and then continuously updated inside a video processing loop by feeding new detections into its [`update`](#tracker.update) method.

##### Arguments:

- `distance_function`: Function used by the tracker to determine the distance between newly detected objects and the objects the tracker is currently tracking. This function should take 2 arguments, the first being a detection of type [`Detection`](#detection), and the second a tracked object of type [`TrackedObject`](#trackedobject), and should return a `float` with the distance it calculates.
- `distance_threshold`: Defines what is the maximum distance that can constitute a match. Detections and tracked objects whose distance are above this threshold won't be matched by the tracker.
- `hit_inertia_min (optional)`: Each tracked objects keeps an internal hit inertia counter which tracks how often it's getting matched to a detection, each time it gets a match this counter goes up, and each time it doesn't it goes down. If it doesn't get any match for a certain amount of frames, and it then gets below the value set by this argument, the object is destroyed. Defaults to `10`.
- `hit_inertia_max (optional)`: Each tracked objects keeps an internal hit inertia counter which tracks how often it's getting matched to a detection, each time it gets a match this counter goes up, and each time it doesn't it goes down. This argument defines how large this inertia can grow, and therefore defines how long an object can live without getting matched to any detections. Defaults to `25`.
- `initialization_delay (optional)`: Each tracked object waits till its internal hit intertia counter goes over `hit_inertia_min` to be considered as a potential object to be returned to the user by the Tracker. The argument `initialization_delay` determines by how much the object's hit inertia counter must exceed `hit_inertia_min` to be considered as initialized and get returned to the user as a real object. Defaults to `(hit_inertia_max - hit_inertia_min) / 2`.
- `detection_threshold (optional)`: Sets the threshold at which the scores of the points in a detection being fed into the tracker must dip below to be ignored by the tracker. Defaults to `0`.
- `point_transience (optional)`: Each tracked object keeps track of how much often of the points its tracking has been getting matched. Points that are getting matches are said to be live, and points which aren't are said to not be live. This determines things like which points in a tracked object get drawn by [`draw_tracked_objects`](#draw_tracked_objects) and which don't. This argument determines how short lived points not getting matched are. Defaults to `4`.
- `filter_setup (optional)`: This parameter can be used to change the parameters of the Kalman Filter that is used by [`TrackedObject`](#trackedobject) instances. Defaults to [`FilterSetup()`](#filtersetup).
- `past_detections_length`: How many past detections to save for each tracked object. Norfair tries to distribute these past detections uniformly through the object's lifetime so they're more representative of it. Very useful if you want to add metric learning to your model, as you can associate an embedding to each detection and access them in your distance function. Defaults to `4`.

### Tracker.update

The function through which the detections found in each frame must be passed to the tracker.

##### Arguments:

- `detections (optional)`:  A list of [`Detection`](#detection)s which represent the detections found in the current frame being processed. If no detections have been found in the current frame, or the user is purposely skipping frames to improve video processing time, this argument should be set to None or ignored, as the update function is needed to advance the state of the Kalman Filters inside the tracker. Defaults to `None`.
- `period (optional)`: The user can chose not to run their detector on all frames, so as to process video faster. This parameter sets every how many frames the detector is getting ran, so that the tracker is aware of this situation and can handle it properly. This argument can be reset on each frame processed, which is useful if the user is dynamically changing how many frames the detector is skipping on a video when working in real-time. Defaults to `1`.

##### Returns:

- A list of [`TrackedObject`](#trackedobject)s.

## Detection

Detections returned by the detector must be converted to a `Detection` object before being used by Norfair.

##### Arguments and Properties:

- `points`: A numpy array of shape `(number of points per object, 2)`, with each row being a point expressed as `x, y` coordinates on the image. The number of points per detection must be constant for each particular tracker.
- `scores`: An array of length `number of points per object` which assigns a score to each of the points defined in `points`. This is used to inform the tracker of which points to ignore; any point with a score below `detection_threshold` will be ignored. This useful for cases in which detections don't always have every point detected, as is often the case in pose estimators.
- `data`: The place to store any extra data which may be useful when calculating the distance function. Anything stored here will be available to use inside the distance function. This enables the development of more interesting trackers which can do things like assign an appearance embedding to each detection to aid in its tracking.

## FilterSetup

This class can be used either to change some parameters of the [KalmanFilter](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html) that the tracker uses, or to fully customize the predictive filter implementation to use (as long as the methods and properties are compatible).
The former case only requires changing the default parameters upon tracker creation: `tracker = Tracker(..., filter_setup=FilterSetup(R=100))`, while the latter requires creating your own class extending `FilterSetup`, and rewriting its `create_filter` method to return your own customized filter.


##### Arguments:

Note that these arguments correspond to the same parameters of the [`filterpy.KalmanFilter` (see docs)](https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html) that this class returns.
- `R`: Multiplier for the sensor measurement noise matrix. Defaults to `4.0`.
- `Q`: Multiplier for the process uncertainty. Defaults to `0.1`.
- `P`: Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables. Defaults to `10.0`.


### FilterSetup.create_filter

This function returns a new predictive filter instance with the current setup, to be used by each new [`TrackedObject`](#trackedobject) that is created. This predictive filter will be used to estimate speed and future positions of the object, to better match the detections during its trajectory.

This method may be overwritten by a subclass of `FilterSetup`, in case that further customizations of the filter parameters or implementation are needed.

##### Arguments:

- `initial_detection`: numpy array of shape `(number of points per object, 2)`, corresponding to the [`Detection.points`](#detection) of the tracked object being born, which shall be used as initial position estimation for it.

##### Returns:
A new `filterpy.KalmanFilter` instance (or an API compatible object, since the class is not restricted by type checking).

## TrackedObject

The objects returned by the tracker's `update` function on each iteration. They represent the objects currently being tracked by the tracker.

##### Properties:

- `estimate`: Where the tracker predicts the point will be in the current frame based on past detections. A numpy array with the same shape as the detections being fed to the tracker that produced it.
- `id`: The unique identifier assigned to this object by the tracker.
- `last_detection`: The last detection that matched with this tracked object. Useful if you are storing embeddings in your detections and want to do metric learning, or for debugging.
- `last_distance`: The distance the tracker had with the last object it matched with.
- `age`: The age of this object measured in number of frames.
- `live_points`: A boolean mask with shape `(number of points per object)`. Points marked as `True` have recently been matched with detections. Points marked as `False` haven't and are to be considered as stale, and should be ignored. Functions like [`draw_tracked_objects`](#draw_tracked_objects) use this property to determine which points not to draw.
- `initializing_id`: On top of `id`, objects also have an `initializing_id` which is the id they are given internally by the `Tracker`, which is used for debugging. Each new object created by the `Tracker` starts as an uninitialized `TrackedObject`, which needs to reach a certain match rate to be converted into a full blown `TrackedObject`. This is the id assigned to `TrackedObject` while they are getting initialized. 

## Video

Class that provides a simple and pythonic way to interact with video. It returns regular OpenCV frames which enables the usage of the huge number of tools OpenCV provides to modify images.

A simple video inference loop can be defined with just:

```python
video = Video(input_path="video.mp4")
for frame in video:
    # Your modifications to the frame
    video.write(frame)
```

##### Arguments:

- `camera (optional)`: An integer representing the device id of the camera to be used as the video source. Webcams tend to have an id of `0`. Arguments `camera` and `input_path` can't be used at the same time, one must be chosen.
- `input_path (optional)`: A string consisting of the path to the video file to be used as the video source. Arguments `camera` and `input_path` can't be used at the same time, one must be chosen.
- `output_path (optional)`: The path to the output video to be generated. Can be a folder or a file name. Defaults to `"."`.
- `output_fps (optional)`: The frames per second at which to encode the output video file. If not provided it is set to be equal to the input video source's fps. This argument is useful when using live video cameras as a video source, where the user may know the input fps, but where the frames are being fed to the output video at a rate that is lower than the video source's fps, due to the latency added by the detector. This can result in an output video that was fed frames at a certain fps, that consequently encodes them at a higher fps (the video source's fps), which results in a sped up video output.
- `label (optional)`: Label to add to the progress bar that appears when processing the current video.
- `codec_fourcc (optional)`: OpenCV encoding for output video file.

### Video.write

Function to which the frames which will compose the output video should get passed to.

##### Arguments:

- `frame`: The OpenCV frame to write to file.

### Video.show

Function that displays the frame passed to it through a GUI. Usually used inside a video inference loop to show the output video.

##### Arguments:

- `frame`: The OpenCV frame to be displayed.
- `downsample_ratio (optional)`: How much to downsample the frame being show. Useful when streaming the GUI video display through a slow internet connection using something like X11 forwarding on an ssh connection. Defaults to `1`.

### Video.get_output_file_path

Function which returns the output path being used in case you are writing your frames to a video file. Useful if you didn't set `output_path`, and want to know what the autogenerated output file path by Norfair will be.

## draw_points

Function that draws a list of detections on a frame.

##### Arguments:

- `frame`: The OpenCV frame to draw on. Modified in place.
- `detections`: List of [`Detection`](#detection)s to be drawn.
- `radius (optional)`: Radius of the circles representing the detected points.
- `thickness (optional)`: Thickness of the circles representing the detected points.
- `color (optional)`: [`Color`](#color) of the circles representing the detected points.

## draw_tracked_objects

Function that draws a list of tracked objects on a frame.

##### Arguments:

- `frame`: The OpenCV frame to draw on. Modified in place.
- `objects`: List of [`TrackedObject`](#trackedobject)s to be drawn.
- `radius (optional)`: Radius of the circles representing the points estimated by the tracked objects.
- `color (optional)`: [`Color`](#color) of the circles representing the points estimated by the tracked objects.
- `id_size (optional)`: Size of the id number being drawn on each tracked object. The id wont get drawn if `id_size` is set to 0.
- `id_thickness (optional)`: Thickness of the id number being drawn on each tracked object.
- `draw_points (optional)`: Boolean determining if the function should draw the points estimated by the tracked objects. If set to `True` the points get drawn, if set to `False` only the id numbers get drawn. Defaults to `True`.

##  draw_debug_metrics

Function that draws debug information about the tracked objects on a frame. Usefull while developing your distance function.

##### Arguments:

- `frame`: The OpenCV frame to draw on. Modified in place.
- `objects`:  List of [`TrackedObject`](#trackedobject)s to be drawn.
- `text_size (optional)`: Size of the text displaying the debug information.
- `text_thickness (optional)`: Thickness of the text displaying the debug information.
- `color (optional)`: [`Color`](#color) of the text displaying the debug information.
- `only_ids (optional)`: List of ids that determines which objects to display the debug information of. Only the objects whose id is in this list will get their debug information drawn on the frame.
- `only_initializing_ids (optional)`: List of `initializing_id`s that determines which objects to display the debug information of. Only objects whose `initializing_id` is in this list will display their debug information. [`TrackedObject`](#trackedobject)s have an internal id called `initializing_id` which is used by the tracker to manage objects which may never be instantiated into full objects, it may be useful to filter objects by this id when debugging objects not correctly initializing, or initializing too often.

## Color

Object which represents an OpenCV color. Its properties are the colors which it can represent. For example, set `Color.blue` to get the OpenCV tuple representing the color blue.

##### Properties:

- `green`
- `white`
- `olive`
- `black`
- `navy`
- `red`
- `maroon`
- `grey`
- `purple`
- `yellow`
- `lime`
- `fuchsia`
- `aqua`
- `blue`
- `teal`
- `silver`

## print_objects_as_table

Function that prints a list of objects and their debug information as a table to console. Useful for debugging.

##### Arguments:

- `tracked_objects`: List of [`TrackedObject`](#trackedobject)s to print as a table.
