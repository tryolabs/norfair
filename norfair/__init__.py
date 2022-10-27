"""
A customizable lightweight Python library for real-time multi-object tracking.

Examples
--------
>>> from norfair import Detection, Tracker, Video, draw_tracked_objects
>>> detector = MyDetector()  # Set up a detector
>>> video = Video(input_path="video.mp4")
>>> tracker = Tracker(distance_function="frobenious", distance_threshold=50)
>>> for frame in video:
>>>    detections = detector(frame)
>>>    norfair_detections = [Detection(points) for points in detections]
>>>    tracked_objects = tracker.update(detections=norfair_detections)
>>>    draw_tracked_objects(frame, tracked_objects)
>>>    video.write(frame)
"""
import sys

from .distances import *
from .drawing import *
from .filter import (
    FilterPyKalmanFilterFactory,
    NoFilterFactory,
    OptimizedKalmanFilterFactory,
)
from .tracker import Detection, Tracker
from .utils import get_cutout, print_objects_as_table
from .video import Video

if sys.version_info >= (3, 8):
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
elif sys.version_info < (3, 8):
    import importlib_metadata

    __version__ = importlib_metadata.version(__name__)
