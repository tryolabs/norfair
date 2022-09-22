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
