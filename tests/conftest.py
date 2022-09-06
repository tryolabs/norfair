import numpy as np
import pytest


@pytest.fixture
def mock_det():
    class FakeDetection:
        def __init__(self, points, scores=None) -> None:
            if not isinstance(points, np.ndarray):
                points = np.array(points)
            self.points = points

            if scores is not None and not isinstance(scores, np.ndarray):
                scores = np.array(scores)
                if scores.ndim == 0 and points.shape[0] > 1:
                    scores = np.full(points.shape[0], scores)
            self.scores = scores

    return FakeDetection


@pytest.fixture
def mock_obj(mock_det):
    class FakeTrackedObject:
        def __init__(self, points, scores=None):
            if not isinstance(points, np.ndarray):
                points = np.array(points)

            self.estimate = points
            self.last_detection = mock_det(points, scores=scores)

    return FakeTrackedObject
