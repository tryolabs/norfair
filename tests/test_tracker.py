from ipaddress import ip_address

import numpy as np
import pytest

from norfair import (
    Detection,
    FilterPyKalmanFilterFactory,
    OptimizedKalmanFilterFactory,
    Tracker,
)


@pytest.mark.parametrize("filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()])
def test_simple(filter_factory):
    #
    # tests a simple static detection
    #
    tracker = Tracker("frobenius", initialization_delay=3, distance_threshold=100, hit_counter_max=3, filter_factory=filter_factory)

    # simulate 4 consecutive frames receiving the same detection
    detections = [Detection(points=np.array([[1, 1]]))]
    assert len(tracker.update(detections)) == 0
    assert len(tracker.update(detections)) == 0
    assert len(tracker.update(detections)) == 0
    tracked_objects = tracker.update(detections)
    assert len(tracked_objects) == 1
    obj = tracked_objects[0]
    np.testing.assert_almost_equal(tracked_objects[0].estimate, np.array([[1, 1]]))
    assert obj.age == 3
    assert obj.hit_counter == 4 # TODO: check this result

    # simulate that the object dissapears
    tracked_objects = tracker.update()
    assert len(tracked_objects) == 1
    obj = tracked_objects[0]
    assert obj.age == 4
    assert obj.hit_counter == 3

    tracked_objects = tracker.update()
    assert len(tracked_objects) == 1
    obj = tracked_objects[0]
    assert obj.age == 5
    assert obj.hit_counter == 2


    tracked_objects = tracker.update()
    assert len(tracked_objects) == 1
    obj = tracked_objects[0]
    assert obj.age == 6
    assert obj.hit_counter == 1


    tracked_objects = tracker.update()
    assert len(tracked_objects) == 1
    obj = tracked_objects[0]
    assert obj.age == 7
    assert obj.hit_counter == 0

    tracked_objects = tracker.update()
    assert len(tracked_objects) == 1
    obj = tracked_objects[0]
    assert obj.age == 8
    assert obj.hit_counter == -1 # TODO: check this result

    assert len(tracker.update()) == 0


@pytest.mark.parametrize("filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()])
def test_moving(filter_factory):
    #
    # Test a simple case of a moving object
    #
    tracker = Tracker("frobenius", initialization_delay=3, distance_threshold=100, filter_factory=filter_factory)

    assert len(tracker.update([Detection(points=np.array([[1, 1]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 2]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 3]]))])) == 0
    tracked_objects = tracker.update([Detection(points=np.array([[1, 4]]))])
    assert len(tracked_objects) == 1

    # check that the estimated position makes sense
    assert tracked_objects[0].estimate[0][0] == 1
    assert 3 < tracked_objects[0].estimate[0][1] <= 4



@pytest.mark.parametrize("filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()])
def test_distance_t(filter_factory):
    #
    # Test a moving object with a small distance threshold
    #
    tracker = Tracker("frobenius", initialization_delay=1, distance_threshold=1, filter_factory=filter_factory)

    # should not match because the distance is too large
    assert len(tracker.update([Detection(points=np.array([[1, 1]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 2]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 3]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 4]]))])) == 0
    # a closer point should match
    tracked_objects = tracker.update([Detection(points=np.array([[1, 4.1]]))])
    assert len(tracked_objects) == 1

    # check that the estimated position makes sense
    assert tracked_objects[0].estimate[0][0] == 1
    assert 4 < tracked_objects[0].estimate[0][1] <= 4.5


# TODO tests list:
#   - detections with different labels
#   - partial matches where some points are missing
#   - pointwise_hit_counter_max
#   - past detections
