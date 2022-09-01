from ipaddress import ip_address

import numpy as np
import pytest

from norfair import (
    Detection,
    FilterPyKalmanFilterFactory,
    OptimizedKalmanFilterFactory,
    Tracker,
)


def test_params():
    #
    # test some invalid initializations
    #
    with pytest.raises(ValueError):
        Tracker("frobenius", distance_threshold=10, initialization_delay=-1)
    with pytest.raises(ValueError):
        Tracker(
            "frobenius",
            distance_threshold=10,
            initialization_delay=1,
            hit_counter_max=0,
        )
    with pytest.raises(ValueError):
        Tracker(
            "frobenius",
            distance_threshold=10,
            initialization_delay=1,
            hit_counter_max=1,
        )
    with pytest.raises(ValueError):
        Tracker(
            "_bad_distance",
            distance_threshold=10,
            initialization_delay=1,
            hit_counter_max=1,
        )


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_simple(filter_factory):
    for delay in [0, 1, 3]:
        for counter_max in [delay + 1, delay + 3]:
            #
            # tests a simple static detection
            #
            tracker = Tracker(
                "frobenius",
                initialization_delay=delay,
                distance_threshold=100,
                hit_counter_max=counter_max,
                filter_factory=filter_factory,
            )

            detections = [Detection(points=np.array([[1, 1]]))]

            # test the delay
            for age in range(delay):
                assert len(tracker.update(detections)) == 0

            # build up hit_counter from delay+1 to counter_max
            for age in range(delay, counter_max):
                tracked_objects = tracker.update(detections)
                assert len(tracked_objects) == 1
                obj = tracked_objects[0]
                np.testing.assert_almost_equal(
                    tracked_objects[0].estimate, np.array([[1, 1]])
                )
                assert obj.age == age
                assert obj.hit_counter == age + 1

            # check that counter is capped at counter_max
            for age in range(counter_max, counter_max + 3):
                tracked_objects = tracker.update(detections)
                assert len(tracked_objects) == 1
                obj = tracked_objects[0]
                np.testing.assert_almost_equal(
                    tracked_objects[0].estimate, np.array([[1, 1]])
                )
                assert obj.age == age
                assert obj.hit_counter == counter_max

            # check that counter goes down to 0 wen no detections
            for counter in range(counter_max - 1, -1, -1):
                age += 1
                tracked_objects = tracker.update()
                assert len(tracked_objects) == 1
                obj = tracked_objects[0]
                np.testing.assert_almost_equal(
                    tracked_objects[0].estimate, np.array([[1, 1]])
                )
                assert obj.age == age
                assert obj.hit_counter == counter

            # check that object dissapears in the next frame
            assert len(tracker.update()) == 0


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_moving(filter_factory):
    #
    # Test a simple case of a moving object
    #
    tracker = Tracker(
        "frobenius",
        initialization_delay=3,
        distance_threshold=100,
        filter_factory=filter_factory,
    )

    assert len(tracker.update([Detection(points=np.array([[1, 1]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 2]]))])) == 0
    assert len(tracker.update([Detection(points=np.array([[1, 3]]))])) == 0
    tracked_objects = tracker.update([Detection(points=np.array([[1, 4]]))])
    assert len(tracked_objects) == 1

    # check that the estimated position makes sense
    assert tracked_objects[0].estimate[0][0] == 1
    assert 3 < tracked_objects[0].estimate[0][1] <= 4


@pytest.mark.parametrize(
    "filter_factory", [FilterPyKalmanFilterFactory(), OptimizedKalmanFilterFactory()]
)
def test_distance_t(filter_factory):
    #
    # Test a moving object with a small distance threshold
    #
    tracker = Tracker(
        "frobenius",
        initialization_delay=1,
        distance_threshold=1,
        filter_factory=filter_factory,
    )

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
