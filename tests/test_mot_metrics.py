import os.path

import numpy as np
import pandas as pd

from norfair import Tracker, metrics, FilterPyKalmanFilterFactory

DATASET_PATH = "train"
MOTA_ERROR_THRESHOLD = 0.0

FRAME_SKIP_PERIOD = 1
DETECTION_THRESHOLD = 0.01
DISTANCE_THRESHOLD = 0.9
DIAGONAL_PROPORTION_THRESHOLD = 1 / 18
POINTWISE_HIT_COUNTER_MAX = 3
HIT_COUNTER_MAX = 2

def keypoints_distance(detected_pose, tracked_pose):
    norm_orders = [1, 2, np.inf]
    distances = 0
    diagonal = 0

    hor_min_pt = min(detected_pose.points[:, 0])
    hor_max_pt = max(detected_pose.points[:, 0])
    ver_min_pt = min(detected_pose.points[:, 1])
    ver_max_pt = max(detected_pose.points[:, 1])

    # Set keypoint_dist_threshold based on object size, and calculate
    # distance between detections and tracker estimations
    for p in norm_orders:
        distances += np.linalg.norm(
            detected_pose.points - tracked_pose.estimate, ord=p, axis=1
        )
        diagonal += np.linalg.norm(
            [hor_max_pt - hor_min_pt, ver_max_pt - ver_min_pt], ord=p
        )

    distances = distances / len(norm_orders)

    keypoint_dist_threshold = diagonal * DIAGONAL_PROPORTION_THRESHOLD

    match_num = np.count_nonzero(
        (distances < keypoint_dist_threshold)
        * (detected_pose.scores > DETECTION_THRESHOLD)
        * (tracked_pose.last_detection.scores > DETECTION_THRESHOLD)
    )
    return 1 / (1 + match_num)

def test_mot_metrics():
    """Tests that Norfair's MOT metrics didn't get worse

     Configurable so that it allows some margin on how much worse metrics could get before
     the test fails. Margin configured through MOTA_ERROR_THRESHOLD.

     Raises:
         If the previous metrics file its not found.
     """
    # Load previous metrics
    try:
        previous_metrics = pd.read_fwf('tests/metrics.txt')
        previous_metrics.columns = [column_name.lower() for column_name in previous_metrics.columns]
        previous_metrics = previous_metrics.set_index(previous_metrics.columns[0])
    except FileNotFoundError as e:
        raise e

    accumulator = metrics.Accumulators()
    sequences_paths = [element.path for element in os.scandir(DATASET_PATH) if element.is_dir()]
    for input_path in sequences_paths:
        # Search vertical resolution in seqinfo.ini
        seqinfo_path = os.path.join(input_path, "seqinfo.ini")
        info_file = metrics.InformationFile(file_path=seqinfo_path)

        all_detections = metrics.DetectionFileParser(
            input_path=input_path, information_file=info_file
        )

        tracker = Tracker(
            distance_function=keypoints_distance,
            distance_threshold=DISTANCE_THRESHOLD,
            detection_threshold=DETECTION_THRESHOLD,
            pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
            hit_counter_max=HIT_COUNTER_MAX,
            filter_factory=FilterPyKalmanFilterFactory(),
        )

        # Initialize accumulator for this video
        accumulator.create_accumulator(input_path=input_path, information_file=info_file)

        for frame_number, detections in enumerate(all_detections):
            if frame_number % FRAME_SKIP_PERIOD == 0:
                tracked_objects = tracker.update(
                    detections=detections, period=FRAME_SKIP_PERIOD
                )
            else:
                detections = []
                tracked_objects = tracker.update()

            accumulator.update(predictions=tracked_objects)

    accumulator.compute_metrics()
    new_metrics = accumulator.summary_dataframe
    new_metrics.columns = [column_name.lower() for column_name in new_metrics.columns]

    # Unify the scores to be able to compare them. new metrics is the percentage
    # expressed between 0 and 1, the previous metrics have the percentage as a string
    # with the % character at the end
    new_overall_mota = np.around(new_metrics.loc["OVERALL", "mota"] * 100, 1)
    previous_overall_mota = np.around(float(previous_metrics.loc["OVERALL", "mota"][:-1]), 1)

    accumulator.print_metrics()
    assert new_overall_mota >= previous_overall_mota * (1 - MOTA_ERROR_THRESHOLD), f"New overall MOTA score: {new_overall_mota} is too low, previous overall MOTA score: {previous_overall_mota}"
