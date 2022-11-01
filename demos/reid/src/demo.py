import random

random.seed(1337)

import cv2
import numpy as np
import typer
from utils import get_hist
from video_generator import generate_video

from norfair import Tracker, Video, draw_points, draw_tracked_objects, get_cutout
from norfair.filter import OptimizedKalmanFilterFactory


def embedding_distance(matched_not_init_trackers, unmatched_trackers):
    snd_embedding = unmatched_trackers.last_detection.embedding

    if snd_embedding is None:
        for detection in reversed(unmatched_trackers.past_detections):
            if detection.embedding is not None:
                snd_embedding = detection.embedding
                break
        else:
            return 1

    for detection_fst in matched_not_init_trackers.past_detections:
        if detection_fst.embedding is None:
            continue

        distance = 1 - cv2.compareHist(
            snd_embedding, detection_fst.embedding, cv2.HISTCMP_CORREL
        )
        if distance < 0.5:
            return distance
    return 1


def main(
    output_path: str = "./out.mp4",
    skip_period: int = 1,
    disable_reid: bool = False,
    border_size: int = 10,
):
    video_path, _, video_predictions = generate_video()

    if disable_reid:
        tracker = Tracker(
            initialization_delay=1,
            distance_function="euclidean",
            hit_counter_max=10,
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_threshold=50,
            past_detections_length=5,
        )
    else:
        tracker = Tracker(
            initialization_delay=1,
            distance_function="euclidean",
            hit_counter_max=10,
            filter_factory=OptimizedKalmanFilterFactory(),
            distance_threshold=50,
            past_detections_length=5,
            reid_distance_function=embedding_distance,
            reid_distance_threshold=0.5,
            reid_hit_counter_max=500,
        )

    video = Video(input_path=video_path, output_path=output_path)
    for i, cv2_frame in enumerate(video):
        if i % skip_period == 0:
            detections = video_predictions[i]
            frame = cv2_frame.copy()
            for detection in detections:
                cut = get_cutout(detection.points, frame)
                if cut.shape[0] > 0 and cut.shape[1] > 0:
                    detection.embedding = get_hist(cut)
                else:
                    detection.embedding = None

            tracked_objects = tracker.update(detections=detections, period=skip_period)
        else:
            tracked_objects = tracker.update()
        draw_points(cv2_frame, detections)
        draw_tracked_objects(cv2_frame, tracked_objects)
        frame_with_border = np.ones(
            shape=(
                cv2_frame.shape[0] + 2 * border_size,
                cv2_frame.shape[1] + 2 * border_size,
                cv2_frame.shape[2],
            ),
            dtype=cv2_frame.dtype,
        )
        frame_with_border *= 254
        frame_with_border[
            border_size:-border_size, border_size:-border_size
        ] = cv2_frame
        video.write(frame_with_border)


if __name__ == "__main__":
    typer.run(main)
