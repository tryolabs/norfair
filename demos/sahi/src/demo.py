from typing import List

import numpy as np
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import PredictionResult
from utils import create_arg_parser, obtain_detection_model

from norfair import Detection, Tracker, Video, draw_boxes, draw_tracked_boxes
from norfair.filter import OptimizedKalmanFilterFactory


def get_detections(object_prediction_list: PredictionResult) -> List[Detection]:
    detections = []
    for prediction in object_prediction_list:
        bbox = prediction.bbox

        detection_as_xyxy = bbox.to_voc_bbox()
        bbox = np.array(
            [
                [detection_as_xyxy[0], detection_as_xyxy[1]],
                [detection_as_xyxy[2], detection_as_xyxy[3]],
            ]
        )
        detections.append(
            Detection(
                points=bbox,
                scores=np.array([prediction.score.value for _ in bbox]),
                label=prediction.category.id,
            )
        )
    return detections


def main(
    video_path: str,
    output_path: str,
    distance_threshold: float,
    skip_period: int,
    initialization_delay: int,
    hit_counter_max: int,
    enable_sahi: bool,
    slice_size: int,
    overlap_ratio: float,
    model_confidence_threshold: float,
):
    detection_model = obtain_detection_model(model_confidence_threshold)

    tracker = Tracker(
        initialization_delay=initialization_delay,
        distance_function="iou",
        hit_counter_max=hit_counter_max,
        filter_factory=OptimizedKalmanFilterFactory(),
        distance_threshold=distance_threshold,
    )

    video = Video(input_path=video_path, output_path=output_path)

    for i, frame in enumerate(video):
        if i % skip_period == 0:
            if enable_sahi:
                result = get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                )
            else:
                result = get_prediction(frame, detection_model)

            detections = get_detections(result.object_prediction_list)
            tracked_objects = tracker.update(detections=detections, period=skip_period)
        else:
            tracked_objects = tracker.update()

        draw_boxes(frame, detections)
        draw_tracked_boxes(frame, tracked_objects)
        video.write(frame)


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    main(
        video_path=args.file,
        output_path=args.output_path,
        distance_threshold=args.distance_threshold,
        skip_period=args.skip_period,
        initialization_delay=args.initialization_delay,
        hit_counter_max=args.hit_counter_max,
        enable_sahi=args.enable_sahi,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio,
        model_confidence_threshold=args.model_confidence_threshold,
    )
