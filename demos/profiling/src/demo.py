import argparse
import os
import sys
import time

import norfair
from norfair import Tracker, Video

sys.path.append("/demo/src")

from utils import (
    DETECTION_THRESHOLD,
    DISTANCE_THRESHOLD,
    HIT_COUNTER_MAX,
    INITIALIZATION_DELAY,
    POINTWISE_HIT_COUNTER_MAX,
    get_distance_function,
    get_filter_setup,
    get_model,
    get_postprocesser,
    get_preprocesser,
    model_path,
    model_resolution,
)


def process_video(
    video,
    detector,
    tracker,
    preprocesser,
    postprocesser,
    frame_skip_period=1,
    create_video=False,
):

    tracker_time_mean = 0
    detector_time_mean = 0

    start_time = time.time()

    total_frames = 0
    for frame_number, frame in enumerate(video):
        total_frames += 1

        if frame_number % frame_skip_period == 0:

            start_frame_time = time.time()
            data = preprocesser(frame)
            cmap, paf = detector(data)
            detections = postprocesser(cmap, paf)
            detector_time = time.time()

            tracked_objects = tracker.update(
                detections=detections, period=frame_skip_period
            )
            tracker_time = time.time()

            tracker_time_mean += tracker_time - detector_time
            detector_time_mean += detector_time - start_frame_time
        else:
            start_frame_time = time.time()
            tracked_objects = tracker.update()
            tracker_time = time.time()
            tracker_time_mean += tracker_time - start_frame_time

        if create_video:
            norfair.draw_tracked_objects(frame, tracked_objects)
            video.write(frame)

    total_time = time.time() - start_time
    fps = total_frames / total_time
    tracker_time_mean /= total_frames
    detector_time_mean /= total_frames
    return fps, tracker_time_mean, detector_time_mean


if __name__ == "__main__":

    # Flags
    parser = argparse.ArgumentParser(
        description="profile various trackers and distance functions."
    )
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--skip-frame", dest="skip_frame", type=int, default=1, help="Frame skip period"
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        default="./",
        help="Output path",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="densenet",
        help="Model (should be 'densenet' or 'resnet')",
    )
    parser.add_argument(
        "--filter-setup",
        dest="filter_setup",
        default="filterpy",
        help="Filter setup to use ('none', 'filterpy' or 'optimized')",
    )
    parser.add_argument(
        "--distance-function",
        dest="distance_function",
        default="keypoints_vote",
        help="Distance function to use ('keypoints_vote' or 'euclidean')",
    )
    parser.add_argument(
        "--use-all",
        dest="use_all",
        help="Use all filter setups and distance functions",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use-all-filters",
        dest="use_all_filters",
        help="Use all filter setups",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--use-all-distances",
        dest="use_all_distances",
        help="Use all distance functions",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--make-video",
        dest="make_video",
        help="Create output video files",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    model_weights_path = model_path[args.model]
    if not os.path.exists(model_weights_path):
        model_weights_path = model_weights_path.replace("_trt.pth", ".pth", 1)

    model_height, model_width = model_resolution[args.model]

    if args.use_all:
        args.use_all_filters = True
        args.use_all_distances = True

    if args.use_all_filters:
        filter_setups = ["none", "filterpy", "optimized"]
    elif args.filter_setup in ["none", "filterpy", "optimized"]:
        filter_setups = [args.filter_setup]
    else:
        raise ValueError(
            "'filter_setup' argument should be either 'none', 'filterpy' or 'optimized'"
        )

    if args.use_all_distances:
        distance_functions = ["keypoints_vote", "euclidean"]
    elif args.distance_function in ["keypoints_vote", "euclidean"]:
        distance_functions = [args.distance_function]
    else:
        raise ValueError(
            "'distance_function' argument should be either 'keypoints_vote' or 'euclidean'"
        )

    profiling = {}
    for input_path in args.files:
        profiling[input_path] = {}
        for fs in filter_setups:
            profiling[input_path][fs] = {}
            for df in distance_functions:
                profiling[input_path][fs][df] = {
                    input_path: {
                        fs: {
                            df: {
                                "fps": None,
                                "tracker time": None,
                                "detector time": None,
                            }
                        }
                    }
                }

    # Process Videos
    for fs in filter_setups:
        filter_setup = get_filter_setup(fs)
        for df in distance_functions:
            for input_path in args.files:
                print("")
                print("Filter: ", fs)
                print("Distance: ", df)
                print("Video: ", input_path)
                video_name = os.path.splitext(os.path.basename(input_path))[0]
                video_name = f"{video_name}_{df}_{fs}.mp4"
                output_path = os.path.join(args.output_path, video_name)
                video = Video(input_path=input_path, output_path=output_path)

                model_trt = get_model(model_weights_path, model_height, model_width)

                distance_function = get_distance_function(
                    df, video.input_width, video.input_height
                )
                tracker = Tracker(
                    distance_function=distance_function,
                    detection_threshold=DETECTION_THRESHOLD,
                    distance_threshold=DISTANCE_THRESHOLD,
                    hit_counter_max=HIT_COUNTER_MAX,
                    initialization_delay=INITIALIZATION_DELAY,
                    pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
                    filter_factory=filter_setup,
                )

                preprocess = get_preprocesser(model_width, model_height)
                postprocess = get_postprocesser(video.input_width, video.input_height)

                fps, tracker_time, detector_time = process_video(
                    video,
                    detector=model_trt,
                    tracker=tracker,
                    preprocesser=preprocess,
                    postprocesser=postprocess,
                    frame_skip_period=args.skip_frame,
                    create_video=args.make_video,
                )

                profiling[input_path][fs][df] = {
                    "fps": fps,
                    "tracker time": tracker_time,
                    "detector time": detector_time,
                }

    for input_path, prof_ip in profiling.items():
        print("")
        print(input_path)
        for filter_setup, prof_ip_fs in prof_ip.items():
            for distance_function, prof_ip_fs_df in prof_ip_fs.items():
                print(f"Filter: {filter_setup}, Distance: {distance_function}")
                print(
                    f"FPS: {prof_ip_fs_df['fps']:.3}, Tracker time: {prof_ip_fs_df['tracker time']:.2}, Detector time: {prof_ip_fs_df['detector time']:.2}"
                )
