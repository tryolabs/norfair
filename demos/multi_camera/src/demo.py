import argparse
import os
import pickle
from logging import warning

import cv2
import numpy as np
import torch

from norfair import Palette, Video, get_cutout, mean_manhattan
from norfair.camera_motion import HomographyTransformation, MotionEstimator
from norfair.common_reference_ui import set_reference
from norfair.drawing.drawer import Drawer
from norfair.multi_camera import MultiCameraClusterizer
from norfair.tracker import Detection, Tracker


def embedding_distance_detection_and_tracker(detection, tracker):
    embedding_correlations = []
    past_detections = tracker.past_detections + [tracker.last_detection]
    for past_det in past_detections:
        if past_det.embedding is None:
            continue

        embedding_correlations.append(
            1
            - cv2.compareHist(
                past_det.embedding, detection.embedding, cv2.HISTCMP_CORREL
            )
        )

    if len(embedding_correlations) > 0:
        return np.mean(embedding_correlations)
    else:
        return 2


def embedding_distance(tracker1, tracker2):
    embedding_correlations = []
    past_detections_1 = tracker1.past_detections + [tracker1.last_detection]
    past_detections_2 = tracker2.past_detections + [tracker2.last_detection]

    for past_det_1 in past_detections_1:
        if past_det_1.embedding is None:
            continue
        for past_det_2 in past_detections_2:
            if past_det_2.embedding is None:
                continue

            embedding_correlations.append(
                1
                - cv2.compareHist(
                    past_det_1.embedding, past_det_2.embedding, cv2.HISTCMP_CORREL
                )
            )

    if len(embedding_correlations) > 0:
        return np.mean(embedding_correlations)
    else:
        return 2


def get_hist(image):
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


def draw_feet(
    frame,
    clusters,
    initial_transformations,
    reference_motion_transformation=None,
    thickness=None,
    radius=None,
    text_size=None,
    text_thickness=None,
    draw_cluster_ids=True,
):
    if thickness is None:
        thickness = -1
    if radius is None:
        radius = int(round(max(max(frame.shape) * 0.005, 1)))

    for cluster in clusters:
        color = Palette.choose_color(cluster.id)
        cluster_center = 0
        cluster_is_alive = False
        for tracked_object in cluster.tracked_objects.values():
            if tracked_object.live_points.any():
                cluster_is_alive = True
                point = get_absolute_feet(
                    tracked_object, initial_transformations[tracked_object.camera_name]
                )
                if reference_motion_transformation is not None:
                    point = reference_motion_transformation.abs_to_rel(
                        np.array([point])
                    )[0]

                cluster_center += point
                frame = Drawer.circle(
                    frame,
                    tuple(point.astype(int)),
                    radius=radius,
                    color=color,
                    thickness=thickness,
                )

        if draw_cluster_ids and cluster_is_alive:
            cluster_center /= len(cluster.tracked_objects)
            frame = Drawer.text(
                frame,
                f"{cluster.id}",
                tuple(cluster_center.astype(int)),
                size=text_size,
                color=color,
                thickness=text_thickness,
            )
    return frame


def draw_cluster_bboxes(
    images,
    clusters,
    draw_cluster_ids=True,
    thickness=None,
    text_thickness=None,
    text_size=None,
):
    for cluster in clusters:
        color = Palette.choose_color(cluster.id)
        for path, tracked_object in cluster.tracked_objects.items():
            if tracked_object.live_points.any():
                frame = images[path]

                if thickness is None:
                    current_thickness = max(int(max(frame.shape) / 500), 1)
                else:
                    current_thickness = thickness

                # draw the bbox
                points = tracked_object.estimate.astype(int)
                frame = Drawer.rectangle(
                    frame,
                    tuple(points),
                    color=color,
                    thickness=current_thickness,
                )

                if draw_cluster_ids:
                    text = f"{cluster.id}"

                    # the anchor will become the bottom-left of the text,
                    # we select-top left of the bbox compensating for the thickness of the box
                    text_anchor = (
                        points[0, 0] - current_thickness // 2,
                        points[0, 1] - current_thickness // 2 - 1,
                    )

                    frame = Drawer.text(
                        frame,
                        text,
                        position=text_anchor,
                        size=text_size,
                        color=color,
                        thickness=text_thickness,
                    )
                    images[path] = frame
    return images


def get_mask_from_boxes(frame, boxes):
    # create a mask of ones
    mask = np.ones(frame.shape[:2], frame.dtype)
    # set to 0 all detections
    for b in boxes:
        i = b.astype(int)
        mask[i[0, 1] : i[1, 1], i[0, 0] : i[1, 0]] = 0
    return mask


def yolo_detections_to_norfair_detections(yolo_detections, frame):
    norfair_detections = []
    boxes = []
    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        detection_as_xyxy = detection_as_xyxy.cpu().numpy()
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        boxes.append(bbox)
        points = bbox.copy()
        scores = np.array([detection_as_xyxy[4], detection_as_xyxy[4]])

        # get embedding
        t_shirt_bbox_x = 0.7 * bbox[:, 0] + 0.3 * bbox[::-1, 0]
        top = np.min(bbox[:, 1])
        bottom = np.max(bbox[:, 1])
        t_shirt_bbox_y = np.array([top * 0.7 + bottom * 0.3, top * 0.4 + bottom * 0.6])

        bbox[:, 0] = t_shirt_bbox_x
        bbox[:, 1] = t_shirt_bbox_y

        cut = get_cutout(bbox, frame)
        if cut.shape[0] > 0 and cut.shape[1] > 0:
            embedding = get_hist(cut)
        else:
            embedding = None

        norfair_detections.append(
            Detection(
                points=points,
                scores=scores,
                label=detection_as_xyxy[-1].item(),
                embedding=embedding,
            )
        )

    return norfair_detections, boxes


def get_absolute_feet(tracked_object, initial_transformation):
    bbox_relative = tracked_object.estimate
    feet = np.array([[bbox_relative[:, 0].mean(), bbox_relative[:, 1].max()]])
    try:
        return initial_transformation.rel_to_abs(tracked_object.rel_to_abs(feet))[0]
    except AttributeError:
        return initial_transformation.rel_to_abs(feet)[0]


def run():
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--reference", type=str, default=None, help="Image or Video for reference"
    )
    parser.add_argument(
        "--use-motion-estimator-reference",
        action="store_true",
        help="If your reference is a video where the camera might move, you should use a motion estimator.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[360, 288],
        help="Output resolution for each subblock",
    )
    parser.add_argument(
        "--ui-width",
        type=int,
        default=None,
        help="Image width in the UI",
    )
    parser.add_argument(
        "--ui-height",
        type=int,
        default=None,
        help="Image height in the UI",
    )
    parser.add_argument(
        "--use-motion-estimator-footage",
        action="store_true",
        help="If your footage are a video where the camera might move, you should use a motion estimator. This argument will apply the motion estimator for all your videos indifferently.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n",
        help="YOLO model to use, possible values are yolov5n, yolov5s, yolov5m, yolov5l, yolov5x",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold of detections",
        default=0.2,
    )
    parser.add_argument(
        "--foot-distance-threshold",
        type=float,
        default=0.2,
        help="Maximum spatial distance that two tracked objects of different videos can have in order to match",
    )
    parser.add_argument(
        "--reid-embedding-correlation-threshold",
        type=float,
        default=0.5,
        help="Threshold for embedding match during a reid phase after object has been lost. (The 1-correlation distance we use is bounded in [0, 2])",
    )
    parser.add_argument(
        "--embedding-correlation-threshold",
        type=float,
        default=1,
        help="Threshold for embedding match.  (The 1-correlation distance we use is bounded in [0, 2]",
    )
    parser.add_argument(
        "--max-votes-grow",
        type=int,
        default=3,
        help="Amount of votes we need before increasing the size of a cluster",
    )
    parser.add_argument(
        "--max-votes-split",
        type=int,
        default=15,
        help="Amount of votes we need before decreasing the size of a cluster",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=3,
        help="How long into the past should we consider past clusters",
    )
    parser.add_argument(
        "--joined-distance",
        type=str,
        default="mean",
        help="How a distance between clusters is done when associating trackers from different videos. Either 'mean' or 'max'",
    )
    parser.add_argument(
        "--keep-id-criteria",
        type=str,
        default="hit_counter",
        help="When splitting a cluster, we have to choose which subcluster will keep the id of the old cluster. Either 'age' or 'hit_counter'",
    )
    parser.add_argument(
        "--initialization-delay",
        type=float,
        default=19,
        help="Min detections needed to start the tracked object",
    )
    parser.add_argument(
        "--clusterizer-initialization-delay",
        type=int,
        default=15,
        help="Minimum age of a cluster (or it's objects) to be returned",
    )
    parser.add_argument(
        "--maximum-time-since-last-update",
        type=int,
        default=1,
        help="Filter tracked objects that were not detected recently to not be considered in the distance function of the clusterizer",
    )
    parser.add_argument(
        "--hit-counter-max",
        type=int,
        default=20,
        help="Max iteration the tracked object is kept after when there are no detections",
    )
    parser.add_argument(
        "--reid-hit-counter-max",
        type=int,
        default=150,
        help="Maximum amount of frames trying to reidentify the object. (Use a value >=0)",
    )
    parser.add_argument(
        "--nms-threshold", type=float, help="Iou threshold for detector", default=0.15
    )
    parser.add_argument(
        "--image-size", type=int, help="Size of the images for detector", default=480
    )
    parser.add_argument(
        "--classes", type=int, nargs="+", default=[0], help="Classes to track"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Max points sampled to calculate camera motion",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=7,
        help="Min distance between points sampled to calculate camera motion",
    )
    parser.add_argument(
        "--no-mask-detections",
        dest="mask_detections",
        action="store_false",
        default=True,
        help="By default we don't sample regions where objects were detected when estimating camera motion. Pass this flag to disable this behavior",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Name of the output file",
    )

    args = parser.parse_args()

    model = torch.hub.load("ultralytics/yolov5", args.model)
    model.conf_threshold = 0
    model.iou_threshold = args.nms_threshold
    model.image_size = args.image_size
    model.classes = args.classes

    if args.mask_detections:

        def mask_generator(frame):
            detections = model(frame)
            detections, boxes = yolo_detections_to_norfair_detections(detections, frame)
            return get_mask_from_boxes(frame, boxes)

    else:
        mask_generator = None

    videos = {}
    trackers = {}
    initial_transformations = {}
    tracked_objects = {}
    motion_estimators = {}
    images = {}

    motion_estimator = MotionEstimator(
        max_points=args.max_points,
        min_distance=args.min_distance,
    )

    motion_estimator_reference = None
    motion_estimator_footage = None

    first_video_is_reference = args.reference is None
    if args.use_motion_estimator_footage:
        motion_estimator_footage = motion_estimator
        for path in args.files:
            motion_estimators[path] = motion_estimator

    if args.use_motion_estimator_reference or (
        args.use_motion_estimator_footage and first_video_is_reference
    ):
        motion_estimator_reference = motion_estimator

    # set the initial transformation for all the videos (common reference)
    if first_video_is_reference:
        reference_path = args.files[0]
        initial_transformations[reference_path] = HomographyTransformation(np.eye(3))
    else:
        reference_path = args.reference
    for path in args.files[first_video_is_reference:]:

        initial_transformations[path] = set_reference(
            reference_path,
            path,
            motion_estimator_footage=motion_estimator_footage,
            motion_estimator_reference=motion_estimator_reference,
            mask_generator=mask_generator,
            image_width=args.ui_width,
            image_height=args.ui_height,
        )

    # initialize the reference if it exists
    reference = {"video": None, "image": None, "motion_estimator": None}
    image_reference = None
    if not first_video_is_reference:
        # if failing to read it as an image, try to read it as a video
        image_reference = cv2.imread(args.reference)
        reference["image"] = image_reference
        if image_reference is None:
            video = Video(input_path=path)
            image_reference = next(video.__iter__())
            reference["video"] = video
            reference["motion_estimator"] = motion_estimator_reference

    # now initialize the videos and their trackers
    fps = None
    total_frames = None
    distance_functions = {}

    def get_distance_function_from_threshold(threshold):
        # compare embeddings if spatial distance is small enough
        def conditional_embedding_to_spatial(detection, tracked_object):
            if mean_manhattan(detection, tracked_object) < threshold:
                return embedding_distance_detection_and_tracker(
                    detection, tracked_object
                )
            else:
                return 2

        return conditional_embedding_to_spatial

    for path in args.files:
        extension = os.path.splitext(path)[1]
        if args.output_name is None:
            output_path = f"output_multi_camera_demo{extension}"
        else:
            output_path = args.output_name

        video = Video(input_path=path, output_path=output_path)

        # check the fps and total frames
        if fps is None:
            fps = video.output_fps
            total_frames = int(video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            current_fps = video.output_fps
            current_total_frames = int(
                video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            if current_fps != fps:
                warning(
                    f"{args.files[0]} is at {fps} FPS, but {path} is at {current_fps} FPS."
                )
            if total_frames != current_total_frames:
                warning(
                    f"{args.files[0]} has {total_frames} frames, but {path} has {current_total_frames} frames."
                )
        if image_reference is None:
            image_reference = next(video.__iter__())
            height = image_reference.shape[0]
        else:
            height = next(video.__iter__()).shape[0]

        videos[path] = video
        distance_functions[path] = get_distance_function_from_threshold(height / 10)
        trackers[path] = Tracker(
            distance_function=distance_functions[path],
            detection_threshold=args.confidence_threshold,
            distance_threshold=args.embedding_correlation_threshold,
            initialization_delay=args.initialization_delay,
            hit_counter_max=args.hit_counter_max,
            camera_name=path,
            past_detections_length=10,
            reid_distance_function=embedding_distance,
            reid_distance_threshold=args.reid_embedding_correlation_threshold,
            reid_hit_counter_max=args.reid_hit_counter_max,
        )
        tracked_objects[path] = []

    big_black_frame = np.zeros(
        tuple(
            [
                args.resolution[1]
                * ((len(args.files) + (not first_video_is_reference) + 1) // 2),
                args.resolution[0] * 2,
                3,
            ]
        ),
        dtype=np.uint8,
    )

    height_reference = image_reference.shape[0]

    def normalized_foot_distance(tracker1, tracker2):
        return (
            np.linalg.norm(
                get_absolute_feet(
                    tracker1, initial_transformations[tracker1.camera_name]
                )
                - get_absolute_feet(
                    tracker2, initial_transformations[tracker2.camera_name]
                )
            )
            / height_reference
        )

    def clusterizer_distance(tracker1, tracker2):
        # if the foot distance is small
        # then compare the embeddings
        if normalized_foot_distance(tracker1, tracker2) < args.foot_distance_threshold:
            return embedding_distance(tracker1, tracker2)
        else:
            return 2

    multicamera_clusterizer = MultiCameraClusterizer(
        clusterizer_distance,
        args.embedding_correlation_threshold,
        join_distance_by=args.joined_distance,
        max_votes_grow=args.max_votes_grow,
        max_votes_split=args.max_votes_grow,
        memory=args.memory,
        initialization_delay=args.clusterizer_initialization_delay,
        reid_hit_counter_max=args.reid_hit_counter_max,
        maximum_time_since_last_update=args.maximum_time_since_last_update,
        keep_id_criteria=args.keep_id_criteria,
    )

    while True:
        try:
            big_black_frame_copy = big_black_frame.copy()
            for path in args.files:

                frame = next(videos[path].__iter__())
                images[path] = frame

                detections = model(frame)
                detections, boxes = yolo_detections_to_norfair_detections(
                    detections,
                    frame,
                )
                if args.mask_detections:
                    mask = get_mask_from_boxes(frame, boxes)
                else:
                    mask = None

                if args.use_motion_estimator_footage:
                    coord_transformations = motion_estimators[path].update(frame, mask)
                else:
                    coord_transformations = None

                tracked_objects[path] = trackers[path].update(
                    detections=detections, coord_transformations=coord_transformations
                )

            clusters = multicamera_clusterizer.update(list(tracked_objects.values()))

            images = draw_cluster_bboxes(images, clusters)

            # fit images to single image
            for n, path in enumerate(args.files):
                row = n // 2
                column = n % 2
                frame = images[path]
                frame = cv2.resize(
                    frame, tuple(args.resolution), interpolation=cv2.INTER_AREA
                )

                height, width, channels = frame.shape

                big_black_frame_copy[
                    row * height : (row + 1) * height,
                    column * width : (column + 1) * width,
                ] = frame

            if not first_video_is_reference:
                coord_transformations = None
                if reference["video"] is not None:
                    frame = next(reference["video"].__iter__())

                    if reference["motion_estimator"] is not None:
                        if args.args.mask_detections:
                            mask = mask_generator(frame)
                        else:
                            mask = None
                        coord_transformations = reference["motion_estimator"].update(
                            frame, mask
                        )
                else:
                    frame = reference["image"].copy()

                frame = draw_feet(
                    frame, clusters, initial_transformations, coord_transformations
                )

                frame = cv2.resize(
                    frame, tuple(args.resolution), interpolation=cv2.INTER_AREA
                )

                height, width, channels = frame.shape

                row = len(args.files) // 2
                is_at_center = bool((len(args.files) + 1) % 2)

                if is_at_center:
                    x0 = args.resolution[0] // 2
                else:
                    x0 = args.resolution[0]

                big_black_frame_copy[row * height :, x0 : x0 + width] = frame

            videos[args.files[0]].write(big_black_frame_copy)
        except StopIteration:
            break


if __name__ == "__main__":
    run()
