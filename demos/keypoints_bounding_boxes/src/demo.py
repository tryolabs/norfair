import argparse
import sys
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

import norfair
from norfair import Detection, Tracker, Video
from norfair.drawing import Color

# Import openpose
openpose_install_path = (
    "/openpose"  # Insert the path to your openpose instalation folder here
)
try:
    sys.path.append(openpose_install_path + "/build/python")
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
    )
    raise e

# Define constants
DETECTION_THRESHOLD = 0.6
DISTANCE_THRESHOLD = 0.8
HIT_COUNTER_MAX = 45
INITIALIZATION_DELAY = 4
POINTWISE_HIT_COUNTER_MAX = 10

############### OPENPOSE ##################

# Wrapper implementation for OpenPose detector
class OpenposeDetector:
    def __init__(self, num_gpu_start=None):
        # Set OpenPose flags
        config = {}
        config["model_folder"] = openpose_install_path + "/models/"
        config["model_pose"] = "BODY_25"
        config["logging_level"] = 3
        config["output_resolution"] = "-1x-1"
        config["net_resolution"] = "-1x768"
        config["num_gpu"] = 1
        config["alpha_pose"] = 0.6
        config["render_threshold"] = 0.05
        config["scale_number"] = 1
        config["scale_gap"] = 0.3
        config["disable_blending"] = False

        # If GPU version is built, and multiple GPUs are available,
        # you can change the ID using the num_gpu_start parameter
        if num_gpu_start is not None:
            config["num_gpu_start"] = num_gpu_start

        # Starting OpenPose
        self.detector = op.WrapperPython()
        self.detector.configure(config)
        self.detector.start()

    def __call__(self, image):
        return self.detector.emplaceAndPop(image)


########################################


############### YOLO ###################


class YOLO:
    def __init__(self, model_name: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # load model
        self.model = torch.hub.load("ultralytics/yolov5", model_name)

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    detections_as_xyxy = yolo_detections.xyxy[0]
    for detection_as_xyxy in detections_as_xyxy:
        bbox = np.array(
            [
                [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
            ]
        )
        scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
        label = int(detection_as_xyxy[5].item())
        norfair_detections.append(Detection(points=bbox, scores=scores, label=label))

    return norfair_detections


########################################


# Distance function
def keypoints_distance(detected_pose, tracked_pose):
    # Use different distances for bounding boxes and keypoints
    if detected_pose.label != 0:
        detection_centroid = np.sum(detected_pose.points, axis=0) / len(
            detected_pose.points
        )
        tracked_centroid = np.sum(tracked_pose.estimate, axis=0) / len(
            detected_pose.points
        )
        distances = np.linalg.norm(detection_centroid - tracked_centroid, axis=0)
        return distances / (KEYPOINT_DIST_THRESHOLD + distances)

    else:
        distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
        match_num = np.count_nonzero(
            (distances < KEYPOINT_DIST_THRESHOLD)
            * (detected_pose.scores > DETECTION_THRESHOLD)
            * (tracked_pose.last_detection.scores > DETECTION_THRESHOLD)
        )
        return 1 / (1 + match_num)


if __name__ == "__main__":

    # CLI configuration
    parser = argparse.ArgumentParser(description="Track objects in a video.")
    parser.add_argument("files", type=str, nargs="+", help="Video files to process")
    parser.add_argument(
        "--model-name", type=str, default="yolov5m6", help="YOLOv5 model name"
    )
    parser.add_argument(
        "--img-size", type=int, default="720", help="YOLOv5 inference size (pixels)"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default="0.25",
        help="YOLOv5 object confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default="0.45",
        help="YOLOv5 IOU threshold for NMS",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Inference device: 'cpu' or 'cuda'"
    )
    args = parser.parse_args()

    # Process Videos
    detector = OpenposeDetector()
    datum = op.Datum()
    model = YOLO(args.model_name, device=args.device)

    for input_path in args.files:
        print(f"Video: {input_path}")
        video = Video(input_path=input_path)
        tracker = Tracker(
            distance_function=keypoints_distance,
            distance_threshold=DISTANCE_THRESHOLD,
            detection_threshold=DETECTION_THRESHOLD,
            hit_counter_max=HIT_COUNTER_MAX,
            initialization_delay=INITIALIZATION_DELAY,
            pointwise_hit_counter_max=POINTWISE_HIT_COUNTER_MAX,
        )
        KEYPOINT_DIST_THRESHOLD = video.input_height / 40

        for frame in video:
            datum.cvInputData = frame
            detector(op.VectorDatum([datum]))
            detected_poses = datum.poseKeypoints

            if detected_poses is not None:
                openpose_detections = (
                    []
                    if not detected_poses.any()
                    else [
                        Detection(p, scores=s, label=-1)
                        for (p, s) in zip(
                            detected_poses[:, :, :2], detected_poses[:, :, 2]
                        )
                    ]
                )
            else:
                openpose_detections = []

            yolo_out = model(
                frame,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                image_size=args.img_size,
                classes=args.classes,
            )
            yolo_detections = yolo_detections_to_norfair_detections(yolo_out)
            detections = openpose_detections + yolo_detections

            tracked_objects = tracker.update(detections=detections)

            norfair.draw_tracked_objects(
                frame,
                [person for person in tracked_objects if person.label == -1],
                color=Color.green,
            )
            norfair.draw_tracked_boxes(
                frame,
                [obj for obj in tracked_objects if obj.label > 0],
                color_by_label=True,
            )

            video.write(frame)
