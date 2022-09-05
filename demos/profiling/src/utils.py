import json

import cv2
import numpy as np
import PIL.Image
import torch
import torch2trt
import torchvision.transforms as transforms
import trt_pose.coco
import trt_pose.models
from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects

from norfair import Detection
from norfair.distances import (
    create_keypoints_voting_distance,
    create_normalized_mean_euclidean_distance,
)
from norfair.filter import (
    FilterPyKalmanFilterFactory,
    NoFilterFactory,
    OptimizedKalmanFilterFactory,
)

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device("cuda")

model_path = {
    "resnet": "/trt_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth",
    "densenet": "/trt_pose/densenet121_baseline_att_256x256_B_epoch_160_trt.pth",
}

model_resolution = {"resnet": [224, 224], "densenet": [256, 256]}

DETECTION_THRESHOLD = 0.01
DISTANCE_THRESHOLD = 1 / 5
KEYPOINT_DIST_SCALE_FACTOR = 1 / 3
HIT_COUNTER_MAX = 20
INITIALIZATION_DELAY = 3
POINTWISE_HIT_COUNTER_MAX = 6


def get_preprocesser(model_width, model_height):
    def preprocess(image):
        global device
        device = torch.device("cuda")
        image = cv2.cvtColor(
            cv2.resize(image, (model_width, model_height)), cv2.COLOR_BGR2RGB
        )
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    return preprocess


def get_postprocesser(video_width, video_height):
    with open("/trt_pose/tasks/human_pose/human_pose.json", "r") as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    parse_objects = ParseObjects(topology)

    def postprocess(cmap, paf):
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)

        count = int(counts[0])

        detections = []

        for i in range(count):
            points = []
            scores = []
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])

                if k >= 0:
                    peak = peaks[0][j][k]
                    x = float(peak[1]) * video_width
                    y = float(peak[0]) * video_height

                    points.append([x, y])
                    scores.append(1)
                else:
                    points.append([0, 0])
                    scores.append(0)

            detections.append(Detection(np.array(points), scores=np.array(scores)))

        return detections

    return postprocess


def get_distance_function(distance_function, video_width=None, video_height=None):
    if distance_function == "keypoints_vote":
        return rescaled_keypoints_vote
    elif distance_function == "euclidean":
        return create_normalized_mean_euclidean_distance(video_width, video_height)
    else:
        raise ValueError(
            "'distance_function' argument should be either 'keypoints_vote' or 'euclidean'"
        )


def get_model(
    model_weights, model_height, model_width, optimize_model=None, pose_decriptor=None
):

    if optimize_model is None:
        optimize_model = not model_weights.endswith("_trt.pth")
    if optimize_model:
        weights_file_name = model_weights[:-4]  # remove .pth

        if pose_decriptor is None:
            # First, let's load the JSON file which describes the human pose task.
            with open("/trt_pose/tasks/human_pose/human_pose.json", "r") as f:
                pose_decriptor = json.load(f)

        # Next, we'll load our model.
        num_parts = len(pose_decriptor["keypoints"])
        num_links = len(pose_decriptor["skeleton"])

        if "resnet18" in model_weights:
            model = (
                trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links)
                .cuda()
                .eval()
            )
        elif "densenet121" in model_weights:
            model = (
                trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links)
                .cuda()
                .eval()
            )
        else:
            raise ValueError("Model should be 'resnet18' or 'densenet121'")

        # Next, let's load the model weights.
        model.load_state_dict(torch.load(model_weights))

        # In order to optimize with TensorRT using the python library *torch2trt* we'll also need to create some example data.
        # The dimensions of this data should match the dimensions that the network was trained with.
        data = torch.zeros((1, 3, model_height, model_width)).cuda()

        # Next, we'll use torch2trt.  We'll enable fp16_mode to allow optimizations to use reduced half precision.
        model_trt = torch2trt.torch2trt(
            model, [data], fp16_mode=True, max_workspace_size=1 << 25
        )

        # The optimized model may be saved so that we do not need to perform optimization again, we can just load the model.
        optimized_model_path = f"{weights_file_name}_trt.pth"
        torch.save(model_trt.state_dict(), optimized_model_path)
    else:
        optimized_model_path = model_weights

    # We could then load the optimized model using *torch2trt* as follows.
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(optimized_model_path))

    return model_trt


def rescaled_keypoints_vote(detected_pose, tracked_pose):
    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)

    match_num = np.count_nonzero(
        (
            distances
            < KEYPOINT_DIST_SCALE_FACTOR
            * np.linalg.norm(tracked_pose.estimate[11] - tracked_pose.estimate[15])
        )
        * (detected_pose.scores > DETECTION_THRESHOLD)
        * (tracked_pose.last_detection.scores > DETECTION_THRESHOLD)
    )
    return 1 / (1 + match_num)


def get_filter_setup(filter_setup):
    if filter_setup == "none":
        return NoFilterFactory()
    elif filter_setup == "filterpy":
        return FilterPyKalmanFilterFactory()
    elif filter_setup == "optimized":
        return OptimizedKalmanFilterFactory()
    else:
        raise ValueError(
            "'filter_setup' argument should be either 'none', 'filterpy' or 'optimized'"
        )
