import numpy as np
import torch
import torchvision.ops.boxes as bops

MAX_DISTANCE = 10000


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def iou_pytorch(detection, tracked_object):
    # Slower but simplier version of iou

    detection_points = np.concatenate([detection.points[0], detection.points[1]])
    tracked_object_points = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    box_a = torch.tensor([detection_points], dtype=torch.float)
    box_b = torch.tensor([tracked_object_points], dtype=torch.float)
    iou = bops.box_iou(box_a, box_b)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return np.float(1 / iou if iou else MAX_DISTANCE)


def iou(detection, tracked_object):
    # Detection points will be box A
    # Tracked objects point will be box B.

    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return 1 / iou if iou else (MAX_DISTANCE)
