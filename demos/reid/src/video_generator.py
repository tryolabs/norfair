"""
The following code used to generate videos was extracted from https://github.com/wmuron/motpy with adaptations

MIT License

Copyright (c) 2020 Wiktor Muron

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import math
import random

import cv2
import numpy as np
from utils import collision_detected, get_color

from norfair import Detection


class Actor:
    """Actor is a box moving in 2d space"""

    color_i = 0

    def __init__(
        self,
        color=None,
        max_omega: float = 0.05,
        miss_prob: float = 0.1,
        det_err_sigma: float = 1.0,
        canvas_size: int = 400,
    ):

        self.max_omega = max_omega
        self.miss_prob = miss_prob
        self.det_err_sigma = det_err_sigma
        self.canvas_size = canvas_size

        # randomize size
        self.width = random.randint(50, 120)
        self.height = random.randint(50, 120)

        # randomize motion
        self.omega_x = random.uniform(-self.max_omega, self.max_omega)
        self.omega_y = random.uniform(-self.max_omega, self.max_omega)
        self.fi_x = random.randint(-180, 180)
        self.fi_y = random.randint(-90, 90)

        # let's treat color as a kind of feature
        if color is None:
            self.color = get_color(Actor.color_i)
            Actor.color_i += 1

        self.disappear_steps = 0

    def position_at(self, step: int):
        half = self.canvas_size / 2 - 50
        x = half * math.cos(self.omega_x * step + self.fi_x) + half
        y = half * math.cos(self.omega_y * step + self.fi_y) + half
        return (x, y)

    def detections(self, step: int):
        """returns ground truth and potentially missing detection for a given actor"""
        xmin, ymin = self.position_at(step)
        box_gt = [xmin, ymin, xmin + self.width, ymin + self.height]

        # detection has some noise around the face coordinates
        box_pred = [random.gauss(0, self.det_err_sigma) + v for v in box_gt]

        # wrap boxes and features as detections
        det_gt = Detection(
            points=np.vstack(
                (
                    [box_gt[0], box_gt[1]],
                    [box_gt[2], box_gt[1]],
                    [box_gt[0], box_gt[3]],
                    [box_gt[2], box_gt[3]],
                )
            ),
            scores=np.array([1.0 for _ in box_gt]),
            embedding=self.color,
        )
        feature_pred = [random.gauss(0, 5) + v for v in self.color]

        det_pred = None
        if box_pred is not None:
            det_pred = Detection(
                points=np.vstack(
                    (
                        [box_pred[0], box_pred[1]],
                        [box_pred[2], box_pred[1]],
                        [box_pred[0], box_pred[3]],
                        [box_pred[2], box_pred[3]],
                    )
                ),
                scores=np.array([random.uniform(0.5, 1.0) for _ in box_pred]),
                embedding=feature_pred,
            )
        return det_gt, det_pred


def data_generator(
    canvas_size: int,
    num_steps: int = 1000,
    num_objects: int = 1,
    max_omega: float = 0.01,
    miss_prob: float = 0.1,
    det_err_sigma: float = 1.0,
):

    actors = [
        Actor(
            max_omega=max_omega,
            miss_prob=miss_prob,
            det_err_sigma=det_err_sigma,
            canvas_size=canvas_size,
        )
        for _ in range(num_objects)
    ]

    for step in range(num_steps):
        dets_gt, dets_pred = [], []

        for actor in actors:
            det_gt, det_pred = actor.detections(step)
            append_det = True

            for past_det in dets_gt:
                if collision_detected(det_gt, past_det):
                    append_det = False
                    break

            dets_gt.append(det_gt)
            if append_det and det_pred is not None:
                dets_pred.append(det_pred)

        dets_gt.reverse()
        dets_pred.reverse()

        yield dets_gt, dets_pred


def generate_video(
    num_steps: int = 500,
    num_objects: int = 10,
    max_omega: float = 0.03,
    miss_prob: float = 0.05,
    det_err_sigma: float = 1.5,
    output_path: str = "demo.avi",
    fps: int = 30,
    canvas_size: int = 800,
):
    def _empty_canvas(canvas_size=(canvas_size, canvas_size, 3)):
        img = np.ones(canvas_size, dtype=np.uint8) * 30
        return img

    video = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"DIVX"), fps, (canvas_size, canvas_size)
    )

    detections_gt = []
    detections_pred = []

    data_gen = data_generator(
        canvas_size,
        num_steps,
        num_objects,
        max_omega,
        miss_prob,
        det_err_sigma,
    )
    for dets_gt, dets_pred in data_gen:
        img = _empty_canvas()

        # overlay actor shapes
        for det_gt in dets_gt:
            xmin, ymin = det_gt.points[0]
            xmax, ymax = det_gt.points[-1]
            feature = det_gt.embedding
            for channel in range(3):
                img[int(ymin) : int(ymax), int(xmin) : int(xmax), channel] = feature[
                    channel
                ]

        video.write(img)
        detections_gt.append(dets_gt)
        detections_pred.append(dets_pred)
    video.release()
    return output_path, detections_gt, detections_pred
