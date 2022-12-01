from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .drawer import Drawable


def _centroid(tracked_points: np.ndarray) -> Tuple[int, int]:
    num_points = tracked_points.shape[0]
    sum_x = np.sum(tracked_points[:, 0])
    sum_y = np.sum(tracked_points[:, 1])
    return int(sum_x / num_points), int(sum_y / num_points)


def _build_text(drawable: "Drawable", draw_labels, draw_ids, draw_scores):
    text = ""
    if draw_labels and drawable.label is not None:
        text = str(drawable.label)
    if draw_ids and drawable.id is not None:
        if len(text) > 0:
            text += "-"
        text += str(drawable.id)
    if draw_scores and drawable.scores is not None:
        if len(text) > 0:
            text += "-"
        text += str(np.round(np.mean(drawable.scores), 4))
    return text
