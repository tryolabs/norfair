import numpy as np
cimport numpy as np

DTYPE = float
ctypedef np.float_t DTYPE_t


def iou_opt(
        np.ndarray[DTYPE_t, ndim=2] candidates,
        np.ndarray[DTYPE_t, ndim=2] objects
) -> np.ndarray:
    """
    Calculate IoU between bounding boxes. 
    Based on https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/bbox.pyx

    Parameters
    ----------
    candidates : numpy.ndarray
        (N, 4) numpy.ndarray of float containing candidates bounding boxes.
    objects : numpy.ndarray
        (K, 4) numpy.ndarray of float containing objects bounding boxes.

    Returns
    -------
    numpy.ndarray
        (N, K) numpy.ndarray of IoU between candidates and objects.
    """
    cdef unsigned int N = candidates.shape[0]
    cdef unsigned int K = objects.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ious = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t cand_area, inter_height, inter_width, obj_area, union_area
    cdef unsigned int k, n

    for k in range(K):
        obj_area = (objects[k, 2] - objects[k, 0]) * (objects[k, 3] - objects[k, 1])
        for n in range(N):
            inter_width = (min(candidates[n, 2], objects[k, 2]) -
                           max(candidates[n, 0], objects[k, 0]))
            if inter_width > 0:
                inter_height = (min(candidates[n, 3], objects[k, 3]) -
                                max(candidates[n, 1], objects[k, 1]))
                if inter_height > 0:
                    cand_area = (candidates[n, 2] - candidates[n, 0]) * (
                                candidates[n, 3] - candidates[n, 1])
                    union_area = float(cand_area + obj_area -
                                       inter_width * inter_height)
                    ious[n, k] = inter_width * inter_height / union_area
    return 1 - ious
