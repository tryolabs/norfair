import cv2

from norfair import Detection


def get_color(index: int):
    colors = [
        (247, 129, 191),
        (153, 153, 153),
        (77, 175, 74),
        (228, 26, 28),
        (55, 126, 184),
        (255, 255, 255),
        (152, 78, 163),
        (86, 180, 233),
        (204, 121, 167),
        (240, 228, 66),
    ]
    return colors[index % len(colors)]


def get_hist(image):
    hist = cv2.calcHist(
        [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)],
        [0, 1],
        None,
        [128, 128],
        [0, 256, 0, 256],
    )
    return cv2.normalize(hist, hist).flatten()


def collision_detected(det_first: Detection, det_snd: Detection):

    fst_xmin, fst_ymin = det_first.points[0]
    fst_xmax, fst_ymax = det_first.points[-1]
    snd_xmin, snd_ymin = det_snd.points[0]
    snd_xmax, snd_ymax = det_snd.points[-1]

    return (
        fst_xmin < snd_xmax
        and fst_xmax > snd_xmin
        and fst_ymin < snd_ymax
        and fst_ymax > snd_ymin
    )
