import cv2
import numpy as np

from norfair import Palette

CONNECTED_INDEXES = [1, 2, 4, 3, 1, 5, 7, 3, 7, 8, 6, 5, 6, 2, 4, 8]


class PixelCoordinatesProjecter:
    def __init__(
        self,
        image_size,
        focal_length_ndc=None,
        principal_point_ndc=None,
        focal_length_pixel=None,
        principal_point_pixel=None,
    ):
        """
        How to use?
        - Create a PixelCoordinatesProjecter instance providing the image_size (You may provide focal length and principal points in NDC or pixel coordinates)
        - Receive points in a 2D array in either NDC or eye coordinates (n, 3), and transform them to pixel coordinates with ndc_2_pixel or eye_2_pixel (n, 2)

        Further details:
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        https://google.github.io/mediapipe/solutions/objectron.html#coordinate-systems
        """

        self.image_size = np.array(image_size)

        ndc_params_is_none = all(
            elem is None for elem in (focal_length_ndc, principal_point_ndc)
        )
        pixel_params_is_none = all(
            elem is None for elem in (focal_length_pixel, principal_point_pixel)
        )

        if ndc_params_is_none and not pixel_params_is_none:
            self.focal_length_pixel = focal_length_pixel
            self.principal_point_pixel = principal_point_pixel
            self.compute_ndc_parameters_from_pixel_parameters()

        elif pixel_params_is_none and not ndc_params_is_none:
            self.focal_length_ndc = focal_length_ndc
            self.principal_point_ndc = principal_point_ndc
            self.compute_pixel_parameters_from_ndc_parameters()

        elif ndc_params_is_none and pixel_params_is_none:
            self.focal_length_ndc = np.array((1.0, 1.0))
            self.principal_point_ndc = np.array((0.0, 0.0))
            self.compute_pixel_parameters_from_ndc_parameters()
        else:
            raise ValueError(
                "You cannot provide parameters in both NDC and pixel coordinates. Choose one."
            )

    def compute_pixel_parameters_from_ndc_parameters(self):
        """
        FOCAL LENGTH
        fx_pixel = fx_ndc * image_width / 2
        fy_pixel = fy_ndc * image_height / 2

        PRINCIPAL POINT
        px_pixel = (1-px_ndc)*image_width/2
        py_pixel = (1-py_ndc)*image_height/2
        """
        self.focal_length_pixel = self.focal_length_ndc * self.image_size / 2
        self.principal_point_pixel = (
            (1 - self.principal_point_ndc) * self.image_size / 2
        )

    def compute_ndc_parameters_from_pixel_parameters(self):
        """
        FOCAL LENGTH
        fx_ndc = fx_pixel * 2.0 / image_width
        fy_ndc = fy_pixel * 2.0 / image_height

        PRINCIPAL POINT
        px_ndc = -px_pixel * 2.0 / image_width  + 1.0
        py_ndc = -py_pixel * 2.0 / image_height + 1.0
        """
        self.focal_length_ndc = 2 * self.focal_length_pixel / self.image_size
        self.principal_point_ndc = (
            1.0 - 2.0 * self.principal_point_pixel / self.image_size
        )

    def eye_2_ndc(self, points_eye):
        """
        Takes points in eye coordinates (X, Y, Z) and convert them to NDC coordinates
        x_ndc = -fx_ndc * X / Z + px_ndc
        y_ndc = -fy_ndc * Y / Z + py_ndc
        z_ndc = 1 / Z
        """

        points_ndc = (np.ones(points_eye.shape).T / points_eye[:, 2]).T
        points_ndc[:, :2] *= -points_eye[:, :2] * self.focal_length_ndc
        points_ndc[:, :2] += self.principal_point_ndc
        return points_ndc

    def eye_2_pixel(self, points_eye):
        """
        Takes points in eye coordinates and project them to the pixel 2d coordinates
        x_pixel = -fx_pixel * X / Z + px_pixel
        y_pixel =  fy_pixel * Y / Z + py_pixel
        """

        points_pixel = (points_eye[:, :2].T / points_eye[:, 2]).T
        points_pixel *= self.focal_length_pixel * np.array([-1, 1])
        points_pixel += self.principal_point_pixel

        return points_pixel

    def ndc_2_pixel(self, points_ndc):
        """
        Takes points in NDC coordinates and project them to the pixel 2d coordinates
        x_pixel = (1 + x_ndc) / 2.0 * image_width
        y_pixel = (1 - y_ndc) / 2.0 * image_height
        """

        points_pixel = ((points_ndc[:, :2] * np.array([1, -1])) + 1) / 2
        points_pixel *= self.image_size
        return points_pixel


def draw_3d_tracked_boxes(
    frame,
    objects,
    projecter,
    border_colors=None,
    border_width=None,
    id_size=None,
    id_thickness=None,
    draw_box=True,
    draw_only_alive=True,
):
    frame_scale = frame.shape[0] / 100
    if border_width is None:
        border_width = int(frame_scale * 0.5)
    if id_size is None:
        id_size = frame_scale / 10
    if id_thickness is None:
        id_thickness = int(frame_scale / 5)
    if isinstance(border_colors, tuple):
        border_colors = [border_colors]

    for n, obj in enumerate(objects):
        if draw_only_alive and not obj.live_points.any():
            continue
        if border_colors is None:
            color = Palette.choose_color(obj.id)
        else:
            color = border_colors[n % len(border_colors)]

        # project the points to the pixel space
        pixel_points = projecter(obj.estimate)

        # sort the points to draw the lines
        sorted_points = np.array([pixel_points[index] for index in CONNECTED_INDEXES])

        # draw box
        if draw_box:
            frame = cv2.polylines(
                frame,
                [sorted_points.astype(np.int32).reshape((-1, 1, 2))],
                isClosed=False,
                color=color,
                thickness=border_width,
            )

        if id_size > 0:
            id_draw_position = pixel_points[0].astype(int)
            cv2.putText(
                frame,
                str(obj.id),
                tuple(id_draw_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                id_size,
                color,
                id_thickness,
                cv2.LINE_AA,
            )
    return frame


def scaled_euclidean(detection: "Detection", tracked_object: "TrackedObject") -> float:
    """
    Average euclidean distance between the points in detection and estimates in tracked_object, rescaled by the object diagonal
    See `np.linalg.norm`.
    """
    obj_estimate = tracked_object.estimate
    diagonal = np.linalg.norm(obj_estimate[1] - obj_estimate[8])
    return np.linalg.norm(detection.points - obj_estimate, axis=1).mean() / diagonal
