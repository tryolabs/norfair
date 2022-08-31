import argparse

import numpy as np
import cv2
import mediapipe as mp

from norfair import Video, Detection, Tracker
from norfair.drawing import Paths

from utils import PixelCoordinatesProjecter, draw_3d_tracked_boxes, scaled_euclidean


parser = argparse.ArgumentParser(description="3d-Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--model-name", type=str, default="Shoe", help="model name ('Shoe', 'Chair', 'Cup', 'Camera')")
parser.add_argument("--max-objects", type=int, default="2", help="Maximum number of objects at a time")
parser.add_argument("--hit-counter-max", type=int, default="40", help="Maximum value that hit counters may take")
parser.add_argument("--initialization-delay", type=int, default="10", help="Initialization delay")
parser.add_argument("--conf-threshold", type=float, default="0.1", help="Detector threshold") 
parser.add_argument("--distance-threshold", type=float, default="0.5", help="Distance threshold")
parser.add_argument("--output-path", type=str, default=".", help="Output path")
parser.add_argument("--draw-paths", action="store_true", help="Draw path of the centroid")
args = parser.parse_args()

mp_objectron = mp.solutions.objectron

with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=args.max_objects,
                            min_detection_confidence=args.conf_threshold,
                            model_name=args.model_name) as objectron:

    for input_path in args.files:
        video = Video(input_path = input_path, output_path = args.output_path)

        tracker = Tracker(
            distance_function=scaled_euclidean,
            distance_threshold=args.distance_threshold,
            hit_counter_max=args.hit_counter_max,
            initialization_delay=args.initialization_delay,
        )
        projecter = None
        for frame in video:
            if projecter is None:
                # Set function that projects 3d objets to the pixel space
                projecter = PixelCoordinatesProjecter(frame.shape[:2][::-1])

                if args.draw_paths:
                    # initialize path drawer
                    def get_points_to_draw(points3d):
                        return [projecter.eye_2_pixel(points3d)[0].astype(int)]
                    path_drawer = Paths(get_points_to_draw)

            detections = []

            # Convert the BGR image to RGB and process it with MediaPipe Objectron.
            results = objectron.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if not results.detected_objects:
                video.write(frame)
                tracker.update(detections)
                continue
 
            for detected_object in results.detected_objects:
                points = np.array([[p.x, p.y, p.z] for p in detected_object.landmarks_3d.landmark])
                detections.append(Detection(points=points))

            tracked_objects = tracker.update(detections)

            frame = draw_3d_tracked_boxes(frame, tracked_objects, projecter=projecter.eye_2_pixel)
            if args.draw_paths:
                frame = path_drawer.draw(frame, tracked_objects)

            video.write(frame)
