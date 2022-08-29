import argparse

import numpy as np
import cv2
import mediapipe as mp

from norfair import Video, Detection, Tracker, Color
from norfair.distances import mean_euclidean

parser = argparse.ArgumentParser(description="3d-Track objects in a video.")
parser.add_argument("files", type=str, nargs="+", help="Video files to process")
parser.add_argument("--model-name", type=str, default="Shoe", help="model name ('Shoe', 'Chair', 'Cup', 'Camera')")
parser.add_argument("--max-objects", type=int, default="2", help="Maximum number of objects at a time")
parser.add_argument("--conf-threshold", type=float, default="0.1", help="Detector threshold") # 0.2
parser.add_argument("--distance-threshold", type=float, default="0.5", help="Detector threshold") # 0.2
parser.add_argument("--output-path", type=str, default=".", help="Output path")
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=args.max_objects,
                            min_detection_confidence=args.conf_threshold,
                            model_name=args.model_name) as objectron:

    for input_path in args.files:
        video = Video(input_path = input_path, output_path = args.output_path)

        tracker = Tracker(
            distance_function=mean_euclidean,
            distance_threshold=args.distance_threshold,
        )
        for image in video:
            detections = []

            # Convert the BGR image to RGB and process it with MediaPipe Objectron.
            results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Draw box landmarks.
            if not results.detected_objects:
                video.write(image)
                tracker.update(detections)
                continue
 
            annotated_image = image.copy()
            for detected_object in results.detected_objects:
 
                points = np.array([[p.x, p.y, p.z] for p in detected_object.landmarks_3d.landmark])
                detections.append(Detection(points=points, data=detected_object))


            tracked_objects = tracker.update(detections)
            
            for obj in tracked_objects:
                color_spec = mp_drawing.DrawingSpec(color=Color.random(obj.id), thickness=5)
                mp_drawing.draw_landmarks(annotated_image, obj.last_detection.data.landmarks_2d, mp_objectron.BOX_CONNECTIONS, color_spec, color_spec)
                # mp_drawing.draw_axis(annotated_image, obj.last_detection.data.rotation, obj.last_detection.data.translation)

            video.write(annotated_image)
            
