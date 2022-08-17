import argparse
import os.path

import numpy as np

import sys
sys.path.append('../../../ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker
from norfair import drawing, metrics, video

parser = argparse.ArgumentParser(
    description="Evaluate a basic tracker on MOTChallenge data. Display on terminal the MOTChallenge metrics results "
)
parser.add_argument(
    "dataset_path",
    type=str,
    nargs="?",
    help="Path to the MOT Challenge train dataset folder (test dataset doesn't provide labels)",
)
parser.add_argument(
    "--make-video",
    dest="make_video",
    action="store_true",
    help="Generate an output video, using the frames provided by the MOTChallenge dataset.",
)
parser.add_argument(
    "--save-pred",
    dest="save_pred",
    action="store_true",
    help="Generate a text file with your predictions",
)
parser.add_argument(
    "--save-metrics",
    dest="save_metrics",
    action="store_true",
    help="Generate a text file with your MOTChallenge metrics results",
)
parser.add_argument(
    "--output-path", dest="output_path", type=str, nargs="?", default=".", help="Output path"
)
parser.add_argument(
    "--select-sequences",
    dest="select_sequences",
    type=str,
    nargs="+",
    help="If you want to select a subset of sequences in your dataset path. Insert the names of the sequences you want to process.",
)

args = parser.parse_args()

output_path = args.output_path

if args.save_metrics:
    print("Saving metrics file at " + os.path.join(output_path, "metrics.txt"))
if args.save_pred:
    print("Saving predictions files at " + os.path.join(output_path, "predictions/"))
if args.make_video:
    print("Saving videos at " + os.path.join(output_path, "videos/"))

if args.select_sequences is None:
    sequences_paths = [f.path for f in os.scandir(args.dataset_path) if f.is_dir()]
else:
    sequences_paths = [
        os.path.join(args.dataset_path, f) for f in args.select_sequences
    ]

accumulator = metrics.Accumulators()

class PartialStaticTracker:
    def __init__(self, estimate, obj_id, live_points):
        self.estimate = estimate
        self.id = obj_id
        self.live_points = live_points


for input_path in sequences_paths:
    # Search vertical resolution in seqinfo.ini
    seqinfo_path = os.path.join(input_path, "seqinfo.ini")
    info_file = metrics.InformationFile(file_path=seqinfo_path)

    all_detections = metrics.DetectionFileParser(
        input_path=input_path, information_file=info_file
    )

    if args.save_pred:
        predictions_text_file = metrics.PredictionsTextFile(
            input_path=input_path, save_path=output_path, information_file=info_file
        )

    if args.make_video:
        video_file = video.VideoFromFrames(
            input_path=input_path, save_path=output_path, information_file=info_file
            )
    
    class ArgsByte:
        def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
            self.mot20 = False
            self.track_thresh = track_thresh
            self.track_buffer = track_buffer
            self.match_thresh = match_thresh

    args_byte = ArgsByte(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    tracker = BYTETracker(args_byte, info_file.search('frameRate'))

    # Initialize accumulator for this video
    accumulator.create_accumulator(input_path=input_path, information_file=info_file)

    img_size = [info_file.search('imHeight'), info_file.search('imWidth')]

    byte_tracked_objects = []
    for frame_number, detections in enumerate(all_detections):

        byte_detections = []
        for det in detections:
            byte_det = np.append(det.points.reshape((1, -1)), det.scores[0])
            byte_detections.append(byte_det)

        if len(byte_detections)>0:
            byte_tracked_objects = tracker.update(
                np.array(byte_detections), img_size, tuple(img_size)
            )

        tracked_objects= []
        for obj in byte_tracked_objects:
            box = obj.tlbr.reshape((2, 2))
            tracked_objects.append(PartialStaticTracker(box, obj.track_id, np.array([True])))


        # Draw detection and tracked object boxes on frame
        if args.make_video:
            frame = next(video_file)
            frame = drawing.draw_boxes(frame, detections=detections)
            frame = drawing.draw_tracked_boxes(frame=frame, objects=tracked_objects)
            video_file.update(frame=frame)

        # Update output text file
        if args.save_pred:
            predictions_text_file.update(predictions=tracked_objects)

        accumulator.update(predictions=tracked_objects)

accumulator.compute_metrics()
accumulator.print_metrics()

if args.save_metrics:
    accumulator.save_metrics(save_path=output_path)
