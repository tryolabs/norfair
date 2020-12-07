import os
import time
import sys
import numpy as np
import random
import cv2
from rich import print
from .utils import validate_points
from rich.progress import BarColumn, Progress, TimeRemainingColumn, track
from norfair import Detection
import norfair
import motmetrics as mm
from collections import OrderedDict
import pandas as pd

class InformationFile:
    def __init__(self, file_path):
        with open(file_path, "r") as myfile:
            self.file = myfile.read()

    def search(self, variable_name):
        index_position_on_this_document = self.file.find(variable_name)
        index_position_on_this_document = index_position_on_this_document + len(variable_name)
        while not self.file[index_position_on_this_document].isdigit():
            index_position_on_this_document += 1
        value_string = ""
        while self.file[index_position_on_this_document].isdigit():
            value_string = value_string + self.file[index_position_on_this_document]
            index_position_on_this_document += 1
        return int(value_string)

def write_predictions(frame_number, objects=None, output_text_file=None):
    '''
    Write tracked object information on the output file (for this frame), in the format
    frame_number, id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1
    '''
    # write tracked objects information in the output file
    for t in range(len(objects)):
        frame_str = str(int(frame_number))
        id_str = str(int(objects[t].id))
        bb_left_str = str((objects[t].estimate[0, 0]))
        bb_top_str = str((objects[t].estimate[0, 1]))  # [0,1]
        bb_width_str = str((objects[t].estimate[1, 0] - objects[t].estimate[0, 0]))
        bb_height_str = str((objects[t].estimate[1, 1] - objects[t].estimate[0, 1]))
        row_text_out = (
            frame_str
            + ","
            + id_str
            + ","
            + bb_left_str
            + ","
            + bb_top_str
            + ","
            + bb_width_str
            + ","
            + bb_height_str
            + ",-1,-1,-1,-1"
        )
        output_text_file.write(row_text_out)
        output_text_file.write("\n")


def process_text_file(text_file_path):
    """
    This function proccess detections and ground truth text files, ordering the data by frame, and making the box coordinates reference to corners positions.
    """
    matrix_data = np.loadtxt(text_file_path, dtype="f", delimiter=",")
    row_order = np.argsort(matrix_data[:, 0])
    matrix_data = matrix_data[row_order]
    # coordinates refer to box corners
    matrix_data[:, 4] = matrix_data[:, 2] + matrix_data[:, 4]
    matrix_data[:, 5] = matrix_data[:, 3] + matrix_data[:, 5]
    return matrix_data


class PredictionsTextFile:
    def __init__(self, input_path, save_path=".", information_file=None):

        file_name = os.path.split(input_path)[1]

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path = seqinfo_path)

        self.length = information_file.search(variable_name = "seqLength")

        predictions_folder = os.path.join(save_path, "predictions")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)

        out_file_name = os.path.join(predictions_folder, file_name + ".txt")
        self.text_file = open(out_file_name, "w+")

        self.frame_number = 1

    def update_text_file(self, predictions):
        write_predictions(
            frame_number=self.frame_number, objects=predictions, output_text_file=self.text_file
        )
        self.frame_number += 1

        if self.frame_number > self.length:
            self.text_file.close()

def get_detections(input_path, information_file=None):
    class DetectionFileParser:
        """Get Norfair detections from MOTChallenge text files containing detections"""
        def __init__(self, input_path, information_file=None):
            #Get detecions matrix data with rows corresponding to:
            # frame, id, bb_left, bb_top, bb_right, bb_down, conf, x, y, z
            detections_path = os.path.join(input_path, "det/det.txt")
            self.matrix_detections = process_text_file(text_file_path=detections_path)

            if information_file is None:
                seqinfo_path = os.path.join(input_path, "seqinfo.ini")
                information_file = InformationFile(file_path = seqinfo_path)
            length = information_file.search(variable_name = "seqLength")

            self.ordered_by_frame = []

            for frame_number in np.arange(1, length + 1):
                dets_on_this_frame = self.get_dets_from_frame(frame_number)
                self.ordered_by_frame.append(dets_on_this_frame)

        def get_dets_from_frame(self, frame_number):
            """ this function returns a list of norfair Detections class, corresponding to frame=frame_number """

            indexes = np.argwhere(self.matrix_detections[:, 0] == frame_number)
            detections = []
            if len(indexes) > 0:
                actual_det = self.matrix_detections[indexes]
                actual_det.shape = [actual_det.shape[0], actual_det.shape[2]]
                for det in actual_det:
                    points = np.array([[det[2], det[3]], [det[4], det[5]]])
                    conf = det[6]
                    new_detection = Detection(
                        points, np.array([conf, conf])
                    ) 
                    detections.append(new_detection)
            self.actual_detections = detections
            return detections

    return DetectionFileParser(input_path=input_path, information_file=information_file).ordered_by_frame

class Accumulators:
    def __init__(self):
        self.matrixes_predictions = []
        self.paths = []

    def create_accumulator(self, input_path, information_file=None):

        file_name = os.path.split(input_path)[1]

        self.frame_number = 1
        # save the path of this video in a list
        self.paths = np.hstack((self.paths, input_path))
        # initialize a matrix where we will save our predictions for this video (in the MOTChallenge format)
        self.matrix_predictions = []

        # initialize progress bar
        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path = seqinfo_path)
        length = information_file.search(variable_name = "seqLength")
        self.progress_bar_iter = track(
            range(length - 1), description=file_name, transient=False
        )

    def update(self, predictions=None):
        # get the tracked boxes from this frame in an array
        for obj in predictions:
            new_row = [
                self.frame_number,
                obj.id,
                obj.estimate[0, 0],
                obj.estimate[0, 1],
                obj.estimate[1, 0] - obj.estimate[0, 0],
                obj.estimate[1, 1] - obj.estimate[0, 1],
                -1,
                -1,
                -1,
                -1,
            ]
            if np.shape(self.matrix_predictions)[0] == 0:
                self.matrix_predictions = new_row
            else:
                self.matrix_predictions = np.vstack((self.matrix_predictions, new_row))
        self.frame_number += 1
        # advance in progress bar
        try:
            next(self.progress_bar_iter)
        except StopIteration:
            self.matrixes_predictions.append(self.matrix_predictions)
            return

    def compute_metrics(
        self,
        metrics=None,
        generate_overall=True,
    ):
        if metrics is None:
            metrics = list(mm.metrics.motchallenge_metrics)

        self.summary = eval_motChallenge(
            matrixes_predictions=self.matrixes_predictions,
            paths=self.paths,
            metrics=metrics,
            generate_overall=generate_overall,
        )

        return self.summary

    def save_metrics(self, save_path=".", file_name="metrics.txt"):

        if not os.path.exists(save_path):
            os.makedirs(save_folder)
        metrics_path = os.path.join(save_path, file_name)
        metrics_file = open(metrics_path, "w+")
        metrics_file.write(self.summary)
        metrics_file.close()

    def print_metrics(self):
        print(self.summary)


def load_motchallenge(matrix_data, min_confidence=-1):
    """Load MOT challenge data.

    This is modification of the function load_motchallenge from the MOT library, defined in io.py
    The panda dataframe is generated from a numpy array (matrix_data) instead of a text file.

    Params
    ------
    matrix_data : array  of float that has [frame, id, X, Y, width, height, conf, cassId, visibility] in each row, for each prediction on a particular video

    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """

    df = pd.DataFrame(
        data=matrix_data,
        columns=[
            "FrameId",
            "Id",
            "X",
            "Y",
            "Width",
            "Height",
            "Confidence",
            "ClassId",
            "Visibility",
            "unused",
        ],
    )
    df = df.set_index(["FrameId", "Id"])
    # Account for matlab convention.
    df[["X", "Y"]] -= (1, 1)

    # Removed trailing column
    del df["unused"]

    # Remove all rows without sufficient confidence
    return df[df["Confidence"] >= min_confidence]


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        print("Comparing ", k, "...")
        if k in gts:
            accs.append(
                mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5)
            )
            names.append(k)

    return accs, names


def eval_motChallenge(matrixes_predictions, paths, metrics=None, generate_overall=True):
    gt = OrderedDict(
        [
            (
                os.path.split(p)[1],
                mm.io.loadtxt(
                    os.path.join(p, "gt/gt.txt"),
                    fmt="mot15-2D",
                    min_confidence=1,
                ),
            )
            for p in paths
        ]
    )

    ts = OrderedDict(
        [
            (os.path.split(paths[n])[1], load_motchallenge(matrixes_predictions[n]))
            for n in range(len(paths))
        ]
    )

    mh = mm.metrics.create()

    accs, names = compare_dataframes(gt, ts)

    if metrics == None:
        metrics = list(mm.metrics.motchallenge_metrics)
    mm.lap.default_solver = "scipy"
    print("Computing metrics...")
    summary = mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=generate_overall
    )
    summary = mm.io.render_summary(
        summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
    )
    return summary
