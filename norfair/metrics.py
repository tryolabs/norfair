import os

import numpy as np
from rich import print
from rich.progress import track

from norfair import Detection

try:
    import motmetrics as mm
    import pandas as pd
except ImportError:
    from .utils import DummyMOTMetricsImport

    mm = DummyMOTMetricsImport()
    pandas = DummyMOTMetricsImport()
from collections import OrderedDict


class InformationFile:
    def __init__(self, file_path):
        self.path = file_path
        with open(file_path, "r") as myfile:
            file = myfile.read()
        self.lines = file.splitlines()

    def search(self, variable_name):
        for line in self.lines:
            if line[: len(variable_name)] == variable_name:
                result = line[len(variable_name) + 1 :]
                break
        else:
            raise ValueError(f"Couldn't find '{variable_name}' in {self.path}")
        if result.isdigit():
            return int(result)
        else:
            return result


class PredictionsTextFile:
    """Generates a text file with your predicted tracked objects, in the MOTChallenge format.
    It needs the 'input_path', which is the path to the sequence being processed,
    the 'save_path', and optionally the 'information_file' (in case you don't give an
    'information_file', is assumed there is one in the input_path folder).
    """

    def __init__(self, input_path, save_path=".", information_file=None):

        file_name = os.path.split(input_path)[1]

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)

        self.length = information_file.search(variable_name="seqLength")

        predictions_folder = os.path.join(save_path, "predictions")
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)

        out_file_name = os.path.join(predictions_folder, file_name + ".txt")
        self.text_file = open(out_file_name, "w+")

        self.frame_number = 1

    def update(self, predictions, frame_number=None):
        if frame_number is None:
            frame_number = self.frame_number
        """
        Write tracked object information in the output file (for this frame), in the format
        frame_number, id, bb_left, bb_top, bb_width, bb_height, -1, -1, -1, -1
        """
        for obj in predictions:
            frame_str = str(int(frame_number))
            id_str = str(int(obj.id))
            bb_left_str = str((obj.estimate[0, 0]))
            bb_top_str = str((obj.estimate[0, 1]))  # [0,1]
            bb_width_str = str((obj.estimate[1, 0] - obj.estimate[0, 0]))
            bb_height_str = str((obj.estimate[1, 1] - obj.estimate[0, 1]))
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
            self.text_file.write(row_text_out)
            self.text_file.write("\n")

        self.frame_number += 1

        if self.frame_number > self.length:
            self.text_file.close()


class DetectionFileParser:
    """Get Norfair detections from MOTChallenge text files containing detections"""

    def __init__(self, input_path, information_file=None):
        self.frame_number = 1

        # Get detecions matrix data with rows corresponding to:
        # frame, id, bb_left, bb_top, bb_right, bb_down, conf, x, y, z
        detections_path = os.path.join(input_path, "det/det.txt")

        self.matrix_detections = np.loadtxt(detections_path, dtype="f", delimiter=",")
        row_order = np.argsort(self.matrix_detections[:, 0])
        self.matrix_detections = self.matrix_detections[row_order]
        # Coordinates refer to box corners
        self.matrix_detections[:, 4] = (
            self.matrix_detections[:, 2] + self.matrix_detections[:, 4]
        )
        self.matrix_detections[:, 5] = (
            self.matrix_detections[:, 3] + self.matrix_detections[:, 5]
        )

        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)
        self.length = information_file.search(variable_name="seqLength")

        self.sorted_by_frame = []
        for frame_number in range(1, self.length + 1):
            self.sorted_by_frame.append(self.get_dets_from_frame(frame_number))

    def get_dets_from_frame(self, frame_number):
        """this function returns a list of norfair Detections class, corresponding to frame=frame_number"""

        indexes = np.argwhere(self.matrix_detections[:, 0] == frame_number)
        detections = []
        if len(indexes) > 0:
            actual_det = self.matrix_detections[indexes]
            actual_det.shape = [actual_det.shape[0], actual_det.shape[2]]
            for det in actual_det:
                points = np.array([[det[2], det[3]], [det[4], det[5]]])
                conf = det[6]
                new_detection = Detection(points, np.array([conf, conf]))
                detections.append(new_detection)
        self.actual_detections = detections
        return detections

    def __iter__(self):
        self.frame_number = 1
        return self

    def __next__(self):
        if self.frame_number <= self.length:
            self.frame_number += 1
            # Frame_number is always 1 unit bigger than the corresponding index in self.sorted_by_frame, and
            # also we just incremented the frame_number, so now is 2 units bigger than the corresponding index
            return self.sorted_by_frame[self.frame_number - 2]

        raise StopIteration()


class Accumulators:
    def __init__(self):
        self.matrixes_predictions = []
        self.paths = []

    def create_accumulator(self, input_path, information_file=None):
        # Check that motmetrics is installed here, so we don't have to process
        # the whole dataset before failing out if we don't.
        mm.metrics

        file_name = os.path.split(input_path)[1]

        self.frame_number = 1
        # Save the path of this video in a list
        self.paths = np.hstack((self.paths, input_path))
        # Initialize a matrix where we will save our predictions for this video (in the MOTChallenge format)
        self.matrix_predictions = []

        # Initialize progress bar
        if information_file is None:
            seqinfo_path = os.path.join(input_path, "seqinfo.ini")
            information_file = InformationFile(file_path=seqinfo_path)
        length = information_file.search(variable_name="seqLength")
        self.progress_bar_iter = track(
            range(length - 1), description=file_name, transient=False
        )

    def update(self, predictions=None):
        # Get the tracked boxes from this frame in an array
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
        # Advance in progress bar
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

        self.summary_text, self.summary_dataframe = eval_motChallenge(
            matrixes_predictions=self.matrixes_predictions,
            paths=self.paths,
            metrics=metrics,
            generate_overall=generate_overall,
        )

        return self.summary_dataframe

    def save_metrics(self, save_path=".", file_name="metrics.txt"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        metrics_path = os.path.join(save_path, file_name)
        metrics_file = open(metrics_path, "w+")
        metrics_file.write(self.summary_text)
        metrics_file.close()

    def print_metrics(self):
        print(self.summary_text)


def load_motchallenge(matrix_data, min_confidence=-1):
    """Load MOT challenge data.

    This is a modification of the function load_motchallenge from the py-motmetrics library, defined in io.py
    In this version, the pandas dataframe is generated from a numpy array (matrix_data) instead of a text file.

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
                    os.path.join(p, "gt/gt.txt"), fmt="mot15-2D", min_confidence=1
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

    if metrics is None:
        metrics = list(mm.metrics.motchallenge_metrics)
    mm.lap.default_solver = "scipy"
    print("Computing metrics...")
    summary_dataframe = mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=generate_overall
    )
    summary_text = mm.io.render_summary(
        summary_dataframe,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    return summary_text, summary_dataframe
