import click
import cv2
import os
import time


class Video:
    def __init__(self, input_path, output_path=".", output_extension=None):
        self.input_path = input_path
        self.output_path = output_path
        self.output_video = None
        self.output_filename = Video.generate_output_filename(
            self.input_path, self.output_path, output_extension
        )

        # Read Input Video
        self.video_capture = cv2.VideoCapture(self.input_path)
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            click.echo(click.style("Error reading input video file.", bold=True))
            click.echo(" Make sure file exists and is a video file.")
            if "~" in self.input_path:
                click.echo(" Using ~ as abbreviation for your home folder is not supported.")
            exit()
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_counter = 0

        # Setup progressbar
        file_name = os.path.basename(self.input_path)
        _, terminal_columns = os.popen("stty size", "r").read().split()
        space_for_filename = int(terminal_columns) - 75  # Leave 75 space for progressbar
        abbreviated_file_name = (
            file_name
            if len(file_name) < space_for_filename
            else "{} ... {}".format(
                file_name[: space_for_filename // 2 - 3], file_name[-space_for_filename // 2 + 3 :]
            )
        )
        label = "{} {}x{}@{:.0f}fps".format(
            abbreviated_file_name, self.frame_width, self.frame_height, self.fps
        )
        self.video_progress_bar = click.progressbar(length=total_frames, label=label)

    # This is a generator, note the yield keyword below.
    def __iter__(self):
        with self.video_progress_bar as progress_bar:
            start = time.time()

            # Iterate over video
            for _ in progress_bar:
                self.frame_counter += 1
                _, frame = self.video_capture.read()
                yield frame

            stop = time.time()
            click.echo(f" {self.frame_counter / (stop - start):.2f} fps", nl=False)

        # Cleanup
        if self.output_video is not None:
            self.output_video.release()
        self.video_capture.release()
        cv2.destroyAllWindows()

    def write(self, frame):
        if self.output_video is None:
            fourcc = cv2.VideoWriter_fourcc(*Video.get_output_codec(self.output_filename))
            self.output_video = cv2.VideoWriter(
                self.output_filename, fourcc, self.fps, (self.frame_width, self.frame_height),
            )
        self.output_video.write(frame)
        cv2.waitKey(1)

    def show(self, frame, downsample_ratio=1):
        # Resize to lower resolution for faster streaming over slow connections
        if downsample_ratio is not None:
            frame = cv2.resize(
                frame,
                (
                    int(self.frame_width / downsample_ratio),
                    int(self.frame_height / downsample_ratio),
                ),
            )
        cv2.imshow("Output", frame)
        cv2.waitKey(1)

    @staticmethod
    def generate_output_filename(input_path, output_path=".", extension=None):
        if os.path.isdir(output_path):
            file_name = input_path.split("/")[-1].split(".")[0]
            extension = "avi" if extension is None else extension[-3:].lower()
            return f"{output_path}/{file_name}_out.{extension}"
        else:
            if extension is not None:
                click.echo("Ignoring output_extension: using file extension from output_path")
            return output_path

    @staticmethod
    def get_output_codec(output_path):
        extension = output_path[-3:].lower()
        if "avi" == extension:
            return "XVID"
        elif "mp4" == extension:
            return "mp4v"
        else:
            click.echo(f"Not supported video filename extension: {extension}")
            exit()
