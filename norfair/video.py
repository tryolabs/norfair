import os
import time

import cv2
from rich import print
from rich.progress import BarColumn, Progress, TimeRemainingColumn


class Video:
    def __init__(
        self,
        camera=None,
        input_path=None,
        output_path=".",
        output_fps=None,
        label="",
        codec_fourcc=None,
    ):
        self.camera = camera
        self.input_path = input_path
        self.output_path = output_path
        self.label = label
        self.codec_fourcc = codec_fourcc
        self.output_video = None

        # Input validation
        if (input_path is None and camera is None) or (
            input_path is not None and camera is not None
        ):
            raise ValueError(
                "You must set either 'camera' or 'input_path' arguments when setting 'Video' class"
            )
        if camera is not None and type(camera) is not int:
            raise ValueError(
                "Argument 'camera' refers to the device-id of your camera, and must be an int. Setting it to 0 usually works if you don't know the id."
            )

        # Read Input Video
        if self.input_path is not None:
            self.video_capture = cv2.VideoCapture(self.input_path)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                fail_msg = "[bold red]Error reading input video file:[/bold red] Make sure file exists and is a video file."
                if "~" in self.input_path:
                    fail_msg += "\n[yellow]Using ~ as abbreviation for your home folder is not supported.[/yellow]"
                self._fail(fail_msg)
            description = os.path.basename(self.input_path)
        else:
            self.video_capture = cv2.VideoCapture(self.camera)
            total_frames = 0
            description = f"Camera({self.camera})"
        self.output_fps = (
            output_fps
            if output_fps is not None
            else self.video_capture.get(cv2.CAP_PROP_FPS)
        )
        self.input_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.input_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_counter = 0

        # Setup progressbar
        if self.label:
            description += f" | {self.label}"
        progress_bar_fields = [
            "[progress.description]{task.description}",
            BarColumn(),
            "[yellow]{task.fields[process_fps]:.2f}fps[/yellow]",
        ]
        if self.input_path is not None:
            progress_bar_fields.insert(
                2, "[progress.percentage]{task.percentage:>3.0f}%"
            )
            progress_bar_fields.insert(
                3,
                TimeRemainingColumn(),
            )
        self.progress_bar = Progress(
            *progress_bar_fields,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task = self.progress_bar.add_task(
            self.abbreviate_description(description),
            total=total_frames,
            start=self.input_path is not None,
            process_fps=0,
        )

    # This is a generator, note the yield keyword below.
    def __iter__(self):
        with self.progress_bar as progress_bar:
            start = time.time()

            # Iterate over video
            while True:
                self.frame_counter += 1
                ret, frame = self.video_capture.read()
                if ret is False or frame is None:
                    break
                process_fps = self.frame_counter / (time.time() - start)
                progress_bar.update(
                    self.task, advance=1, refresh=True, process_fps=process_fps
                )
                yield frame

        # Cleanup
        if self.output_video is not None:
            self.output_video.release()
            print(
                f"[white]Output video file saved to: {self.get_output_file_path()}[/white]"
            )
        self.video_capture.release()
        cv2.destroyAllWindows()

    def _fail(self, msg):
        print(msg)
        exit()

    def write(self, frame):
        if self.output_video is None:
            # The user may need to access the output file path on their code
            output_file_path = self.get_output_file_path()
            fourcc = cv2.VideoWriter_fourcc(*self.get_codec_fourcc(output_file_path))
            # Set on first frame write in case the user resizes the frame in some way
            output_size = (
                frame.shape[1],
                frame.shape[0],
            )  # OpenCV format is (width, height)
            self.output_video = cv2.VideoWriter(
                output_file_path,
                fourcc,
                self.output_fps,
                output_size,
            )

        self.output_video.write(frame)
        return cv2.waitKey(1)

    def show(self, frame, downsample_ratio=1):
        # Resize to lower resolution for faster streaming over slow connections
        if downsample_ratio is not None:

            # Note that frame.shape[1] corresponds to width, and opencv format is (width, height)
            frame = cv2.resize(
                frame,
                (
                    frame.shape[1] // downsample_ratio,
                    frame.shape[0] // downsample_ratio,
                ),
            )
        cv2.imshow("Output", frame)
        return cv2.waitKey(1)

    def get_output_file_path(self):
        output_path_is_dir = os.path.isdir(self.output_path)
        if output_path_is_dir and self.input_path is not None:
            base_file_name = self.input_path.split("/")[-1].split(".")[0]
            file_name = base_file_name + "_out.mp4"
            return os.path.join(self.output_path, file_name)
        elif output_path_is_dir and self.camera is not None:
            file_name = f"camera_{self.camera}_out.mp4"
            return os.path.join(self.output_path, file_name)
        else:
            return self.output_path

    def get_codec_fourcc(self, filename):
        if self.codec_fourcc is not None:
            return self.codec_fourcc

        # Default codecs for each extension
        extension = filename[-3:].lower()
        if "avi" == extension:
            return "XVID"
        elif "mp4" == extension:
            return "mp4v"  # When available, "avc1" is better
        else:
            self._fail(
                f"[bold red]Could not determine video codec for the provided output filename[/bold red]: "
                f"[yellow]{filename}[/yellow]\n"
                f"Please use '.mp4', '.avi', or provide a custom OpenCV fourcc codec name."
            )

    def abbreviate_description(self, description):
        """Conditionally abbreviate description so that progress bar fits in small terminals"""
        _, terminal_columns = os.popen("stty size", "r").read().split()
        space_for_description = (
            int(terminal_columns) - 25
        )  # Leave 25 space for progressbar
        if len(description) < space_for_description:
            return description
        else:
            return "{} ... {}".format(
                description[: space_for_description // 2 - 3],
                description[-space_for_description // 2 + 3 :],
            )
