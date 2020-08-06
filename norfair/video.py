import cv2
import os
import time

from rich import print
from rich.progress import Progress, BarColumn, TimeRemainingColumn


class Video:
    def __init__(self, input_path, output_path=".", label="", codec_fourcc=None):
        self.input_path = input_path
        self.output_path = output_path
        self.output_video = None
        self.label = label
        self.codec_fourcc = codec_fourcc

        # Read Input Video
        self.video_capture = cv2.VideoCapture(self.input_path)
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            fail_msg = "[bold red]Error reading input video file:[/bold red] Make sure file exists and is a video file."
            if "~" in self.input_path:
                fail_msg += (
                    "\n[yellow]Using ~ as abbreviation for your home folder is not supported.[/yellow]"
                )
            self._fail(fail_msg)
        self.input_frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.input_frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_counter = 0

        # Setup progressbar
        description = os.path.basename(self.input_path)
        if self.label:
            description += f" | {self.label}"
        _, terminal_columns = os.popen("stty size", "r").read().split()
        space_for_description = int(terminal_columns) - 25  # Leave 25 space for progressbar
        abbreviated_description = (
            description
            if len(description) < space_for_description
            else "{} ... {}".format(
                description[: space_for_description // 2 - 3], description[-space_for_description // 2 + 3 :]
            )
        )
        self.progress_bar = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            "[yellow]{task.fields[fps]:.2f}fps",
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.task = self.progress_bar.add_task(abbreviated_description, total=total_frames, fps=0)

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
                progress_bar.update(self.task, advance=1, refresh=True, fps=process_fps)
                yield frame

        # Cleanup
        if self.output_video is not None:
            self.output_video.release()
        self.video_capture.release()
        cv2.destroyAllWindows()

    def _fail(self, msg):
        print(msg)
        exit()

    def write(self, frame):
        if self.output_video is None:
            # The user may need to access the output file path on their code
            self.output_file_path = self.generate_output_path(".mp4")
            fourcc = cv2.VideoWriter_fourcc(*self.get_codec_fourcc(self.output_file_path))
            output_size = tuple(frame.shape[1::-1])  # Invert frame.shape[0] (rows) -> output_size[1] (height)
            self.output_video = cv2.VideoWriter(self.output_file_path, fourcc, self.fps, output_size,)

        self.output_video.write(frame)
        cv2.waitKey(1)

    def show(self, frame, downsample_ratio=1):
        # Resize to lower resolution for faster streaming over slow connections
        if downsample_ratio is not None:

            # Note that frame.shape[1] corresponds to width, and opencv format is (width, height)
            frame = cv2.resize(
                frame, (frame.shape[1] // downsample_ratio, frame.shape[0] // downsample_ratio),
            )
        cv2.imshow("Output", frame)
        cv2.waitKey(1)

    def generate_output_path(self, extension, prefix=""):
        if os.path.isdir(self.output_path):
            base_file_name = self.input_path.split("/")[-1].split(".")[0]
            file_name = prefix + base_file_name + "_out" + extension
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
