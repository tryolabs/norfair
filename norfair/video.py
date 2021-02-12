import os
import time
from typing import List, Optional, Union

try:
    import cv2
except ImportError:
    from .utils import DummyOpenCVImport

    cv2 = DummyOpenCVImport()
import numpy as np
from rich import print
from rich.progress import BarColumn, Progress, ProgressColumn, TimeRemainingColumn

from .utils import get_terminal_size


class Video:
    def __init__(
        self,
        camera: Optional[int] = None,
        input_path: Optional[str] = None,
        output_path: str = ".",
        output_fps: Optional[float] = None,
        label: str = "",
        codec_fourcc: Optional[str] = None,
    ):
        self.camera = camera
        self.input_path = input_path
        self.output_path = output_path
        self.label = label
        self.codec_fourcc = codec_fourcc
        self.output_video: Optional[cv2.VideoWriter] = None

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
            if "~" in self.input_path:
                self.input_path = os.path.expanduser(self.input_path)
            if not os.path.isfile(self.input_path):
                self._fail(
                    f"[bold red]Error:[/bold red] File '{self.input_path}' does not exist."
                )
            self.video_capture = cv2.VideoCapture(self.input_path)
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                self._fail(
                    f"[bold red]Error:[/bold red] '{self.input_path}' does not seem to be a video file supported by OpenCV. If the video file is not the problem, please check that your OpenCV installation is working correctly."
                )
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
        progress_bar_fields: List[Union[str, ProgressColumn]] = [
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

    def _fail(self, msg: str):
        print(msg)
        exit()

    def write(self, frame: np.array) -> int:
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

    def show(self, frame: np.array, downsample_ratio: float = 1.0) -> int:
        # Resize to lower resolution for faster streaming over slow connections
        if downsample_ratio != 1.0:
            frame = cv2.resize(
                frame,
                (
                    frame.shape[1] // downsample_ratio,
                    frame.shape[0] // downsample_ratio,
                ),
            )
        cv2.imshow("Output", frame)
        return cv2.waitKey(1)

    def get_output_file_path(self) -> str:
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

    def get_codec_fourcc(self, filename: str) -> Optional[str]:
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
            return (
                None  # Had to add this return to make mypya happy. I don't like this.
            )

    def abbreviate_description(self, description: str) -> str:
        """Conditionally abbreviate description so that progress bar fits in small terminals"""
        terminal_columns, _ = get_terminal_size()
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


class VideoFromFrames:
    def __init__(self, input_path, save_path=".", information_file=None):

        if information_file is None:
            information_file = metrics.InformationFile(
                file_path=os.path.join(input_path, "seqinfo.ini")
            )

        file_name = os.path.split(input_path)[1]

        # Search framerate on seqinfo.ini
        fps = information_file.search(variable_name="frameRate")

        # Search resolution in seqinfo.ini
        horizontal_resolution = information_file.search(variable_name="imWidth")
        vertical_resolution = information_file.search(variable_name="imHeight")
        image_size = (horizontal_resolution, vertical_resolution)

        # Search total frames in seqinfo.ini
        self.length = information_file.search(variable_name="seqLength")

        videos_folder = os.path.join(save_path, "videos")
        if not os.path.exists(videos_folder):
            os.makedirs(videos_folder)

        video_path = os.path.join(videos_folder, file_name + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.file_name = file_name
        self.input_path = input_path
        self.frame_number = 1
        self.video = cv2.VideoWriter(video_path, fourcc, fps, image_size)  # Video file
        self.image_extension = information_file.search("imExt")
        self.image_directory = information_file.search("imDir")

    def __iter__(self):
        self.frame_number = 1
        return self

    def __next__(self):
        if self.frame_number <= self.length:
            frame_path = os.path.join(
                self.input_path,
                self.image_directory,
                str(self.frame_number).zfill(6) + self.image_extension,
            )
            self.frame_number += 1

            return cv2.imread(frame_path)
        raise StopIteration()

    def update(self, frame):
        self.video.write(frame)
        cv2.waitKey(1)

        if self.frame_number > self.length:
            cv2.destroyAllWindows()
            self.video.release()
