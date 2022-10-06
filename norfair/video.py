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

from norfair import metrics

from .utils import get_terminal_size


class Video:
    """
    Class that provides a simple and pythonic way to interact with video.

    It returns regular OpenCV frames which enables the usage of the huge number of tools OpenCV provides to modify images.

    Parameters
    ----------
    camera : Optional[int], optional
        An integer representing the device id of the camera to be used as the video source.

        Webcams tend to have an id of `0`. Arguments `camera` and `input_path` can't be used at the same time, one must be chosen.
    input_path : Optional[str], optional
        A string consisting of the path to the video file to be used as the video source.

        Arguments `camera` and `input_path` can't be used at the same time, one must be chosen.
    output_path : str, optional
        The path to the output video to be generated.
        Can be a folder were the file will be created or a full path with a file name.
    output_fps : Optional[float], optional
        The frames per second at which to encode the output video file.

        If not provided it is set to be equal to the input video source's fps.
        This argument is useful when using live video cameras as a video source,
        where the user may know the input fps,
        but where the frames are being fed to the output video at a rate that is lower than the video source's fps,
        due to the latency added by the detector.
    label : str, optional
        Label to add to the progress bar that appears when processing the current video.
    output_fourcc : Optional[str], optional
        OpenCV encoding for output video file.
        By default we use `mp4v` for `.mp4` and `XVID` for `.avi`. This is a combination that works on most systems but
        it results in larger files. To get smaller files use `avc1` or `H264` if available.
        Notice that some fourcc are not compatible with some extensions.
    output_extension : str, optional
        File extension used for the output video. Ignored if `output_path` is not a folder.

    Examples
    --------
    >>> video = Video(input_path="video.mp4")
    >>> for frame in video:
    >>>     # << Your modifications to the frame would go here >>
    >>>     video.write(frame)
    """

    def __init__(
        self,
        camera: Optional[int] = None,
        input_path: Optional[str] = None,
        output_path: str = ".",
        output_fps: Optional[float] = None,
        label: str = "",
        output_fourcc: Optional[str] = None,
        output_extension: str = "mp4",
    ):
        self.camera = camera
        self.input_path = input_path
        self.output_path = output_path
        self.label = label
        self.output_fourcc = output_fourcc
        self.output_extension = output_extension
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

    def write(self, frame: np.ndarray) -> int:
        """
        Write one frame to the output video.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to write to file.

        Returns
        -------
        int
            _description_
        """
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

    def show(self, frame: np.ndarray, downsample_ratio: float = 1.0) -> int:
        """
        Display a frame through a GUI. Usually used inside a video inference loop to show the output video.

        Parameters
        ----------
        frame : np.ndarray
            The OpenCV frame to be displayed.
        downsample_ratio : float, optional
            How much to downsample the frame being show.

            Useful when streaming the GUI video display through a slow internet connection using something like X11 forwarding on an ssh connection.

        Returns
        -------
        int
            _description_
        """
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
        """
        Calculate the output path being used in case you are writing your frames to a video file.

        Useful if you didn't set `output_path`, and want to know what the autogenerated output file path by Norfair will be.

        Returns
        -------
        str
            The path to the file.
        """
        if not os.path.isdir(self.output_path):
            return self.output_path

        if self.input_path is not None:
            file_name = self.input_path.split("/")[-1].split(".")[0]
        else:
            file_name = "camera_{self.camera}"
        file_name = f"{file_name}_out.{self.output_extension}"

        return os.path.join(self.output_path, file_name)

    def get_codec_fourcc(self, filename: str) -> Optional[str]:
        if self.output_fourcc is not None:
            return self.output_fourcc

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
    def __init__(
        self, input_path, save_path=".", information_file=None, make_video=True
    ):

        if information_file is None:
            information_file = metrics.InformationFile(
                file_path=os.path.join(input_path, "seqinfo.ini")
            )
        if make_video:
            file_name = os.path.split(input_path)[1]

            # Search framerate on seqinfo.ini
            fps = information_file.search(variable_name="frameRate")

            # Search resolution in seqinfo.ini
            horizontal_resolution = information_file.search(variable_name="imWidth")
            vertical_resolution = information_file.search(variable_name="imHeight")
            image_size = (horizontal_resolution, vertical_resolution)

            videos_folder = os.path.join(save_path, "videos")
            if not os.path.exists(videos_folder):
                os.makedirs(videos_folder)

            video_path = os.path.join(videos_folder, file_name + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.file_name = file_name
            # Video file
            self.video = cv2.VideoWriter(video_path, fourcc, fps, image_size)

        self.length = information_file.search(variable_name="seqLength")
        self.input_path = input_path
        self.frame_number = 1
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
