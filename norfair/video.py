import cv2
import os
import time

from rich import print
from rich.progress import Progress, BarColumn, TimeRemainingColumn

class Video():

    def __init__(self, input_path, output_path="."):
        self.input_path = input_path
        self.output_path = output_path
        self.output_video = None

        # Read Input Video
        self.video_capture = cv2.VideoCapture(self.input_path)
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            print("[bold red]Error reading input video file:[/bold red] Make sure file exists and is a video file.")
            if "~" in self.input_path:
                print("Using ~ as abbreviation for your home folder is not supported.")
            exit()
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.frame_counter = 0

        # Setup progressbar
        file_name = os.path.basename(self.input_path)
        _, terminal_columns = os.popen('stty size', 'r').read().split()
        space_for_filename = int(terminal_columns) - 40  # Leave 40 space for progressbar
        abbreviated_file_name = (
            file_name if len(file_name) < space_for_filename
            else "{} ... {}".format(file_name[:space_for_filename // 2 - 3],
                                    file_name[-space_for_filename // 2 + 3:])
        )
        label = "{} {}x{}@{:.0f}fps".format(
            abbreviated_file_name, self.frame_width, self.frame_height, self.fps
        )
        self.progress_bar = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            "[yellow]{task.fields[fps]:.2f}fps",
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False
        )
        self.task = self.progress_bar.add_task(label, total=total_frames, fps=0)

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

    def write(self, frame):
        if self.output_video is None:
            # The user may need to access the output file path on their code
            self.output_file_path = self.generate_output_path('.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output_video = cv2.VideoWriter(
                # TODO: Can we do mp4 instead of avi? Make this configurable?
                self.output_file_path,
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
        self.output_video.write(frame)
        cv2.waitKey(1)

    def show(self, frame, downsample_ratio=1):
        # Resize to lower resolution for faster streaming over slow connections
        if downsample_ratio is not None:
            frame = cv2.resize(
                frame,
                (int(self.frame_width / downsample_ratio),
                 int(self.frame_height / downsample_ratio))
            )
        cv2.imshow("Output", frame)
        cv2.waitKey(1)

    def generate_output_path(self, extension, prefix=''):
        if os.path.isdir(self.output_path):
            base_file_name = self.input_path.split('/')[-1].split('.')[0]
            file_name = prefix + base_file_name + "_out" + extension
            return os.path.join(self.output_path, file_name)
        elif self.output_path[-4:] in [".avi", ".mp4"]:
            return self.output_path
        else:
            # TODO: Check that we support mp4
            print(f"'{self.output_path}' is not an existing folder or an avi/mp4 filename.")
            print("Exiting...")
            exit()
