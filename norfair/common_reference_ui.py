import os
import pickle
import tkinter as tk
import tkinter.filedialog
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError

from norfair import Video
from norfair.camera_motion import HomographyTransformationGetter, TransformationGetter


def resize_image(image, desired_width=None, desired_height=None):
    aspect_ratio = image.height / image.width

    if (desired_width is None) and (desired_height is not None):
        desired_width = int(desired_height / aspect_ratio)
    elif (desired_width is not None) and (desired_height is None):
        desired_height = int(aspect_ratio * desired_width)

    return image.resize((desired_width, desired_height), Image.LANCZOS)


def set_reference(
    reference: str,
    footage: str,
    transformation_getter: TransformationGetter = None,
    mask_generator=None,
    image_width=None,
    image_height=None,
    motion_estimator_footage=None,
    motion_estimator_reference=None,
):
    """
    Get a transformation to relate the coordinate transformations between footage absolute frame (first image in footage) and reference absolute frame (first image in reference).

    UI usage:

        The UI has the puropose of annotating points that match in the reference and the footage (either images or videos), to estimate a transformation.

        To add a point, just click a pair of points (one from the footage window, and another from the reference window) and select `"Add"`.
        To remove a point, just select the corresponding point at the bottom left corner, and select `"Remove"`.
        You can also ignore points, by clicking them and selecting `"Ignore"`. The transformation will not used ingored points.
        To 'uningnore' points that have been previously ignored, just click them and select `"Unignore"`.

        To resize the footage or the reference image, you can use the `"+"` and `"-"` buttons in the `'Resize footage'` and `'Resize reference'` sections of the Menu.

        If either footage or reference are videos, you can jump to future frames to pick points that match.
        For example, to jump 215 frames in the footage, just write that number next to `'Frames to skip (footage)'`, and select `"Skip frames"`.

        You can go back to the first frame of the video (in either footage or reference) by selecting "Reset video".

        Once a transformation has been estimated (you will know that if the `"Finished"` button is green), you can test it:
        To Test your transformation, Select the `"Test"` mode, and pick a point in either the reference or the footage, and see the associated point in the other window.
        You can go back to the `"Annotate"` mode keep adding more associated points until you are satisfied with the estimated transformation.

        You can also save the state (points and transformation you have) to a `.pkl` file using the `"Save"` button, so that you can later load that state from the UI with the `"Load"` button.

        You can swap the reference points with the footage points (inverting the transformation) with the `"Invert"` button. This is particularly useful if you have previously saved a state in which the reference was the current footage, and the footage was the current reference.

        Once you are happy with the transformation, just click on `"Finished"`.

    Argumentsco:
     - reference: str
        Path to the reference image or video

     - footage: str
        Path to the footage image or video

     - transformation_getter: TransformationGetter, optional
        TransformationGetter defining the type of transformation you want to fix between reference and footage.
        Since the transformation can be really far from identity (given that the perspectives in footage and reference can be immensely different),
        and also knowing that outliers shouldn't be common given that a human is picking the points, it is recommended to use a high ransac_reproj_threshold (~ 1000)

     - mask_generator: optional function that creates a mask (np.ndarray) from a PIL image. This mask is then provided to the corresponding MotionEstimator to avoid
        sampling points within the mask.

     - image_width: int, optional
        Width of the image of the UI. If the height is not provided, then it will be calculated so that the aspect ratio is preserved.

     - image_height: int, optional
        Height of the image of the UI. If the width is not provided, then it will be calculated so that the aspect ratio is preserved.

     - motion_estimator_footage: MotionEstimator, optional
        When using videos for the footage, you can provide a MotionEstimator to relate the coordinates in all the frames in the video.
        The motion estimator is only useful if the camera in the video of the footage can move. Otherwise, avoid using it.

     - motion_estimator_reference: MotionEstimator, optional
        When using videos the reference, you can provide a MotionEstimator to relate the coordinates in all the frames in the video.
        The motion estimator is only useful if the camera in the video of the reference can move. Otherwise, avoid using it.

     returns: CoordinatesTransformation instance
        The provided transformation_getter will fit a transformation from the reference (as 'absolute') to the footage (as 'relative').
        CoordinatesTransformation.abs_to_rel will give the transformation from the first frame in the reference to the first frame in the footage.
        CoordinatesTransformation.rel_to_abs will give the transformation from the first frame in the footage to the first frame in the reference.
    """

    global window

    global transformation

    global button_finish
    global button_says_ignore
    global button_ignore

    global reference_point_canvas
    global footage_point_canvas

    global canvas_reference
    global canvas_footage

    global reference_original_size
    global reference_canvas_size
    global footage_original_size
    global footage_canvas_size

    global footage_point
    global reference_point

    global skipper

    global points
    global points_sampled

    global mode_annotate

    global frame_options_annotations
    global handling_mark_functions
    global handle_mark_annotation

    if transformation_getter is None:
        transformation_getter = HomographyTransformationGetter(
            method=cv2.RANSAC,
            ransac_reproj_threshold=1000,
            max_iters=2000,
            confidence=0.995,
            proportion_points_used_threshold=0.9,
        )

    skipper = {}

    radius = None
    if (image_width is None) and (image_height is None):
        image_height = 450
    elif (image_width is not None) and (image_height is None):
        radius = max(int(image_width / 100), 1)
    if radius is None:
        radius = max(int(image_height / 100), 1)

    points = {}
    points_sampled = len(points)

    transformation = None

    window = tk.Tk()
    window.title("Norfair - Set Reference Coordinates")
    window.configure(bg="LightSkyBlue1")

    frame_options = tk.Frame()
    frame_images = tk.Frame()
    frame_options_annotations = tk.Frame(master=frame_options)

    # utilities

    def estimate_transformation(points):
        global button_finish
        prev_pts = np.array(
            [point["reference"] for point in points.values() if not point["ignore"]]
        )  # use current points as reference points
        curr_pts = np.array(
            [point["footage"] for point in points.values() if not point["ignore"]]
        )  # use previous points as footage points (to deduce reference -> footage)

        button_finish.configure(fg="black", highlightbackground="green")
        try:
            transformation = transformation_getter(curr_pts, prev_pts)[1]
        except:
            transformation = None

        if transformation is not None:
            button_finish.configure(fg="black", highlightbackground="green")
        else:
            button_finish.configure(fg="grey", highlightbackground="SystemButtonFace")
        return transformation

    def test_transformation(
        change_of_coordinates,
        canvas,
        point,
        original_size,
        canvas_size,
        motion_transformation=None,
    ):
        point_in_new_coordinates = change_of_coordinates(np.array([point]))[0]

        try:
            point_in_new_coordinates = motion_transformation.rel_to_abs(
                np.array([point_in_new_coordinates])
            )[0]
        except AttributeError:
            pass

        point_in_canvas_coordinates = np.multiply(
            point_in_new_coordinates,
            np.array(
                [canvas_size[0] / original_size[0], canvas_size[1] / original_size[1]]
            ),
        ).astype(int)

        draw_point_in_canvas(canvas, point_in_canvas_coordinates, "blue")

    def remove_drawings_in_canvas(canvas):
        if len(canvas.find_withtag("myPoint")) > 0:
            canvas.delete("myPoint")

    def draw_point_in_canvas(canvas, point, color="green"):
        remove_drawings_in_canvas(canvas)
        canvas.create_oval(
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
            fill=color,
            tags="myPoint",
        )

    ######### MAKE SUBBLOCK TO FINISH

    frame_options_finish = tk.Frame(master=frame_options)

    space = tk.Label(
        master=frame_options_finish,
        text="",
        foreground="white",
        width=40,
        height=1,
    )
    button_finish = tk.Button(
        master=frame_options_finish,
        text="Finished!",
        width=30,
        height=1,
        bg="blue",
        fg="gray",
        command=lambda: handle_finish(),
    )

    def handle_finish():
        global window
        global transformation

        if transformation is not None:
            window.destroy()
            for info in skipper.values():
                if info["video"] is not None:
                    info["video"].video_capture.release()
            cv2.destroyAllWindows()
            return transformation
        else:
            print("Can't leave without estimating the transformation.")

    space.pack(side=tk.TOP)
    button_finish.pack(side=tk.TOP)
    frame_options_finish.pack(side=tk.BOTTOM)

    ###### MAKE SUBBLOCK TO SEE POINTS AND CHOOSE THEM
    def handle_mark_annotation(key):
        def handle_annotation(event):
            global skipper
            global reference_original_size
            global reference_canvas_size
            global footage_original_size
            global footage_canvas_size
            global button_says_ignore
            global button_ignore
            global points

            points[key]["marked"] = not points[key]["marked"]

            marked_points = [
                point["ignore"] for point in points.values() if point["marked"]
            ]

            if (len(marked_points) > 0) and (all(marked_points)):
                button_says_ignore = False
                button_ignore.configure(text="Unignore")
            else:
                button_says_ignore = True
                button_ignore.configure(text="Ignore")

            if points[key]["marked"]:
                points[key]["button"].configure(fg="black", highlightbackground="red")

                try:
                    footage_point_in_rel_coords = skipper["footage"][
                        "motion_transformation"
                    ].rel_to_abs(np.array([points[key]["footage"]]))[0]
                except AttributeError:
                    footage_point_in_rel_coords = points[key]["footage"]
                footage_point_in_rel_coords = np.multiply(
                    footage_point_in_rel_coords,
                    np.array(
                        [
                            footage_canvas_size[0] / footage_original_size[0],
                            footage_canvas_size[1] / footage_original_size[1],
                        ]
                    ),
                ).astype(int)

                try:
                    reference_point_in_rel_coords = skipper["reference"][
                        "motion_transformation"
                    ].rel_to_abs(np.array([points[key]["reference"]]))[0]
                except AttributeError:
                    reference_point_in_rel_coords = points[key]["reference"]

                reference_point_in_rel_coords = np.multiply(
                    reference_point_in_rel_coords,
                    np.array(
                        [
                            reference_canvas_size[0] / reference_original_size[0],
                            reference_canvas_size[1] / reference_original_size[1],
                        ]
                    ),
                ).astype(int)

                if points[key]["ignore"]:
                    color = "gray"
                else:
                    color = "red"

                draw_point_in_canvas(
                    canvas_footage, footage_point_in_rel_coords, color=color
                )
                draw_point_in_canvas(
                    canvas_reference, reference_point_in_rel_coords, color=color
                )
            else:
                if points[key]["ignore"]:
                    points[key]["button"].configure(
                        fg="gray", highlightbackground="gray"
                    )
                else:
                    points[key]["button"].configure(
                        fg="black", highlightbackground="SystemButtonFace"
                    )

                points[key]["button"].configure(
                    fg="black", highlightbackground="SystemButtonFace"
                )
                canvas_footage.delete("myPoint")
                canvas_reference.delete("myPoint")

        return handle_annotation

    handling_mark_functions = {}
    for key, couple in points.items():

        handling_mark_functions[key] = handle_mark_annotation(key)

        new_button = tk.Button(
            master=frame_options_annotations,
            text=f"{key}: reference ({couple['reference'][0]}, {couple['reference'][1]}) <-> footage ({couple['footage'][0]}, {couple['footage'][1]})",
            width=35,
            height=1,
            bg="blue",
            fg="black",
            highlightbackground="SystemButtonFace",
        )

        new_button.bind("<Button>", handling_mark_functions[key])

        new_button.pack(side=tk.TOP)
        points[key]["button"] = new_button

    frame_options_annotations.pack(side=tk.BOTTOM)

    ######## Add clickable windows for the frames

    reference_point = None
    reference_point_canvas = None
    footage_point = None
    footage_point_canvas = None

    motion_transformation = None
    motion_estimator_backup = None
    try:
        image = Image.open(footage)
        video = None
        fps = None
        total_frames = None
    except UnidentifiedImageError:
        video = Video(input_path=footage)
        total_frames = int(video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.output_fps
        image = cv2.cvtColor(next(video.__iter__()), cv2.COLOR_BGR2RGB)
        if motion_estimator_footage is not None:
            motion_estimator_backup = deepcopy(motion_estimator_footage)
            if mask_generator is not None:
                mask = mask_generator(image)
            else:
                mask = None
            motion_transformation = motion_estimator_footage.update(image, mask)
        image = Image.fromarray(image)

    footage_original_size = (image.width, image.height)

    resized_image = resize_image(
        image, desired_width=image_width, desired_height=image_height
    )

    footage_photo = ImageTk.PhotoImage(resized_image)
    footage_canvas_size = (footage_photo.width(), footage_photo.height())

    canvas_footage = tk.Canvas(
        frame_images,
        width=footage_canvas_size[0],
        height=footage_canvas_size[1],
        bg="gray",
    )
    footage_image_container = canvas_footage.create_image(
        0, 0, anchor=tk.NW, image=footage_photo
    )

    def reference_coord_chosen_in_footage(event):
        global footage_point
        global footage_point_canvas
        global transformation
        global canvas_reference
        global reference_original_size
        global reference_canvas_size
        global footage_original_size
        global footage_canvas_size
        global skipper

        footage_point_canvas = (event.x, event.y)
        draw_point_in_canvas(canvas_footage, footage_point_canvas)

        footage_point = np.array(
            [
                event.x * (footage_original_size[0] / footage_canvas_size[0]),
                event.y * (footage_original_size[1] / footage_canvas_size[1]),
            ]
        )
        print("Footage window clicked at: ", footage_point.round(1))

        try:
            footage_point = skipper["footage"]["motion_transformation"].abs_to_rel(
                np.array([footage_point])
            )[0]
        except AttributeError:
            pass

        footage_point = footage_point.round(1)

        if not mode_annotate:
            if transformation is not None:
                test_transformation(
                    transformation.rel_to_abs,
                    canvas_reference,
                    footage_point,
                    reference_original_size,
                    reference_canvas_size,
                    skipper["reference"]["motion_transformation"],
                )
            else:
                print("Can't test the transformation yet, still need more points")

    canvas_footage.bind("<Button>", reference_coord_chosen_in_footage)
    canvas_footage.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    skipper["footage"] = {
        "video": video,
        "total_frames": total_frames,
        "current_frame": 1,
        "fps": fps,
        "button_skip": None,
        "entry_skip": None,
        "button_reset": None,
        "motion_estimator": motion_estimator_footage,
        "motion_transformation": motion_transformation,
        "motion_estimator_backup": motion_estimator_backup,
        "canvas": canvas_footage,
        "image_container": footage_image_container,
        "current_frame_label": None,
        "path": footage,
        "original_image": image,
    }

    motion_transformation = None
    motion_estimator_backup = None
    try:
        image = Image.open(reference)
        video = None
        fps = None
        total_frames = None
    except UnidentifiedImageError:
        video = Video(input_path=reference)
        total_frames = int(video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.output_fps
        image = cv2.cvtColor(next(video.__iter__()), cv2.COLOR_BGR2RGB)
        if motion_estimator_reference is not None:
            motion_estimator_backup = deepcopy(motion_estimator_reference)
            if mask_generator is not None:
                mask = mask_generator(image)
            else:
                mask = None
            motion_transformation = motion_estimator_reference.update(image, mask)

        image = Image.fromarray(image)

    reference_original_size = (image.width, image.height)
    resized_image = resize_image(
        image, desired_width=image_width, desired_height=image_height
    )

    reference_photo = ImageTk.PhotoImage(resized_image)
    reference_canvas_size = (reference_photo.width(), reference_photo.height())

    canvas_reference = tk.Canvas(
        frame_images,
        width=reference_canvas_size[0],
        height=reference_canvas_size[1],
        bg="gray",
    )
    reference_image_container = canvas_reference.create_image(
        0, 0, anchor=tk.NW, image=reference_photo
    )

    def reference_coord_chosen_in_reference(event):
        global reference_point
        global reference_point_canvas
        global transformation
        global canvas_footage
        global reference_original_size
        global reference_canvas_size
        global footage_original_size
        global footage_canvas_size
        global skipper

        reference_point_canvas = (event.x, event.y)
        draw_point_in_canvas(canvas_reference, reference_point_canvas)

        reference_point = np.array(
            [
                event.x * (reference_original_size[0] / reference_canvas_size[0]),
                event.y * (reference_original_size[1] / reference_canvas_size[1]),
            ]
        )
        print("Reference window clicked at: ", reference_point.round(1))

        try:
            reference_point = skipper["reference"]["motion_transformation"].abs_to_rel(
                np.array([reference_point])
            )[0]
        except AttributeError:
            pass

        reference_point = reference_point.round(1)

        if not mode_annotate:
            if transformation is not None:
                test_transformation(
                    transformation.abs_to_rel,
                    canvas_footage,
                    reference_point,
                    footage_original_size,
                    footage_canvas_size,
                    skipper["footage"]["motion_transformation"],
                )
            else:
                print("Can't test the transformation yet, still need more points")

    canvas_reference.bind("<Button>", reference_coord_chosen_in_reference)
    canvas_reference.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    skipper["reference"] = {
        "video": video,
        "total_frames": total_frames,
        "current_frame": 1,
        "fps": fps,
        "button_skip": None,
        "entry_skip": None,
        "button_reset": None,
        "motion_estimator": motion_estimator_reference,
        "motion_transformation": motion_transformation,
        "motion_estimator_backup": motion_estimator_backup,
        "canvas": canvas_reference,
        "image_container": reference_image_container,
        "current_frame_label": None,
        "path": reference,
        "original_image": image,
    }
    ######### MAKE SUBBLOCK FOR LOGO

    frame_options_logo = tk.Frame(master=frame_options)
    image = Image.open(
        os.path.join(os.path.dirname(__file__), "./logo_ui/logo-dark.png")
    )
    image = image.resize((300, 70))
    image = ImageTk.PhotoImage(image)

    image_label = tk.Label(
        frame_options_logo, image=image, width=40, height=60, bg="LightSkyBlue1"
    )

    image_label.pack(side=tk.TOP, fill="both", expand="yes")
    frame_options_logo.pack(side=tk.TOP, fill="both", expand="yes")

    ###### MAKE SUBBLOCK FOR TITLE
    frame_options_title = tk.Frame(master=frame_options)
    title = tk.Label(
        master=frame_options_title,
        text="Options",
        foreground="white",
        background="#34A2FE",
        width=40,
        height=1,
    )
    title.pack(side=tk.TOP)
    frame_options_title.pack(side=tk.TOP)

    ############### VIDEO CATEGORY SUBBLOCKS (FPS, TOTAL FRAMES, SKIP FRAMES, RESET VIDEO)

    def get_reset_video_handler(video_type):
        def handle_reset_video(event):
            global skipper
            global canvas_footage
            global canvas_reference
            global reference_canvas_size
            global footage_canvas_size

            if skipper[video_type]["current_frame"] > 1:
                skipper[video_type]["video"].video_capture.release()
                cv2.destroyAllWindows()
                video = Video(input_path=skipper[video_type]["path"])
                image = cv2.cvtColor(next(video.__iter__()), cv2.COLOR_BGR2RGB)
                skipper[video_type]["video"] = video
                if skipper[video_type]["motion_estimator"] is not None:
                    skipper[video_type]["motion_estimator"] = deepcopy(
                        skipper[video_type]["motion_estimator_backup"]
                    )
                    if mask_generator is not None:
                        mask = mask_generator(image)
                    else:
                        mask = None
                    skipper[video_type]["motion_transformation"] = skipper[video_type][
                        "motion_estimator"
                    ].update(image, mask)
                skipper[video_type]["current_frame"] = 1
                image = Image.fromarray(image)
                skipper[video_type]["original_image"] = image

                if video_type == "reference":
                    size = reference_canvas_size
                else:
                    size = footage_canvas_size
                image = resize_image(
                    image, desired_width=size[0], desired_height=size[1]
                )
                image = ImageTk.PhotoImage(image)

                skipper[video_type]["canvas"].itemconfig(
                    skipper[video_type]["image_container"], image=image
                )
                skipper[video_type]["canvas"].imgref = image
                canvas_footage.delete("myPoint")
                canvas_reference.delete("myPoint")

                skipper[video_type]["current_frame_label"].config(
                    text=f"Total frames {video_type}: 1/{skipper[video_type]['total_frames']}"
                )

        return handle_reset_video

    def get_skiper_handler(video_type):
        entry_skip = skipper[video_type]["entry_skip"]

        def handle_skip_frame(event):
            global skipper
            global canvas_footage
            global canvas_reference
            global reference_canvas_size
            global footage_canvas_size

            try:
                frames_to_skip = int(entry_skip.get())
            except:
                print(
                    f"Frames to skip has to be an integer, but you wrote '{entry_skip.get()}'"
                )
                return
            video = skipper[video_type]["video"]
            change_image = False
            motion_estimator = skipper[video_type]["motion_estimator"]
            motion_transformation = skipper[video_type]["motion_transformation"]

            while (frames_to_skip > 0) and (
                skipper[video_type]["current_frame"]
                < skipper[video_type]["total_frames"]
            ):
                change_image = True
                frames_to_skip -= 1
                skipper[video_type]["current_frame"] += 1

                image = next(video.__iter__())

                if motion_estimator is not None:
                    if mask_generator is not None:
                        mask = mask_generator(image)
                    else:
                        mask = None
                    motion_transformation = motion_estimator.update(
                        np.array(image), mask
                    )

            skipper[video_type]["motion_estimator"] = motion_estimator
            skipper[video_type]["motion_transformation"] = motion_transformation

            if change_image:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                skipper[video_type]["original_image"] = image
                if video_type == "reference":
                    size = reference_canvas_size
                else:
                    size = footage_canvas_size
                image = resize_image(
                    image, desired_width=size[0], desired_height=size[1]
                )
                image = ImageTk.PhotoImage(image)

                skipper[video_type]["canvas"].itemconfig(
                    skipper[video_type]["image_container"], image=image
                )
                skipper[video_type]["canvas"].imgref = image

                frame_number = skipper[video_type]["current_frame"]
                total_frames = skipper[video_type]["total_frames"]

                skipper[video_type]["current_frame_label"].config(
                    text=f"Total frames {video_type}: {frame_number}/{total_frames}"
                )
                canvas_footage.delete("myPoint")
                canvas_reference.delete("myPoint")

        return handle_skip_frame

    skiper_handlers = {}
    reset_video_handlers = {}
    for video_type in skipper.keys():

        if skipper[video_type]["fps"] is not None:
            ###### MAKE SUBBLOCK FOR FPS

            fps = skipper[video_type]["fps"]

            frame_options_fps = tk.Frame(master=frame_options)
            title = tk.Label(
                master=frame_options_fps,
                text=f"FPS {video_type}: {round(fps, 2)}",
                foreground="white",
                background="#34A2FE",
                width=40,
                height=1,
            )
            title.pack(side=tk.TOP)
            frame_options_fps.pack(side=tk.TOP)

            ###### MAKE SUBBLOCK FOR TOTAL FRAMES
            frame_number = skipper[video_type]["current_frame"]
            total_frames = skipper[video_type]["total_frames"]

            frame_options_total_frames = tk.Frame(master=frame_options)
            title = tk.Label(
                master=frame_options_total_frames,
                text=f"Total frames {video_type}: {frame_number}/{total_frames}",
                foreground="white",
                background="#34A2FE",
                width=40,
                height=1,
            )
            title.pack(side=tk.TOP)
            frame_options_total_frames.pack(side=tk.TOP)
            skipper[video_type]["current_frame_label"] = title

            ###### MAKE SUBBLOCK FOR SKIP FRAMES
            frame_options_skip = tk.Frame(master=frame_options)
            text_skip_frames = tk.Label(
                master=frame_options_skip,
                text=f"Frames to skip ({video_type})",
                foreground="white",
                background="#5f9ea0",
                width=20,
                height=1,
            )
            entry_skip = tk.Entry(
                master=frame_options_skip, fg="black", bg="white", width=5
            )
            skipper[video_type]["entry_skip"] = entry_skip
            skiper_handlers[video_type] = get_skiper_handler(video_type)

            button_skip = tk.Button(
                master=frame_options_skip,
                text="Skip frames",
                width=10,
                height=1,
                bg="blue",
                fg="black",
            )

            button_skip.bind("<Button>", skiper_handlers[video_type])

            skipper[video_type]["button_skip"] = button_skip

            text_skip_frames.pack(side=tk.LEFT)
            entry_skip.pack(side=tk.LEFT)
            button_skip.pack(side=tk.LEFT)

            frame_options_skip.pack(side=tk.TOP)

            ###### MAKE SUBBLOCK TO RESET VIDEO
            frame_options_reset_video = tk.Frame(master=frame_options)
            text_reset_video = tk.Label(
                master=frame_options_reset_video,
                text=f"Go to frame 1 ({video_type})",
                foreground="white",
                background="#5f9ea0",
                width=20,
                height=1,
            )

            reset_video_handlers[video_type] = get_reset_video_handler(video_type)

            button_reset = tk.Button(
                master=frame_options_reset_video,
                text="Reset video",
                width=16,
                height=1,
                bg="blue",
                fg="black",
            )

            button_reset.bind("<Button>", reset_video_handlers[video_type])

            skipper[video_type]["button_reset"] = button_reset

            text_reset_video.pack(side=tk.LEFT)
            button_reset.pack(side=tk.LEFT)

            frame_options_reset_video.pack(side=tk.TOP)

    ###### MAKE SUBBLOCK TO ADD ANNOTATION

    frame_options_add = tk.Frame(master=frame_options)
    text_add_annotation = tk.Label(
        master=frame_options_add,
        text="Add annotation",
        foreground="white",
        background="#5f9ea0",
        width=20,
        height=1,
    )
    button_add_annotation = tk.Button(
        master=frame_options_add,
        text="Add",
        width=16,
        height=1,
        bg="blue",
        fg="black",
        command=lambda: handle_add_annotation(),
    )

    def create_annotation_button(point):
        global points_sampled
        global frame_options_annotations
        global handling_mark_functions
        global handle_mark_annotation

        handling_mark_functions[points_sampled] = handle_mark_annotation(points_sampled)

        new_button = tk.Button(
            master=frame_options_annotations,
            text=f"{points_sampled}: reference ({point['reference'][0]}, {point['reference'][1]}) <-> footage ({point['footage'][0]}, {point['footage'][1]})",
            width=35,
            height=1,
            bg="blue",
            fg="black",
            highlightbackground="SystemButtonFace",
        )

        new_button.bind("<Button>", handling_mark_functions[points_sampled])
        new_button.pack(side=tk.TOP)

        canvas_footage.delete("myPoint")
        canvas_reference.delete("myPoint")
        return new_button

    def handle_add_annotation():
        global points
        global points_sampled
        global reference_point
        global footage_point
        global mode_annotate
        global transformation

        if mode_annotate:
            if (footage_point is not None) and (reference_point is not None):
                new_point = {
                    "reference": reference_point,
                    "footage": footage_point,
                    "button": None,
                    "marked": False,
                    "ignore": False,
                }
                points[points_sampled] = new_point

                points[points_sampled]["button"] = create_annotation_button(new_point)

                reference_point = None
                footage_point = None
                points_sampled += 1

                transformation = estimate_transformation(points)
            else:
                print(
                    "Need to pick a point from the footage and from the reference to annotate them"
                )
        else:
            print("Can't annotate in 'Test' mode.")

    text_add_annotation.pack(side=tk.LEFT)
    button_add_annotation.pack(side=tk.LEFT)

    frame_options_add.pack(side=tk.TOP)

    ###### MAKE SUBBLOCK TO LOAD OR SAVE TRANSFORMATION

    frame_options_files = tk.Frame(master=frame_options)
    text_files = tk.Label(
        master=frame_options_files,
        text="Save/Load state",
        foreground="white",
        background="#5f9ea0",
        width=20,
        height=1,
    )
    button_save_state = tk.Button(
        master=frame_options_files,
        text="Save",
        width=6,
        height=1,
        bg="blue",
        fg="black",
        command=lambda: handle_save_state(),
    )
    button_load_state = tk.Button(
        master=frame_options_files,
        text="Load",
        width=6,
        height=1,
        bg="blue",
        fg="black",
        command=lambda: handle_load_state(),
    )

    def handle_save_state():
        global points
        global transformation
        global window

        initialfile = f"{os.path.splitext(os.path.basename(reference))[0]}_to_{os.path.splitext(os.path.basename(footage))[0]}.pkl"

        file = tkinter.filedialog.asksaveasfile(
            parent=window, initialfile=initialfile, mode="wb", defaultextension=".pkl"
        )
        if file is not None:
            points_reference = []
            points_footage = []
            is_ignored = []
            for point in points.values():
                points_reference.append(point["reference"])
                points_footage.append(point["footage"])
                is_ignored.append(point["ignore"])
            pickle.dump(
                {
                    "reference": points_reference,
                    "footage": points_footage,
                    "ignore": is_ignored,
                    "transformation": transformation,
                },
                file,
            )
            file.close()

    def handle_load_state():
        global points
        global transformation
        global points_sampled
        global window
        global button_finish
        global reference_point
        global footage_point
        global canvas_footage
        global canvas_reference

        initialfile = f"{os.path.splitext(os.path.basename(reference))[0]}_to_{os.path.splitext(os.path.basename(footage))[0]}.pkl"

        file = tkinter.filedialog.askopenfile(
            parent=window, initialfile=initialfile, mode="rb"
        )
        if file is not None:

            loaded_state = pickle.load(file)

            points_reference = loaded_state["reference"]
            points_footage = loaded_state["footage"]
            transformation = loaded_state["transformation"]
            is_ignored = loaded_state["ignore"]

            if transformation is not None:
                button_finish.configure(fg="black", highlightbackground="green")
            else:
                button_finish.configure(
                    fg="grey", highlightbackground="SystemButtonFace"
                )

            points = recreate_all_buttons(points_reference, points_footage, is_ignored)

            remove_drawings_in_canvas(canvas_footage)
            remove_drawings_in_canvas(canvas_reference)

            reference_point = None
            footage_point = None

            file.close()

    text_files.pack(side=tk.LEFT)
    button_save_state.pack(side=tk.LEFT)
    button_load_state.pack(side=tk.LEFT)
    frame_options_files.pack(side=tk.TOP)

    def recreate_all_buttons(points_reference, points_footage, is_ignored):
        global points
        global points_sampled

        # remove existing buttons
        for point in points.values():
            point["button"].destroy()
        points = {}

        points_sampled = 0
        # create new points
        for reference_point, footage_point, ignore in zip(
            points_reference, points_footage, is_ignored
        ):

            new_point = {
                "reference": reference_point,
                "footage": footage_point,
                "button": None,
                "marked": False,
                "ignore": ignore,
            }
            points[points_sampled] = new_point
            points[points_sampled]["button"] = create_annotation_button(new_point)
            if ignore:
                points[points_sampled]["button"].configure(
                    fg="gray", highlightbackground="gray"
                )

            points_sampled += 1

        return points

    ###### MAKE SUBBLOCK TO CHANGE BETWEEN ANOTATE AND TEST
    frame_options_annotate_or_test = tk.Frame(master=frame_options)

    mode_annotate = True

    text_annotate_or_test = tk.Label(
        master=frame_options_annotate_or_test,
        text="Mode:",
        foreground="white",
        background="#5f9ea0",
        width=20,
        height=1,
    )
    button_annotate = tk.Button(
        master=frame_options_annotate_or_test,
        text="Annotate",
        width=6,
        height=1,
        bg="blue",
        fg="black",
        highlightbackground="green",
    )
    button_test = tk.Button(
        master=frame_options_annotate_or_test,
        text="Test",
        width=6,
        height=1,
        bg="blue",
        fg="black",
        highlightbackground="red",
    )

    def handle_annotate_mode(event):
        global mode_annotate
        mode_annotate = True
        button_test.configure(fg="black", highlightbackground="red")
        button_annotate.configure(fg="black", highlightbackground="green")

    def handle_test_mode(event):
        global mode_annotate
        mode_annotate = False
        button_test.configure(fg="black", highlightbackground="green")
        button_annotate.configure(fg="black", highlightbackground="red")

    button_annotate.bind("<Button>", handle_annotate_mode)
    button_test.bind("<Button>", handle_test_mode)

    text_annotate_or_test.pack(side=tk.LEFT)
    button_annotate.pack(side=tk.LEFT)
    button_test.pack(side=tk.LEFT)

    frame_options_annotate_or_test.pack(side=tk.TOP)

    ###### MAKE SUBBLOCK TO IGNORE POINTS

    button_says_ignore = True
    frame_options_ignore = tk.Frame(master=frame_options)
    text_ignore = tk.Label(
        master=frame_options_ignore,
        text="Ignore points",
        foreground="white",
        background="#5f9ea0",
        width=20,
        height=1,
    )
    button_ignore = tk.Button(
        master=frame_options_ignore,
        text="Ignore",
        width=16,
        height=1,
        bg="blue",
        fg="black",
        command=lambda: handle_ignore_point(),
    )

    def handle_ignore_point():
        global points
        global transformation
        global button_says_ignore

        if button_says_ignore:
            fg = "gray"
            highlightbackground = "gray"
        else:
            fg = "black"
            highlightbackground = "SystemButtonFace"

        for key, couple in points.items():
            if couple["marked"]:
                points[key]["ignore"] = button_says_ignore
                points[key]["button"].configure(
                    fg=fg, highlightbackground=highlightbackground
                )
                points[key]["marked"] = False
                remove_drawings_in_canvas(canvas_footage)
                remove_drawings_in_canvas(canvas_reference)
        button_says_ignore = True
        button_ignore.configure(text="Ignore")
        transformation = estimate_transformation(points)

    text_ignore.pack(side=tk.LEFT)
    button_ignore.pack(side=tk.LEFT)

    frame_options_ignore.pack(side=tk.TOP)

    ###### MAKE SUBBLOCK TO REMOVE POINTS

    frame_options_remove = tk.Frame(master=frame_options)
    text_remove = tk.Label(
        master=frame_options_remove,
        text="Remove points",
        foreground="white",
        background="#5f9ea0",
        width=20,
        height=1,
    )
    button_remove = tk.Button(
        master=frame_options_remove,
        text="Remove",
        width=16,
        height=1,
        bg="blue",
        fg="black",
        command=lambda: handle_remove_point(),
    )

    def handle_remove_point():
        global points
        global transformation
        points_copy = points.copy()
        for key, couple in points.items():
            if couple["marked"]:
                points_copy.pop(key)
                couple["button"].destroy()

                remove_drawings_in_canvas(canvas_footage)
                remove_drawings_in_canvas(canvas_reference)
        points = points_copy.copy()

        transformation = estimate_transformation(points)

    text_remove.pack(side=tk.LEFT)
    button_remove.pack(side=tk.LEFT)

    frame_options_remove.pack(side=tk.TOP)

    ######  MAKE SUBBLOCK TO INVERT TRANSFORMATION

    frame_options_invert = tk.Frame(master=frame_options)
    text_invert = tk.Label(
        master=frame_options_invert,
        text="Invert (pts and transf)",
        foreground="white",
        background="#5f9ea0",
        width=20,
        height=1,
    )
    button_invert = tk.Button(
        master=frame_options_invert,
        text="Invert",
        width=16,
        height=1,
        bg="blue",
        fg="black",
        command=lambda: handle_invert(),
    )

    def handle_invert():
        global points
        global transformation
        global points_sampled
        global footage_point
        global reference_point
        global canvas_footage
        global canvas_reference

        points_reference = []
        points_footage = []
        is_ignored = []

        for key, couple in points.items():
            # swap footage coordinate with reference
            points_reference.append(couple["footage"])
            points_footage.append(couple["reference"])
            is_ignored.append(couple["ignore"])

        remove_drawings_in_canvas(canvas_footage)
        remove_drawings_in_canvas(canvas_reference)

        points = recreate_all_buttons(points_reference, points_footage, is_ignored)
        points_sampled = len(points)

        reference_point = None
        footage_point = None

        transformation = estimate_transformation(points)

    text_invert.pack(side=tk.LEFT)
    button_invert.pack(side=tk.LEFT)

    frame_options_invert.pack(side=tk.TOP)

    ######  MAKE SUBBLOCK TO RESIZE FRAMES

    def get_handle_resize(video_type, delta_height):
        def handle_resize(event):
            global footage_canvas_size
            global footage_original_size
            global reference_canvas_size
            global reference_original_size
            global skipper

            if video_type == "reference":
                new_canvas_height = max(reference_canvas_size[1] + delta_height, 10)
                new_canvas_width = int(
                    new_canvas_height
                    * reference_canvas_size[0]
                    / reference_canvas_size[1]
                )
                reference_canvas_size = (new_canvas_width, new_canvas_height)
            else:
                new_canvas_height = max(footage_canvas_size[1] + delta_height, 10)
                new_canvas_width = int(
                    new_canvas_height * footage_canvas_size[0] / footage_canvas_size[1]
                )
                footage_canvas_size = (new_canvas_width, new_canvas_height)

            image = resize_image(
                skipper[video_type]["original_image"],
                desired_width=new_canvas_width,
                desired_height=new_canvas_height,
            )
            image = ImageTk.PhotoImage(image)
            skipper[video_type]["canvas"].itemconfig(
                skipper[video_type]["image_container"], image=image
            )
            skipper[video_type]["canvas"].imgref = image
            skipper[video_type]["canvas"].delete("myPoint")

        return handle_resize

    resizer = {}

    for video_type in skipper.keys():

        frame_options_resize = tk.Frame(master=frame_options)

        resizer[video_type] = {}
        resizer[video_type]["handle_increase"] = get_handle_resize(video_type, 10)
        resizer[video_type]["handle_decrease"] = get_handle_resize(video_type, -10)

        text = tk.Label(
            master=frame_options_resize,
            text=f"Resize {video_type}",
            foreground="white",
            background="#5f9ea0",
            width=20,
            height=1,
        )

        button_increase = tk.Button(
            master=frame_options_resize,
            text="+",
            width=6,
            height=1,
            bg="blue",
            fg="black",
        )
        button_decrease = tk.Button(
            master=frame_options_resize,
            text="-",
            width=6,
            height=1,
            bg="blue",
            fg="black",
        )

        button_increase.bind("<Button>", resizer[video_type]["handle_increase"])
        button_decrease.bind("<Button>", resizer[video_type]["handle_decrease"])

        text.pack(side=tk.LEFT)
        button_decrease.pack(side=tk.LEFT)
        button_increase.pack(side=tk.LEFT)

        frame_options_resize.pack(side=tk.TOP)

    ########## pack options with images

    frame_images.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    frame_options.pack(side=tk.LEFT)
    window.mainloop()

    return transformation
