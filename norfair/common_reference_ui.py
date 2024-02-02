# from tkinter import
import tkinter as tk
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageTk, UnidentifiedImageError

from norfair import Video
from norfair.camera_motion import HomographyTransformationGetter, TransformationGetter

transformation = None

window = None

button_finish = None

reference_point_canvas = None
footage_point_canvas = None

canvas_reference = None
canvas_footage = None

reference_original_size = None
reference_canvas_size = None
footage_original_size = None
footage_canvas_size = None

footage_point = None
reference_point = None

skipper = None

points = None
points_sampled = None

mode_annotate = None

frame_options_annotations = None
handling_mark_functions = None
handle_mark_annotation = None


def set_reference(
    reference: str,
    footage: str,
    transformation_getter: TransformationGetter = None,
    detector=None,
    desired_size=700,
    motion_estimator=None,
):

    global window

    global transformation

    global button_finish

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

    desired_size = 700
    radius = max(int(desired_size / 100), 1)

    points = {}
    points_sampled = len(points)

    transformation = None

    window = tk.Tk()

    frame_options = tk.Frame()
    frame_images = tk.Frame()
    frame_options_annotations = tk.Frame(master=frame_options)

    # utilities

    def estimate_transformation(points):
        global button_finish
        if len(points) >= 4:
            curr_pts = np.array(
                [point["reference"] for point in points.values()]
            )  # use current points as reference points
            prev_pts = np.array(
                [point["footage"] for point in points.values()]
            )  # use previous points as footage points (to deduce reference -> footage)

            try:

                button_finish.configure(fg="black", highlightbackground="green")
                return transformation_getter(curr_pts, prev_pts)[1]
            except np.linalg.LinAlgError:
                button_finish.configure(
                    fg="grey", highlightbackground="SystemButtonFace"
                )
                return None
        else:
            button_finish.configure(fg="grey", highlightbackground="SystemButtonFace")
            return None

    def test_transformation(
        change_of_coordinates, canvas, point, original_size, canvas_size
    ):
        point_in_new_coordinates = change_of_coordinates(np.array([point]))[0]
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

    ######### MAKSE SUBBLOCK TO FINISH

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
            points[key]["marked"] = not points[key]["marked"]

            if points[key]["marked"]:
                points[key]["button"].configure(fg="black", highlightbackground="red")
                draw_point_in_canvas(
                    canvas_footage, points[key]["footage_canvas"], color="red"
                )
                draw_point_in_canvas(
                    canvas_reference, points[key]["reference_canvas"], color="red"
                )
            else:
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
            text=f"{key}: reference {couple['reference']} <-> footage {couple['footage']}",
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

    try:
        image = Image.open(footage)
        video = None
        fps = None
        total_frames = None
    except UnidentifiedImageError:
        video = Video(input_path=footage)
        total_frames = int(video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.output_fps
        image = Image.fromarray(cv2.cvtColor(next(video.__iter__()), cv2.COLOR_BGR2RGB))

    footage_original_width = image.width
    footage_original_height = image.height
    footage_original_size = (footage_original_width, footage_original_height)

    image.thumbnail((desired_size, desired_size))
    footage_photo = ImageTk.PhotoImage(image)
    footage_canvas_width = footage_photo.width()
    footage_canvas_height = footage_photo.height()
    footage_canvas_size = (footage_canvas_width, footage_canvas_height)

    canvas_footage = tk.Canvas(
        frame_images, width=footage_photo.width(), height=footage_photo.height()
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
        footage_point = (
            np.around(event.x * (footage_original_width / footage_canvas_width), 1),
            np.around(event.y * (footage_original_height / footage_canvas_height), 1),
        )
        footage_point_canvas = (event.x, event.y)
        draw_point_in_canvas(canvas_footage, footage_point_canvas)
        print("Footage window clicked at: ", footage_point)

        if not mode_annotate:
            if transformation is not None:
                test_transformation(
                    transformation.abs_to_rel,
                    canvas_reference,
                    footage_point,
                    reference_original_size,
                    reference_canvas_size,
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
        "motion_estimator": None,
        "canvas": canvas_footage,
        "image_container": footage_image_container,
        "current_frame_label": None,
    }

    try:
        image = Image.open(reference)
        video = None
        fps = None
        total_frames = None
    except UnidentifiedImageError:
        video = Video(input_path=reference)
        total_frames = int(video.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.output_fps
        image = Image.fromarray(cv2.cvtColor(next(video.__iter__()), cv2.COLOR_BGR2RGB))

    reference_original_width = image.width
    reference_original_height = image.height
    reference_original_size = (reference_original_width, reference_original_height)

    image.thumbnail((desired_size, desired_size))
    reference_photo = ImageTk.PhotoImage(image)
    reference_canvas_width = reference_photo.width()
    reference_canvas_height = reference_photo.height()
    reference_canvas_size = (reference_canvas_width, reference_canvas_height)

    canvas_reference = tk.Canvas(
        frame_images, width=reference_photo.width(), height=reference_photo.height()
    )
    reference_image_container = canvas_reference.create_image(
        0, 0, anchor=tk.NW, image=reference_photo
    )

    def reference_coord_chosen_in_reference(event):
        global reference_point
        global reference_point_canvas
        global transformation
        global canvas_footage
        global footage_original_size
        global footage_canvas_size

        reference_point = (
            np.around(event.x * (reference_original_width / reference_canvas_width), 1),
            np.around(
                event.y * (reference_original_height / reference_canvas_height), 1
            ),
        )
        reference_point_canvas = (event.x, event.y)
        draw_point_in_canvas(canvas_reference, reference_point_canvas)
        print("Reference window clicked at: ", reference_point)

        if not mode_annotate:
            if transformation is not None:
                test_transformation(
                    transformation.rel_to_abs,
                    canvas_footage,
                    reference_point,
                    footage_original_size,
                    footage_canvas_size,
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
        "motion_estimator": None,
        "canvas": canvas_reference,
        "image_container": reference_image_container,
        "current_frame_label": None,
    }

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

    ############### VIDEO CATEGORY SUBBLOCKS (FPS, TOTAL FRAMES, SKIP FRAMES)

    def get_skiper_handler(video_type):
        entry_skip = skipper[video_type]["entry_skip"]

        def handle_skip_frame(event):
            global skipper

            try:
                frames_to_skip = int(entry_skip.get())
            except:
                print(
                    f"Frames to skip has to be an integer, but you wrote '{entry_skip.get()}'"
                )
                return
            video = skipper[video_type]["video"]
            change_image = False
            while (frames_to_skip > 0) and (
                skipper[video_type]["current_frame"]
                < skipper[video_type]["total_frames"]
            ):
                change_image = True
                frames_to_skip -= 1
                skipper[video_type]["current_frame"] += 1

                image = next(video.__iter__())

            if change_image:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image.thumbnail((desired_size, desired_size))
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

        return handle_skip_frame

    skiper_handlers = {}
    for video_type in skipper.keys():

        if skipper[video_type]["fps"] is not None:
            ###### MAKE SUBBLOCK FOR FPS

            fps = skipper[video_type]["fps"]

            frame_options_fps = tk.Frame(master=frame_options)
            title = tk.Label(
                master=frame_options_fps,
                text=f"FPS {video_type}: {fps}",
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

    def handle_add_annotation():
        global points
        global points_sampled
        global reference_point
        global footage_point
        global mode_annotate
        global frame_options_annotations
        global handling_mark_functions
        global handle_mark_annotation
        global transformation

        if mode_annotate:
            if (footage_point is not None) and (reference_point is not None):
                new_point = {
                    "reference": reference_point,
                    "footage": footage_point,
                    "reference_canvas": reference_point_canvas,
                    "footage_canvas": footage_point_canvas,
                    "button": None,
                    "marked": False,
                }
                points[points_sampled] = new_point

                handling_mark_functions[points_sampled] = handle_mark_annotation(
                    points_sampled
                )

                new_button = tk.Button(
                    master=frame_options_annotations,
                    text=f"{points_sampled}: reference {reference_point} <-> footage {footage_point}",
                    width=35,
                    height=1,
                    bg="blue",
                    fg="black",
                    highlightbackground="SystemButtonFace",
                )

                new_button.bind("<Button>", handling_mark_functions[points_sampled])

                new_button.pack(side=tk.TOP)
                points[points_sampled]["button"] = new_button

                reference_point = None
                footage_point = None
                points_sampled += 1

                transformation = estimate_transformation(points)

                canvas_footage.delete("myPoint")
                canvas_reference.delete("myPoint")
            else:
                print(
                    "Need to pick a point from the footage and from the reference to annotate them"
                )
        else:
            print("Can't annotate in 'Test' mode.")

    text_add_annotation.pack(side=tk.LEFT)
    button_add_annotation.pack(side=tk.LEFT)

    frame_options_add.pack(side=tk.TOP)

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

    ########## pack options with images

    frame_images.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    frame_options.pack(side=tk.LEFT)
    window.mainloop()

    return transformation
