# Multi-Camera Demo

In this example, we show how to associate trackers of different synchronized videos in Norfair.

Why would we want that?

- When subjects that are being tracked go out of frame in one video, you might still be able to track them and recognize that it is the same individual if it is still visible in other videos.
- Take footage from one or many videos to a common reference frame. For example, if you are watching a soccer match, you might want to combine the information from different cameras and show the position of the players from a top-down view.

## Example 1: Associating different videos

This method will allow you to associate trackers from different footage of the same scene. You can use as many videos as you want.

```bash 
python3 demo.py video1.mp4 video2.mp4 video3.mp4
```

A UI will appear to associate points in `video1.mp4` with points in the other videos, to set `video1.mp4` as a common frame of reference.

If the videos move, you should also use the `--use-motion-estimator-footage` flag to consider camera movement.

## Example 2: Creating a new perspective

This method will allow you to associate trackers from different footage of the same scen, and create a new perspective of the scene which didn't exist in those videos. You can use as many videos as you want, and also you need to provide one reference (either an image or video) corresponding to the new perspective. In the soccer example, the reference could be a cenital view of a soccer field.

```bash 
python3 demo.py video1.mp4 video2.mp4 video3.mp4 --reference path_to_reference_file
```

As before, you will have to use the UI.

If the videos where you are tracking have camera movement, you should also use the `--use-motion-estimator-footage` flag to consider camera movement in those videos.

If you are using a video for the reference file, and the camera moves in the reference, then you should use the `--use-motion-estimator-reference` flag.


For additional settings, you may display the instructions using `python demo.py --help`.


## UI usage

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