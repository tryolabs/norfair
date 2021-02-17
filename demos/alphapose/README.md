# Tracking pedestrians with AlphaPose

An example of how to integrate Norfair into the video inference loop of a pre existing repository. This example uses Norfair to try out custom trackers on [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).

## Instructions

1. Install Norfair with `pip install norfair[video]`.
2. [Follow the instructions](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch#installation) to install the Pytorch version of AlphaPose.
3. Apply this diff to this [commit](https://github.com/MVIG-SJTU/AlphaPose/commit/ded84d450faf56227680f0527ff7e24ab7268754) on AlphaPose and use their [video_demo.py](https://github.com/MVIG-SJTU/AlphaPose/blob/ded84d450faf56227680f0527ff7e24ab7268754/video_demo.py) to process your video.

    ```diff
    diff --git a/dataloader.py b/dataloader.py
    index ed6ee90..a7dedb0 100644
    --- a/dataloader.py
    +++ b/dataloader.py
    @@ -17,6 +17,8 @@ import cv2
    import json
    import numpy as np
    +import norfair
    import time
    import torch.multiprocessing as mp
    from multiprocessing import Process
    @@ -606,6 +608,17 @@ class WebcamLoader:
            # indicate that the thread should be stopped
            self.stopped = True

    +detection_threshold = 0.2
    +keypoint_dist_threshold = None
    +def keypoints_distance(detected_pose, tracked_pose):
    +    distances = np.linalg.norm(detected_pose.points - tracked_pose.estimate, axis=1)
    +    match_num = np.count_nonzero(
    +        (distances < keypoint_dist_threshold)
    +        * (detected_pose.scores > detection_threshold)
    +        * (tracked_pose.last_detection.scores > detection_threshold)
    +    )
    +    return 1 / (1 + match_num)
    +
    class DataWriter:
        def __init__(self, save_video=False,
                    savepath='demos/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480),
    @@ -624,6 +637,11 @@ class DataWriter:
            if opt.save_img:
                if not os.path.exists(opt.outputpath + '/vis'):
                    os.mkdir(opt.outputpath + '/vis')
    +        self.tracker = norfair.Tracker(
    +            distance_function=keypoints_distance,
    +            distance_threshold=0.3,
    +            detection_threshold=0.2
    +        )

        def start(self):
            # start a thread to read frames from the file video stream
    @@ -672,7 +690,15 @@ class DataWriter:
                        }
                        self.final_result.append(result)
                        if opt.save_img or opt.save_video or opt.vis:
    -                        img = vis_frame(orig_img, result)
    +                        img = orig_img.copy()
    +                        global keypoint_dist_threshold
    +                        keypoint_dist_threshold = img.shape[0] / 30
    +                        detections = [
    +                            norfair.Detection(p['keypoints'].numpy(), scores=p['kp_score'].squeeze().numpy())
    +                            for p in result['result']
    +                        ]
    +                        tracked_objects = self.tracker.update(detections=detections)
    +                        norfair.draw_tracked_objects(img, tracked_objects)
                            if opt.vis:
                                cv2.imshow("AlphaPose Demo", img)
                                cv2.waitKey(30)
    ```

## Explanation

With Norfair, you can try out your own custom tracker on the very accurate poses produced by AlphaPose by just integrating it into AlphaPose itself, and therefore avoiding the difficult job of decoupling the model from the code base.

This produces the following results:

![Norfair AlphaPose demo](../../docs/alphapose.gif)
