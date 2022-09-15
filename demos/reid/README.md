# ReID Demo

An example of how to use Norfair to perform re-identification using appearance embeddings. This example creates a simulation video where boxes are moving around. When boxes get occluded by other boxes we just grab the detection from the top box, so any other box occluded is not getting a detection. At the same time, detections are not perfect and have some noise.

Comparison of not using ReID (left) vs using it (right)

https://user-images.githubusercontent.com/3588715/189763304-05a0bf3c-16d4-4c2a-b7ba-384460cb3533.mp4

## Instructions

1. Build and run the Docker container with `./run.sh`
2. In the container, run the demo with `python demo.py`.

   This will generate two videos, `demo.avi` which is the original simulation video, and `output.mp4` which is the result of the tracking with ReID.

## Explanation

The goal behind this demo is to show how to implement ReID in a simple way, for this to work in a real scenario a good embedding is required along with a good embedding distance function.

ReID logic comes into play after the spatial distance is not able to match correctly with the detections (exactly when `hit_counter` is `0`). After that, trackers with `hit_counter <= 0` will start to calculate ReID distances until `reid_hit_counter` drops to `0`, once it does it will be removed. If a ReID distance is matched, it will merge the trackers.

In our simulation, we used the color of the box as the embedding (with some noise), and the embedding distance is a histogram comparison between the last embedding from `unmatched_init_trackers` and `dead_objects` compared against the embeddings from `matched_not_init_trackers`

Any other embedding can be used for ReID, such a the output of a deep learning model.
