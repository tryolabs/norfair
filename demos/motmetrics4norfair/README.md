# Compute MOTChallenge metricsx

This program allows you to get your predictions and compute MOTChallenge metrics, using a text document with detections in the MOTChallenge format.

You can find MOTchallenge datasets (detections and ground truth files) in https://motchallenge.net

Run the following command to download MOT17 challenge data, including folder with pictures for each video (5.5 GB)
    curl -O https://motchallenge.net/data/MOT17.zip

Alternatively, you can download the same challenge without the corresponding images (9.7 MB)
    curl -O https://motchallenge.net/data/MOT17Labels.zip

You can download the py-motmetrics running the following command on terminal:
    pip install motmetrics==1.2.0

The official py-motmetrics repository can be found in:
https://github.com/cheind/py-motmetrics

Run the following command to display on terminal the motmetrics4norfair help instructions:
    python motmetrics4norfair.py --help