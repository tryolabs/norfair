# Compute MOTChallenge metrics

Generate trackers using a text document with detections in the [MOTChallenge](https://motchallenge.net) format, and compute MOTChallenge metrics comparing your tracked objects with a ground truth text file using [py-motmetrics](https://github.com/cheind/py-motmetrics) library.

## Datasets Download

Run one of the following commands to download the [MOT17](https://motchallenge.net) challege:

```bash
curl -O https://motchallenge.net/data/MOT17.zip # Detections + Ground Truth + Images (5.5GB)

curl -O https://motchallenge.net/data/MOT17Labels.zip # Detections + Ground Truth (9.7 MB)
``` 

## Instructions

1. Install py-motmetrics:
```bash
    pip install motmetrics==1.2.0
``` 
2. Display the motmetrics4norfair instructions: 
```bash
    python motmetrics4norfair.py --help
``` 