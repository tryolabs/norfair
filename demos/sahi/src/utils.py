import argparse
import urllib.request
from os import path
from pathlib import Path

import torch
from sahi import AutoDetectionModel


def download_yolov5_model(model_url: str, destination_path: str):
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            model_url,
            destination_path,
        )


def obtain_detection_model(confidence_threshold: float):
    device = (
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    yolov5_model_path = "./models/yolov5x6.pt"
    download_yolov5_model(
        model_url="https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x6.pt",
        destination_path=yolov5_model_path,
    )
    return AutoDetectionModel.from_pretrained(
        model_type="yolov5",
        model_path=yolov5_model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track objects using sahi in a video.")
    parser.add_argument("file", type=str, help="Video files to process")
    parser.add_argument(
        "--output-path", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        help="Norfair's distance threshold",
        default=0.7,
    )
    parser.add_argument(
        "--skip-period",
        type=int,
        help="Norfair's skip period",
        default=1,
    )
    parser.add_argument(
        "--initialization-delay",
        type=int,
        help="Norfair's initialization delay",
        default=15,
    )
    parser.add_argument(
        "--hit-counter-max",
        type=int,
        help="Norfair's hit counter max",
        default=30,
    )
    parser.add_argument(
        "--disable-sahi",
        dest="enable_sahi",
        help="Disable SAHI implementation, run predictions in the whole video",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        help="Sahi's slice height",
        default=256,
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        help="Sahi's overlap height ratio",
        default=0.2,
    )
    parser.add_argument(
        "--model-confidence-threshold",
        type=float,
        help="Model confidence threshold",
        default=0.3,
    )

    return parser
