import os
from typing import Tuple, Sequence

import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table


def validate_points(points: np.array) -> np.array:
    # If the user is tracking only a single point, reformat it slightly.
    if points.shape == (2,):
        points = points[np.newaxis, ...]
    elif len(points.shape) == 1:
        print_detection_error_message_and_exit(points)
    else:
        if points.shape[1] != 2 or len(points.shape) > 2:
            print_detection_error_message_and_exit(points)
    return points


def print_detection_error_message_and_exit(points):
    print("\n[red]INPUT ERROR:[/red]")
    print(
        f"Each `Detection` object should have a property `points` of shape (num_of_points_to_track, 2), not {points.shape}. Check your `Detection` list creation code."
    )
    print("You can read the documentation for the `Detection` class here:")
    print("https://github.com/tryolabs/norfair/tree/master/docs#detection\n")
    exit()


def print_objects_as_table(tracked_objects: Sequence):
    """Used for helping in debugging"""
    print()
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Id", style="yellow", justify="center")
    table.add_column("Age", justify="right")
    table.add_column("Hit Counter", justify="right")
    table.add_column("Last distance", justify="right")
    table.add_column("Init Id", justify="center")
    for obj in tracked_objects:
        table.add_row(
            str(obj.id),
            str(obj.age),
            str(obj.hit_counter),
            f"{obj.last_distance:.4f}",
            str(obj.initializing_id),
        )
    console.print(table)


def get_terminal_size(default: Tuple[int, int] = (80, 24)) -> Tuple[int, int]:
    columns, lines = default
    for fd in range(0, 3):  # First in order 0=Std In, 1=Std Out, 2=Std Error
        try:
            columns, lines = os.get_terminal_size(fd)
        except OSError:
            continue
        break
    return columns, lines
