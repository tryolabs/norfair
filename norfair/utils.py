import numpy as np
from rich.console import Console
from rich.table import Table


def validate_points(points):
    # If the user is tracking only a single point, reformat it slightly.
    if points.shape == (2,):
        points = points[np.newaxis, ...]
    else:
        if points.shape[1] != 2 or len(points.shape) > 2:
            print(
                f"The shape of `Detection.points` should be (num_of_points_to_track, 2), not {points.shape}."
            )
            print("Check your detection conversion code.")
            exit()
    return points


def print_objects_as_table(tracked_objects):
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
