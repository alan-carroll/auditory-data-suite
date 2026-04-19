"""
Quick harness for the interactive Voronoi boundary picker.

Run with built-in sample points:
    python demo/voronoi_picker_demo.py

Run with a CSV containing x/y coordinates:
    python demo/voronoi_picker_demo.py path/to/coords.csv

The CSV may contain either:
  * x,y
  * number,x,y
  * a header row with x/y columns
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from geometry import alpha_shape, boundary_polygon
import voronoi_picker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the interactive Voronoi boundary picker without "
                    "running a full analysis.",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Optional CSV containing normalized or raw x/y coordinates.",
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=(900, 700),
        metavar=("WIDTH", "HEIGHT"),
        help="Window size in pixels.",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV path to save the accepted buffer points.",
    )
    parser.add_argument(
        "--force-normalize",
        action="store_true",
        help="Scale input coordinates into the 0.1-0.9 range even if they "
             "already look normalized.",
    )
    return parser.parse_args()


def load_points(csv_path: str | None) -> np.ndarray:
    if csv_path is None:
        return sample_points()

    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row in reader:
            cleaned = [cell.strip() for cell in row if cell.strip()]
            if cleaned:
                rows.append(cleaned)

    if not rows:
        raise ValueError("CSV did not contain any coordinate rows.")

    header = [cell.lower() for cell in rows[0]]
    if "x" in header and "y" in header:
        x_idx = header.index("x")
        y_idx = header.index("y")
        data_rows = rows[1:]
    else:
        data_rows = rows
        if len(rows[0]) >= 3:
            x_idx, y_idx = 1, 2
        elif len(rows[0]) >= 2:
            x_idx, y_idx = 0, 1
        else:
            raise ValueError("CSV rows must contain at least two columns.")

    points = []
    for row in data_rows:
        if max(x_idx, y_idx) >= len(row):
            continue
        points.append([float(row[x_idx]), float(row[y_idx])])

    if not points:
        raise ValueError("No usable x/y rows were found in the CSV.")

    return np.asarray(points, dtype=np.float64)


def sample_points() -> np.ndarray:
    return np.asarray([
        [0.18, 0.68],
        [0.24, 0.52],
        [0.28, 0.36],
        [0.38, 0.76],
        [0.42, 0.58],
        [0.46, 0.24],
        [0.54, 0.68],
        [0.58, 0.42],
        [0.64, 0.82],
        [0.68, 0.56],
        [0.76, 0.34],
        [0.82, 0.62],
    ], dtype=np.float64)


def scale_coordinates(
        input_coor: np.ndarray,
        min_coor: float,
        max_coor: float,
        min_scale: float,
        max_scale: float) -> np.ndarray:
    return ((max_scale - min_scale) * (input_coor - min_coor) /
            (max_coor - min_coor)) + min_scale


def maybe_normalize(points: np.ndarray, force: bool = False) -> np.ndarray:
    if not len(points):
        return points

    if not force and points.min() >= 0.0 and points.max() <= 1.0:
        return points

    max_coor = points.max()
    min_coor = points.min()
    if max_coor == min_coor:
        return np.full_like(points, 0.5)

    return scale_coordinates(
        input_coor=points,
        min_coor=min_coor,
        max_coor=max_coor,
        min_scale=0.1,
        max_scale=0.9,
    )


def build_initial_buffer_points(base_points: np.ndarray) -> np.ndarray:
    shape_points = [Point(point) for point in base_points]
    concave_hull, _edge_points = alpha_shape(shape_points, alpha=8)
    concave_hull = boundary_polygon(concave_hull)

    perimeter_length = concave_hull.exterior.length
    num_perimeter_pts = len(concave_hull.exterior.coords)
    avg_edge_length = perimeter_length / num_perimeter_pts
    bonus = concave_hull.boundary.buffer(
        avg_edge_length,
        cap_style=3,
        join_style=3,
    )
    bonus_points = np.asarray([[x, y] for x, y in bonus.exterior.coords], dtype=np.float64)
    if len(bonus_points) > 1 and np.allclose(bonus_points[0], bonus_points[-1]):
        bonus_points = bonus_points[:-1]
    return bonus_points


def save_points(points: np.ndarray, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        writer.writerows(points.tolist())


def main() -> int:
    args = parse_args()
    base_points = maybe_normalize(load_points(args.csv_path), force=args.force_normalize)
    buffer_points = build_initial_buffer_points(base_points)

    print(f"Loaded {len(base_points)} map points.")
    print(f"Starting with {len(buffer_points)} auto-generated border points.")
    print("Move mouse to preview, left click to add, right click to remove, Esc to accept.")

    accepted_points = voronoi_picker.pick_points(
        size=tuple(args.size),
        input_points=base_points,
        buffer_points=buffer_points,
    )

    print(f"Accepted {len(accepted_points)} border points.")
    print(np.array2string(accepted_points, precision=5, separator=", "))

    if args.output:
        save_points(accepted_points, args.output)
        print(f"Saved accepted border points to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
