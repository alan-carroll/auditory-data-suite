import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import voronoi_picker


class VoronoiPickerFileTests(unittest.TestCase):
    def test_export_and_load_buffer_points_round_trip(self):
        points = np.asarray([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer_points.csv"
            voronoi_picker.save_buffer_points_csv(points, path)
            loaded = voronoi_picker.load_buffer_points_csv(path)

        np.testing.assert_allclose(loaded, points)

    def test_load_buffer_points_accepts_plain_xy_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer_points.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([0.1, 0.2])
                writer.writerow([0.3, 0.4])

            loaded = voronoi_picker.load_buffer_points_csv(path)

        np.testing.assert_allclose(
            loaded,
            np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        )

    def test_load_buffer_points_strips_polygon_closing_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "buffer_points.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["x", "y"])
                writer.writerow([0.1, 0.2])
                writer.writerow([0.3, 0.4])
                writer.writerow([0.1, 0.2])

            loaded = voronoi_picker.load_buffer_points_csv(path)

        np.testing.assert_allclose(
            loaded,
            np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        )


if __name__ == "__main__":
    unittest.main()
