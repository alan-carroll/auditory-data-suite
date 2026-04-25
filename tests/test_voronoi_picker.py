import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import voronoi_picker


class FakeRoot:
    def __init__(self):
        self.idle_callbacks = []
        self.timed_callbacks = []
        self.canceled_callbacks = []

    def after_idle(self, callback):
        self.idle_callbacks.append(callback)
        return f"idle-{len(self.idle_callbacks)}"

    def after(self, delay_ms, callback):
        self.timed_callbacks.append((delay_ms, callback))
        return f"after-{len(self.timed_callbacks)}"

    def after_cancel(self, callback_id):
        self.canceled_callbacks.append(callback_id)


class FakeConfigureEvent:
    width = 900
    height = 850


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

    def test_canvas_configure_schedules_redraw(self):
        picker = voronoi_picker.Picker.__new__(voronoi_picker.Picker)
        picker.root = FakeRoot()
        picker.width = 600
        picker.height = 600
        picker._last_canvas_size = None
        picker._closing = False
        picker._redraw_pending = False

        picker.on_canvas_configure(FakeConfigureEvent())

        self.assertEqual(picker.width, 900)
        self.assertEqual(picker.height, 850)
        self.assertTrue(picker._redraw_pending)
        self.assertEqual(picker.root.idle_callbacks, [picker.redraw])
        self.assertEqual(picker._redraw_after_id, "idle-1")

    def test_interactive_redraw_is_throttled(self):
        picker = voronoi_picker.Picker.__new__(voronoi_picker.Picker)
        picker.root = FakeRoot()
        picker._closing = False
        picker._redraw_pending = False
        picker._redraw_after_id = None

        picker.request_redraw(interactive=True)
        picker.request_redraw(interactive=True)

        self.assertTrue(picker._redraw_pending)
        self.assertEqual(
            picker.root.timed_callbacks,
            [(voronoi_picker._INTERACTIVE_REDRAW_MS, picker.redraw)],
        )

    def test_finish_cancels_pending_redraw(self):
        picker = voronoi_picker.Picker.__new__(voronoi_picker.Picker)
        picker.root = FakeRoot()
        picker.root.quit = lambda: None
        picker._closing = False
        picker._redraw_pending = False
        picker._redraw_after_id = None

        picker.request_redraw(interactive=True)
        picker.finish()

        self.assertTrue(picker._closing)
        self.assertFalse(picker._redraw_pending)
        self.assertIsNone(picker._redraw_after_id)
        self.assertEqual(picker.root.canceled_callbacks, ["after-1"])

    def test_unit_square_clip_preserves_inside_polygon(self):
        polygon = np.asarray([
            [0.2, 0.2],
            [0.8, 0.2],
            [0.5, 0.8],
        ])

        clipped = voronoi_picker._clip_polygon_to_unit_square(polygon)

        np.testing.assert_allclose(clipped, polygon)

    def test_unit_square_clip_trims_oversized_polygon(self):
        polygon = np.asarray([
            [-1.0, -1.0],
            [2.0, -1.0],
            [2.0, 2.0],
            [-1.0, 2.0],
        ])

        clipped = voronoi_picker._clip_polygon_to_unit_square(polygon)

        self.assertEqual(len(clipped), 4)
        self.assertTrue(np.all(clipped >= 0.0))
        self.assertTrue(np.all(clipped <= 1.0))


if __name__ == "__main__":
    unittest.main()
