"""
Interactive Voronoi boundary picker with a tiny Tkinter UI.

Controls:
  * moving the mouse previews a pending border point
  * left click commits the current preview point
  * right click removes the last committed border point
  * Esc or closing the window accepts the current points
"""

from __future__ import annotations

import tkinter as tk
from typing import Iterable

import numpy as np
from scipy.spatial import QhullError, Voronoi
from shapely.geometry import Polygon, box


_BACKGROUND = "#d8d8d8"
_REAL_POINT = "#111111"
_BUFFER_POINT = "#f5f5f5"
_PREVIEW_POINT = "#4f86f7"
_CELL_OUTLINE = "#6b6b6b"
_VIEW_BOUNDS = box(0.0, 0.0, 1.0, 1.0)
_POINT_EPS = 1e-9


def _to_array(points: Iterable[Iterable[float]] | None) -> np.ndarray:
    """Normalize optional point inputs to an Nx2 float array."""
    if points is None:
        return np.empty((0, 2), dtype=np.float64)

    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    return np.atleast_2d(arr)


def _dedupe_points(points: np.ndarray, eps: float = _POINT_EPS) -> np.ndarray:
    """Remove duplicate points while preserving order."""
    if len(points) < 2:
        return points.copy()

    unique_points = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if any(np.allclose(point, existing, atol=eps, rtol=0.0)
               for existing in unique_points):
            continue
        unique_points.append(point)

    return np.asarray(unique_points, dtype=np.float64)


def _strip_closed_loop(points: np.ndarray, eps: float = _POINT_EPS) -> np.ndarray:
    """Drop the duplicated closing point often present in polygon rings."""
    if len(points) > 1 and np.allclose(points[0], points[-1], atol=eps, rtol=0.0):
        return points[:-1].copy()
    return points.copy()


def _color_hex(rgb: tuple[float, float, float]) -> str:
    ints = [max(0, min(255, int(round(channel * 255)))) for channel in rgb]
    return "#{:02x}{:02x}{:02x}".format(*ints)


def _cell_colors(real_count: int, buffer_count: int, has_preview: bool) -> list[str]:
    colors = []
    if real_count:
        reds = np.linspace(0.5, 1.0, real_count)
        colors.extend(_color_hex((red, 0.0, 0.0)) for red in reds)

    colors.extend(_BUFFER_POINT for _ in range(buffer_count))

    if has_preview:
        colors.append("#cfe0ff")

    return colors


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """Reconstruct infinite Voronoi regions into finite polygons.

    Adapted from the SciPy Voronoi finite polygons recipe.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2
        if radius == 0:
            radius = 1.0

    all_ridges = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(
            vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        all_ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if all(vertex >= 0 for vertex in region):
            new_regions.append(region)
            continue

        ridges = all_ridges.get(point_idx, [])
        new_region = [vertex for vertex in region if vertex >= 0]

        if not ridges and not new_region:
            new_regions.append([])
            continue

        for neighbor_idx, vertex_a, vertex_b in ridges:
            if vertex_b < 0:
                vertex_a, vertex_b = vertex_b, vertex_a

            if vertex_a >= 0:
                continue

            tangent = vor.points[neighbor_idx] - vor.points[point_idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm == 0:
                continue
            tangent /= tangent_norm
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[point_idx, neighbor_idx]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[vertex_b] + (direction * radius)

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        if not new_region:
            new_regions.append([])
            continue

        region_vertices = np.asarray([new_vertices[vertex] for vertex in new_region])
        region_center = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - region_center[1],
            region_vertices[:, 0] - region_center[0],
        )
        ordered_region = np.asarray(new_region)[np.argsort(angles)]
        new_regions.append(ordered_region.tolist())

    return new_regions, np.asarray(new_vertices)


def _clipped_voronoi_polygons(points: np.ndarray) -> list[np.ndarray]:
    """Return Voronoi cell polygons clipped to the unit square."""
    if len(points) < 2:
        return []

    try:
        vor = Voronoi(points)
    except QhullError:
        return []

    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=4.0)
    polygons = []

    for region in regions:
        if len(region) < 3:
            polygons.append(np.empty((0, 2), dtype=np.float64))
            continue

        polygon = Polygon(vertices[region]).intersection(_VIEW_BOUNDS)
        if polygon.is_empty:
            polygons.append(np.empty((0, 2), dtype=np.float64))
            continue

        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geom: geom.area)
        elif polygon.geom_type != "Polygon":
            polygons.append(np.empty((0, 2), dtype=np.float64))
            continue

        polygons.append(np.asarray(polygon.exterior.coords[:-1], dtype=np.float64))

    return polygons


class Picker:
    def __init__(
            self,
            size: tuple[int, int] = (600, 600),
            title: str = "Voronoi Picker",
            input_points: Iterable[Iterable[float]] | None = None,
            buffer_points: Iterable[Iterable[float]] | None = None):
        self.width = max(200, int(size[0]))
        self.height = max(200, int(size[1]))
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            background=_BACKGROUND,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.input_points = _to_array(input_points)
        self.buffer_points = _dedupe_points(_strip_closed_loop(_to_array(buffer_points)))
        self.preview_point = np.array([0.5, 0.5], dtype=np.float64)
        self._redraw_pending = False

        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-2>", self.on_right_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Control-Button-1>", self.on_right_click)
        self.root.bind("<Escape>", self.finish)
        self.root.protocol("WM_DELETE_WINDOW", self.finish)

        self.request_redraw()

    def on_mouse_move(self, event):
        self.preview_point = self._event_to_point(event)
        self.request_redraw()

    def on_left_click(self, _event):
        existing_points = self.input_points
        if len(self.buffer_points):
            existing_points = np.vstack([existing_points, self.buffer_points])

        if len(existing_points):
            if np.any(np.all(
                    np.isclose(
                        existing_points,
                        self.preview_point,
                        atol=_POINT_EPS,
                        rtol=0.0,
                    ),
                    axis=1)):
                return

        self.buffer_points = np.vstack([self.buffer_points, self.preview_point])
        self.request_redraw()

    def on_right_click(self, _event):
        if len(self.buffer_points):
            self.buffer_points = self.buffer_points[:-1]
            self.request_redraw()

    def finish(self, _event=None):
        self.root.quit()

    def _event_to_point(self, event) -> np.ndarray:
        x = min(max(event.x / float(self.width), 0.0), 1.0)
        y = 1.0 - min(max(event.y / float(self.height), 0.0), 1.0)
        return np.array([x, y], dtype=np.float64)

    def _canvas_coords(self, points: np.ndarray) -> list[float]:
        flat_coords = []
        for x, y in points:
            flat_coords.append(float(x) * self.width)
            flat_coords.append((1.0 - float(y)) * self.height)
        return flat_coords

    def request_redraw(self):
        if self._redraw_pending:
            return

        self._redraw_pending = True
        self.root.after_idle(self.redraw)

    def redraw(self):
        self._redraw_pending = False
        all_points = np.vstack([
            self.input_points,
            self.buffer_points,
            np.atleast_2d(self.preview_point),
        ])
        polygons = _clipped_voronoi_polygons(all_points)
        colors = _cell_colors(
            real_count=len(self.input_points),
            buffer_count=len(self.buffer_points),
            has_preview=True,
        )

        self.canvas.delete("all")

        for polygon, color in zip(polygons, colors):
            if len(polygon) < 3:
                continue

            self.canvas.create_polygon(
                self._canvas_coords(polygon),
                fill=color,
                outline=_CELL_OUTLINE,
                width=1,
            )

        for point in self.input_points:
            self._draw_point(point, fill=_REAL_POINT, radius=4)

        for point in self.buffer_points:
            self._draw_point(point, fill=_BUFFER_POINT, radius=4)

        self._draw_point(self.preview_point, fill=_PREVIEW_POINT, radius=5)

        self.canvas.create_rectangle(
            8,
            8,
            430,
            40,
            fill="#f7f7f7",
            outline="#b5b5b5",
            width=1,
        )
        self.canvas.create_text(
            16,
            24,
            anchor="w",
            fill="#202020",
            font=("TkDefaultFont", 11, "bold"),
            text="Left click: add    Right click: remove last    Esc/Close: accept",
        )

    def _draw_point(self, point: np.ndarray, fill: str, radius: int):
        x = float(point[0]) * self.width
        y = (1.0 - float(point[1])) * self.height
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline="#111111",
            width=1,
        )

    def run(self) -> np.ndarray:
        self.root.mainloop()
        result = self.buffer_points.copy()
        self.root.destroy()
        return result


def pick_points(size=(600, 600), input_points=None, buffer_points=None):
    picker = Picker(
        size=size,
        input_points=input_points,
        buffer_points=buffer_points,
    )
    return picker.run()


if __name__ == "__main__":
    demo_points = np.array([
        [0.25, 0.25],
        [0.75, 0.25],
        [0.5, 0.7],
    ])
    print(pick_points(input_points=demo_points))
