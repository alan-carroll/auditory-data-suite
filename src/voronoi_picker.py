"""
Interactive Voronoi boundary picker with a small Tkinter editor UI.

Controls:
  * Add mode (A): click to add a buffer point
  * Move mode (M): click/drag the highlighted buffer point
  * Delete mode (D): click the highlighted buffer point
  * Pan mode (P): click/drag the view
  * Mouse wheel or Ctrl +/- zooms the view
  * Ctrl+Z undoes, Ctrl+Shift+Z redoes
  * Esc, Accept, or closing the window accepts the current points
"""

from __future__ import annotations

import csv
from os import PathLike
import tkinter as tk
from tkinter import messagebox
from typing import Iterable, Optional

import numpy as np
from scipy.spatial import QhullError, Voronoi
from shapely.geometry import Polygon, box

from dialogs import get_file, save_file


_BACKGROUND = "#d8d8d8"
_REAL_POINT = "#111111"
_BUFFER_POINT = "#f5f5f5"
_PREVIEW_POINT = "#4f86f7"
_HOVER_RING = "#1b74e4"
_DELETE_RING = "#d93025"
_CELL_OUTLINE = "#6b6b6b"
_PANEL_BG = "#f7f7f7"
_VIEW_BOUNDS = box(0.0, 0.0, 1.0, 1.0)
_POINT_EPS = 1e-9
_HIT_RADIUS_PX = 18
_MIN_VIEW_SPAN = 0.05

_ADD = "add"
_MOVE = "move"
_DELETE = "delete"
_PAN = "pan"


def _to_array(points: Optional[Iterable[Iterable[float]]]) -> np.ndarray:
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


def load_buffer_points_csv(path: str | PathLike[str]) -> np.ndarray:
    """Load exported buffer points from a CSV file with x/y columns."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
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
        if len(rows[0]) < 2:
            raise ValueError("CSV rows must contain at least x and y columns.")
        x_idx, y_idx = 0, 1

    points = []
    for row in data_rows:
        if max(x_idx, y_idx) >= len(row):
            continue
        points.append([float(row[x_idx]), float(row[y_idx])])

    if not points:
        raise ValueError("No usable x/y rows were found in the CSV.")

    return _dedupe_points(_strip_closed_loop(np.asarray(points, dtype=np.float64)))


def save_buffer_points_csv(points: Iterable[Iterable[float]],
                           path: str | PathLike[str]) -> None:
    """Export buffer points to a reproducible x/y CSV file."""
    points = _to_array(points)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y"])
        writer.writerows(points.tolist())


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


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None):
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
            input_points: Optional[Iterable[Iterable[float]]] = None,
            buffer_points: Optional[Iterable[Iterable[float]]] = None):
        self.width = max(200, int(size[0]))
        self.height = max(200, int(size[1]))
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(True, True)
        self.root.minsize(800, 800)

        self.mode = tk.StringVar(value=_ADD)
        self.status_text = tk.StringVar()
        self.input_points = _to_array(input_points)
        self.buffer_points = _dedupe_points(
            _strip_closed_loop(_to_array(buffer_points)),
        )
        self.preview_point = np.array([0.5, 0.5], dtype=np.float64)
        self.hover_index: Optional[int] = None
        self.hover_distance_px: Optional[float] = None
        self.drag_index: Optional[int] = None
        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_view: Optional[tuple[float, float, float, float]] = None
        self._move_history_pushed = False
        self._redraw_pending = False
        self._last_canvas_size: Optional[tuple[int, int]] = None
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self.view_xmin = 0.0
        self.view_xmax = 1.0
        self.view_ymin = 0.0
        self.view_ymax = 1.0

        self._build_menu()
        self._build_layout()
        self._bind_events()
        self._update_status()
        self.request_redraw()

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(
            label="Load Buffer Points...",
            accelerator="Ctrl+O",
            command=self.load_points_dialog,
        )
        file_menu.add_command(
            label="Export Buffer Points...",
            accelerator="Ctrl+S",
            command=self.export_points_dialog,
        )
        file_menu.add_separator()
        file_menu.add_command(label="Accept", command=self.finish)
        file_menu.add_command(
            label="Accept and Export...",
            accelerator="Ctrl+E",
            command=self.accept_and_export,
        )
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(
            label="Redo",
            accelerator="Ctrl+Shift+Z",
            command=self.redo,
        )
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear Buffer Points", command=self.clear_points)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        interaction_menu = tk.Menu(menubar, tearoff=False)
        interaction_menu.add_radiobutton(
            label="Add Buffer Point",
            accelerator="A",
            variable=self.mode,
            value=_ADD,
            command=self.on_mode_changed,
        )
        interaction_menu.add_radiobutton(
            label="Move Buffer Point",
            accelerator="M",
            variable=self.mode,
            value=_MOVE,
            command=self.on_mode_changed,
        )
        interaction_menu.add_radiobutton(
            label="Delete Buffer Point",
            accelerator="D",
            variable=self.mode,
            value=_DELETE,
            command=self.on_mode_changed,
        )
        interaction_menu.add_radiobutton(
            label="Pan View",
            accelerator="P",
            variable=self.mode,
            value=_PAN,
            command=self.on_mode_changed,
        )
        menubar.add_cascade(label="Interaction", menu=interaction_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_command(
            label="Zoom In",
            accelerator="Ctrl++",
            command=lambda: self.zoom_view(1.25),
        )
        view_menu.add_command(
            label="Zoom Out",
            accelerator="Ctrl+-",
            command=lambda: self.zoom_view(0.8),
        )
        view_menu.add_command(
            label="Reset View",
            accelerator="0",
            command=self.reset_view,
        )
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

    def _build_layout(self):
        frame = tk.Frame(self.root, background=_PANEL_BG)
        frame.pack(fill="both", expand=True)

        toolbar = tk.Frame(frame, background=_PANEL_BG, padx=6, pady=5)
        toolbar.pack(side="top", fill="x")

        for label, mode in (
                ("Add", _ADD),
                ("Move", _MOVE),
                ("Delete", _DELETE),
                ("Pan", _PAN)):
            tk.Radiobutton(
                toolbar,
                text=label,
                value=mode,
                variable=self.mode,
                indicatoron=False,
                padx=10,
                pady=3,
                command=self.on_mode_changed,
            ).pack(side="left", padx=(0, 4))

        tk.Frame(toolbar, width=10, background=_PANEL_BG).pack(side="left")
        tk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=(0, 4))
        tk.Button(toolbar, text="Redo", command=self.redo).pack(side="left", padx=(0, 8))
        tk.Button(toolbar, text="Load", command=self.load_points_dialog).pack(
            side="left",
            padx=(0, 4),
        )
        tk.Button(toolbar, text="Export", command=self.export_points_dialog).pack(
            side="left",
            padx=(0, 8),
        )
        tk.Button(toolbar, text="Accept", command=self.finish).pack(side="right")
        tk.Button(toolbar, text="Accept + Export", command=self.accept_and_export).pack(
            side="right",
            padx=(0, 4),
        )

        self.canvas = tk.Canvas(
            frame,
            width=self.width,
            height=self.height,
            background=_BACKGROUND,
            highlightthickness=0,
        )
        self.canvas.pack(side="top", fill="both", expand=True)

        status = tk.Label(
            frame,
            textvariable=self.status_text,
            anchor="w",
            background=_PANEL_BG,
            padx=8,
            pady=4,
        )
        status.pack(side="bottom", fill="x")

    def _bind_events(self):
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.root.bind("<Escape>", self.finish)
        self.root.bind_all("<Key-a>", lambda _event: self.set_mode(_ADD))
        self.root.bind_all("<Key-m>", lambda _event: self.set_mode(_MOVE))
        self.root.bind_all("<Key-d>", lambda _event: self.set_mode(_DELETE))
        self.root.bind_all("<Key-p>", lambda _event: self.set_mode(_PAN))
        self.root.bind_all("<Control-z>", self.undo)
        self.root.bind_all("<Command-z>", self.undo)
        self.root.bind_all("<Control-Shift-Z>", self.redo)
        self.root.bind_all("<Command-Shift-Z>", self.redo)
        self.root.bind_all("<Control-y>", self.redo)
        self.root.bind_all("<Command-y>", self.redo)
        self.root.bind_all("<Control-o>", self.load_points_dialog)
        self.root.bind_all("<Command-o>", self.load_points_dialog)
        self.root.bind_all("<Control-s>", self.export_points_dialog)
        self.root.bind_all("<Command-s>", self.export_points_dialog)
        self.root.bind_all("<Control-e>", self.accept_and_export)
        self.root.bind_all("<Command-e>", self.accept_and_export)
        self.root.bind_all("<Control-equal>", lambda _event: self.zoom_view(1.25))
        self.root.bind_all("<Command-equal>", lambda _event: self.zoom_view(1.25))
        self.root.bind_all("<Control-plus>", lambda _event: self.zoom_view(1.25))
        self.root.bind_all("<Command-plus>", lambda _event: self.zoom_view(1.25))
        self.root.bind_all("<Control-minus>", lambda _event: self.zoom_view(0.8))
        self.root.bind_all("<Command-minus>", lambda _event: self.zoom_view(0.8))
        self.root.bind_all("<Key-0>", lambda _event: self.reset_view())
        self.root.protocol("WM_DELETE_WINDOW", self.finish)

    def on_mode_changed(self):
        self.drag_index = None
        self.drag_start = None
        self.drag_view = None
        self._move_history_pushed = False
        self._refresh_hover()
        self._update_status()
        self.request_redraw()

    def set_mode(self, mode: str):
        self.mode.set(mode)
        self.on_mode_changed()

    def on_mouse_move(self, event):
        self.preview_point = self._event_to_point(event)
        self._refresh_hover()
        self._update_status()
        self.request_redraw()

    def on_left_press(self, event):
        mode = self.mode.get()
        self.preview_point = self._event_to_point(event)

        if mode == _ADD:
            self.add_point(self.preview_point)
        elif mode == _DELETE:
            self._refresh_hover()
            if self.hover_index is not None:
                self.delete_point(self.hover_index)
        elif mode == _MOVE:
            self._refresh_hover()
            if self.hover_index is not None:
                self.drag_index = self.hover_index
                self._move_history_pushed = False
        elif mode == _PAN:
            self.drag_start = (event.x, event.y)
            self.drag_view = (
                self.view_xmin,
                self.view_xmax,
                self.view_ymin,
                self.view_ymax,
            )

        self._update_status()

    def on_left_drag(self, event):
        mode = self.mode.get()
        if mode == _MOVE and self.drag_index is not None:
            if not self._move_history_pushed:
                self._push_history()
                self._move_history_pushed = True
            self.buffer_points[self.drag_index] = self._event_to_point(event)
            self.hover_index = self.drag_index
            self.hover_distance_px = 0.0
            self.request_redraw()
        elif mode == _PAN and self.drag_start and self.drag_view:
            self._pan_to_event(event)
            self.request_redraw()

    def on_left_release(self, _event):
        self.drag_index = None
        self.drag_start = None
        self.drag_view = None
        self._move_history_pushed = False
        self._refresh_hover()
        self.request_redraw()

    def on_mouse_wheel(self, event):
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            factor = 0.8
        else:
            factor = 1.25
        self.zoom_view(factor, event=event)

    def on_canvas_configure(self, event):
        size = (max(1, int(event.width)), max(1, int(event.height)))
        if size == self._last_canvas_size:
            return

        self._last_canvas_size = size
        self.width, self.height = size
        self.request_redraw()

    def add_point(self, point: np.ndarray):
        existing_points = self.input_points
        if len(self.buffer_points):
            existing_points = np.vstack([existing_points, self.buffer_points])

        if len(existing_points) and np.any(np.all(
                np.isclose(existing_points, point, atol=_POINT_EPS, rtol=0.0),
                axis=1,
        )):
            return

        self._push_history()
        self.buffer_points = np.vstack([self.buffer_points, point])
        self._refresh_hover()
        self._update_status()
        self.request_redraw()

    def delete_point(self, index: int):
        if index < 0 or index >= len(self.buffer_points):
            return

        self._push_history()
        self.buffer_points = np.delete(self.buffer_points, index, axis=0)
        self.hover_index = None
        self.hover_distance_px = None
        self._update_status()
        self.request_redraw()

    def clear_points(self):
        if not len(self.buffer_points):
            return
        if not messagebox.askyesno(
                "Clear buffer points?",
                "Remove all buffer points from this picker?",
                parent=self.root):
            return
        self._push_history()
        self.buffer_points = np.empty((0, 2), dtype=np.float64)
        self._refresh_hover()
        self._update_status()
        self.request_redraw()

    def undo(self, _event=None):
        if not self._undo_stack:
            return "break"
        self._redo_stack.append(self.buffer_points.copy())
        self.buffer_points = self._undo_stack.pop()
        self._refresh_hover()
        self._update_status()
        self.request_redraw()
        return "break"

    def redo(self, _event=None):
        if not self._redo_stack:
            return "break"
        self._undo_stack.append(self.buffer_points.copy())
        self.buffer_points = self._redo_stack.pop()
        self._refresh_hover()
        self._update_status()
        self.request_redraw()
        return "break"

    def load_points_dialog(self, _event=None):
        path = get_file(
            title="Load buffer points",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            parent=self.root,
        )
        if not path:
            return "break"

        try:
            loaded_points = load_buffer_points_csv(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not load points", str(exc), parent=self.root)
            return "break"

        replace = True
        if len(self.buffer_points):
            response = messagebox.askyesnocancel(
                "Load buffer points",
                "Replace current buffer points? Choose No to append them.",
                parent=self.root,
            )
            if response is None:
                return "break"
            replace = bool(response)

        self._push_history()
        if replace:
            self.buffer_points = loaded_points
        else:
            self.buffer_points = _dedupe_points(np.vstack([
                self.buffer_points,
                loaded_points,
            ]))
        self._refresh_hover()
        self._update_status()
        self.request_redraw()
        return "break"

    def export_points_dialog(self, _event=None) -> bool | str:
        path = save_file(
            title="Export buffer points",
            defaultextension=".csv",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            parent=self.root,
        )
        if not path:
            return "break"

        try:
            save_buffer_points_csv(self.buffer_points, path)
        except OSError as exc:
            messagebox.showerror("Could not export points", str(exc), parent=self.root)
            return "break"

        messagebox.showinfo(
            "Export complete",
            f"Saved {len(self.buffer_points)} points.",
            parent=self.root,
        )
        return True

    def accept_and_export(self, _event=None):
        exported = self.export_points_dialog()
        if exported is True:
            self.finish()
        return "break"

    def finish(self, _event=None):
        self.root.quit()

    def reset_view(self):
        self.view_xmin = 0.0
        self.view_xmax = 1.0
        self.view_ymin = 0.0
        self.view_ymax = 1.0
        self.request_redraw()

    def zoom_view(self, factor: float, event=None):
        factor = max(0.1, float(factor))
        x_span = self.view_xmax - self.view_xmin
        y_span = self.view_ymax - self.view_ymin
        new_x_span = min(1.0, max(_MIN_VIEW_SPAN, x_span / factor))
        new_y_span = min(1.0, max(_MIN_VIEW_SPAN, y_span / factor))

        if event is None:
            center_x = (self.view_xmin + self.view_xmax) / 2.0
            center_y = (self.view_ymin + self.view_ymax) / 2.0
            x_min = center_x - (new_x_span / 2.0)
            y_min = center_y - (new_y_span / 2.0)
        else:
            width, height = self._canvas_size()
            rel_x = min(max(event.x / width, 0.0), 1.0)
            rel_y = min(max(event.y / height, 0.0), 1.0)
            data_x = self.view_xmin + (rel_x * x_span)
            data_y = self.view_ymax - (rel_y * y_span)
            x_min = data_x - (rel_x * new_x_span)
            y_max = data_y + (rel_y * new_y_span)
            y_min = y_max - new_y_span

        self._set_view(x_min, y_min, new_x_span, new_y_span)
        self.request_redraw()

    def _pan_to_event(self, event):
        if not self.drag_start or not self.drag_view:
            return
        start_x, start_y = self.drag_start
        xmin, xmax, ymin, ymax = self.drag_view
        width, height = self._canvas_size()
        x_span = xmax - xmin
        y_span = ymax - ymin
        dx = ((event.x - start_x) / width) * x_span
        dy = ((event.y - start_y) / height) * y_span
        self._set_view(xmin - dx, ymin + dy, x_span, y_span)

    def _set_view(self, x_min: float, y_min: float, x_span: float, y_span: float):
        x_span = min(1.0, max(_MIN_VIEW_SPAN, x_span))
        y_span = min(1.0, max(_MIN_VIEW_SPAN, y_span))
        x_min = min(max(float(x_min), 0.0), 1.0 - x_span)
        y_min = min(max(float(y_min), 0.0), 1.0 - y_span)
        self.view_xmin = x_min
        self.view_xmax = x_min + x_span
        self.view_ymin = y_min
        self.view_ymax = y_min + y_span

    def _event_to_point(self, event) -> np.ndarray:
        width, height = self._canvas_size()
        rel_x = min(max(event.x / width, 0.0), 1.0)
        rel_y = min(max(event.y / height, 0.0), 1.0)
        x = self.view_xmin + rel_x * (self.view_xmax - self.view_xmin)
        y = self.view_ymax - rel_y * (self.view_ymax - self.view_ymin)
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        return np.array([x, y], dtype=np.float64)

    def _canvas_size(self) -> tuple[float, float]:
        width = max(float(self.canvas.winfo_width()), 1.0)
        height = max(float(self.canvas.winfo_height()), 1.0)
        if width <= 1.0:
            width = float(self.width)
        if height <= 1.0:
            height = float(self.height)
        return width, height

    def _canvas_coords(self, points: np.ndarray) -> list[float]:
        flat_coords = []
        for point in points:
            x, y = self._point_to_canvas(point)
            flat_coords.extend([x, y])
        return flat_coords

    def _point_to_canvas(self, point: np.ndarray) -> tuple[float, float]:
        width, height = self._canvas_size()
        x = ((float(point[0]) - self.view_xmin) /
             (self.view_xmax - self.view_xmin)) * width
        y = ((self.view_ymax - float(point[1])) /
             (self.view_ymax - self.view_ymin)) * height
        return x, y

    def _nearest_buffer_index(
            self,
            point: np.ndarray) -> tuple[Optional[int], Optional[float]]:
        if not len(self.buffer_points):
            return None, None

        px, py = self._point_to_canvas(point)
        distances = []
        for buffer_point in self.buffer_points:
            bx, by = self._point_to_canvas(buffer_point)
            distances.append(np.hypot(px - bx, py - by))

        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance > _HIT_RADIUS_PX:
            return None, distance
        return index, distance

    def _refresh_hover(self):
        if self.mode.get() in {_MOVE, _DELETE}:
            self.hover_index, self.hover_distance_px = self._nearest_buffer_index(
                self.preview_point,
            )
        else:
            self.hover_index = None
            self.hover_distance_px = None

    def _push_history(self):
        self._undo_stack.append(self.buffer_points.copy())
        self._redo_stack.clear()

    def _preview_is_new(self) -> bool:
        points = self.input_points
        if len(self.buffer_points):
            points = np.vstack([points, self.buffer_points])
        if not len(points):
            return True
        return not np.any(np.all(
            np.isclose(points, self.preview_point, atol=_POINT_EPS, rtol=0.0),
            axis=1,
        ))

    def _update_status(self):
        mode = self.mode.get()
        parts = [f"Mode: {mode.title()}", f"Buffer points: {len(self.buffer_points)}"]
        if mode in {_MOVE, _DELETE}:
            if self.hover_index is None:
                parts.append("Target: none nearby")
            else:
                parts.append(f"Target: #{self.hover_index + 1}")
        parts.append("A/M/D/P modes, Ctrl+Z undo, Ctrl+Shift+Z redo")
        self.status_text.set("   |   ".join(parts))

    def request_redraw(self):
        if self._redraw_pending:
            return

        self._redraw_pending = True
        self.root.after_idle(self.redraw)

    def redraw(self):
        self._redraw_pending = False
        has_preview = self.mode.get() == _ADD and self._preview_is_new()
        all_point_parts = [self.input_points, self.buffer_points]
        if has_preview:
            all_point_parts.append(np.atleast_2d(self.preview_point))
        all_points = np.vstack(all_point_parts)
        polygons = _clipped_voronoi_polygons(all_points)
        colors = _cell_colors(
            real_count=len(self.input_points),
            buffer_count=len(self.buffer_points),
            has_preview=has_preview,
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

        self._draw_unit_bounds()

        for point in self.input_points:
            self._draw_point(point, fill=_REAL_POINT, radius=4)

        for point in self.buffer_points:
            self._draw_point(point, fill=_BUFFER_POINT, radius=4)

        if self.hover_index is not None:
            ring_color = _DELETE_RING if self.mode.get() == _DELETE else _HOVER_RING
            self._draw_selection_ring(
                self.buffer_points[self.hover_index],
                ring_color,
                self.hover_distance_px,
            )

        if has_preview:
            self._draw_point(self.preview_point, fill=_PREVIEW_POINT, radius=5)

        self._draw_overlay()

    def _draw_unit_bounds(self):
        self.canvas.create_rectangle(
            *self._point_to_canvas(np.array([0.0, 1.0])),
            *self._point_to_canvas(np.array([1.0, 0.0])),
            outline="#303030",
            width=2,
        )

    def _draw_overlay(self):
        width, _height = self._canvas_size()
        text = {
            _ADD: "Click to add buffer point",
            _MOVE: "Click and drag highlighted buffer point",
            _DELETE: "Click highlighted buffer point to delete",
            _PAN: "Drag to pan; mouse wheel zooms",
        }[self.mode.get()]
        self.canvas.create_rectangle(
            8,
            8,
            min(width - 8, 380),
            38,
            fill="#f7f7f7",
            outline="#b5b5b5",
            width=1,
        )
        self.canvas.create_text(
            16,
            23,
            anchor="w",
            fill="#202020",
            font=("TkDefaultFont", 11, "bold"),
            text=text,
        )

    def _draw_point(self, point: np.ndarray, fill: str, radius: int):
        x, y = self._point_to_canvas(point)
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline="#111111",
            width=1,
        )

    def _draw_selection_ring(
            self,
            point: np.ndarray,
            color: str,
            distance_px: Optional[float]):
        x, y = self._point_to_canvas(point)
        distance = _HIT_RADIUS_PX if distance_px is None else distance_px
        closeness = 1.0 - min(max(distance / _HIT_RADIUS_PX, 0.0), 1.0)
        radius = 9 + (closeness * 7)
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=color,
            width=3,
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
