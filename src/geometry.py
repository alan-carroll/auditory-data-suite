import math

import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as geometry
from shapely.geometry import Point
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay, Voronoi

import voronoi_picker
import cli_utils as cli

__all__ = [
    "pick_voronoi", "alpha_shape", "check_voronoi",
    "check_points", "check_csv_points", "scale_coordinates",
]

def scale_coordinates(input_coor, min_coor, max_coor, min_scale, max_scale):
    """
    Simple scaling function for map coordinates.
    """
    return ((max_scale - min_scale) * (input_coor - min_coor) / 
            (max_coor - min_coor)) + min_scale

def alpha_shape(points, alpha):
    """
    Utility for pick_voronoi().
    Compute the alpha shape (concave hull) of a set of points.
    
    Code from:
      http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
    
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the gooeyness of the border. 
      Smaller numbers don't fall inward as much as larger numbers. Too large, 
      and you lose everything!
    """
    if len(points) < 4:
        # If triangle, there is no sense in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        # Semi-perimeter of triangle
        s = (a + b + c) / 2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # Radius filter
        if circum_r < 1.0 / alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    return unary_union(triangles), edge_points

def pick_voronoi(map_points_df, map_width, map_height):
    """
    Generate Voronoi tessellation for map and store data.

    Uses Shapely for geometry -- https://shapely.readthedocs.io/en/latest/

    Creates extended border around original map points to eliminate voronoi 
    cells going to infinity. Newly created points are buffered away from 
    existing border points by the average edge length between border points.
    
    Returns a list of sites containing xy coords and voronoi vertices for all 
    real map points, and the list of extra points used to define boundaries.
    """
    base_pts = map_points_df[["x", "y"]].values
    shape_pts = [Point(pnt) for pnt in base_pts]
    
    # alpha=5 for relaxed boundaries, 10 for tighter
    concave_hull, edge_points = alpha_shape(shape_pts, alpha=8)

    # 3 = Square cap; 3 = Bevel join
    perimeter_length = concave_hull.exterior.length
    num_perimeter_pts = len(concave_hull.exterior.coords)
    avg_edge_length = perimeter_length / num_perimeter_pts
    bonus = concave_hull.boundary.buffer(avg_edge_length, 
                                         cap_style=3, 
                                         join_style=3)
    bonus_pts = np.array([[x, y] for x, y in bonus.exterior.coords])

    # Use vispy-based voronoi program to pick additional border points
    cli.note(
        "Add additional border points to voronoi diagram as necessary."
        "\nLeft-click adds a point."
        "\nRight-click removes last added point."
        "\n<Esc> or exit window to accept points and continue."
    )
    bonus_pts = voronoi_picker.pick_points(size=(round(map_width / 1.5), 
                                                 round(map_height / 1.5)),
                                           input_points=base_pts, 
                                           buffer_points=bonus_pts)
    vor_input = list(base_pts) + list(bonus_pts)
    vor = Voronoi(vor_input)

    sites_list = []
    for idx in map_points_df.index:
        xy = map_points_df.iloc[idx].loc[['x', 'y']].values
        map_num = int(map_points_df.iloc[idx].number)
        vor_indices = np.where(vor.points == xy)[0]
        
        """
        Sometimes 2+ points share an identical floating point coordinate value 
        on either the x or y position. These cases will match a single coord 
        against voronoi indices, and can result in a duplicate polygon in one 
        xy position while failing to draw its own polygon.
        
        This problem is solved by identifying and selecting the index which 
        matches twice (both x and y coordinates).
        """
        matches, counts = np.unique(vor_indices, return_counts=True)
        vor_index = matches[counts == 2][0]

        region_idx = vor.point_region[vor_index]
        vertex_idx = vor.regions[region_idx]
        polygon = vor.vertices[vertex_idx]
        centroid = geometry.Polygon(polygon).centroid.coords.xy

        sites_list.append({
            "number": map_num,
            "x": xy[0],
            "y": xy[1],
            "voronoi_centroid": np.array([centroid[0][0], 
                                          centroid[1][0]]).tolist(),
            "voronoi_vertices": polygon.tolist(),
        })
        
    return sites_list, bonus_pts


def check_voronoi(sites_list, bonus_pts):
    """
    Simple figure showing voronoi diagram and associated map #'s
    """
    _, ax = plt.subplots()
    for row in sites_list:
        polygon = row["voronoi_vertices"]
        ax.fill(*zip(*polygon), alpha=0.4)
        ax.text(row["x"], row["y"], row["number"])
    ax.plot(*zip(*bonus_pts), "bd")

    plt.ion()
    plt.show()


def check_points(numbers_image, points_dict):
    """
    Simple figure matching map #'s with points.
    Used to test/debug initial map image OCR work.
    """
    _, ax = plt.subplots()
    ax.imshow(numbers_image)
    for i in points_dict:
        ax.plot(points_dict[i]["point"][1],
                points_dict[i]["point"][0],
                'ko', ms=5)
        ax.text(points_dict[i]["point"][1],
                points_dict[i]["point"][0],
                points_dict[i]["ocr"],
                color="#5A86AD")
    plt.ion()
    plt.show()


def check_csv_points(points_df):
    """
    Simple figure matching map #'s with points.
    Used to test/debug initial map .csv work.
    """
    _, ax = plt.subplots()
    ax.scatter(points_df.x, points_df.y)
    for row in points_df.itertuples():
        ax.text(row.x, row.y, row.number)
    plt.ion()
    plt.show()