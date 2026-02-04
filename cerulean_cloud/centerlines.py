"""
centerline creation and aspect ratio factor functions
"""

import centerline.geometry
import networkx as nx
from shapely.geometry import LineString
import geopandas as gpd
import json
import math
import numpy as np


def compute_tree_diameter(graph):
    """
    Computes the longest path (diameter) of a weighted tree graph.

    Uses two passes of depth-first search (DFS):
        1. Start from an arbitrary node to find the farthest node (node_a).
        2. Start from node_a to find the farthest node from it (node_b).

    Returns:
        list: An ordered list of nodes (coordinates) representing the longest path.
    """

    def farthest_node(g, start):
        distances = {}
        stack = [(start, 0)]
        while stack:
            node, dist = stack.pop()
            if node not in distances:
                distances[node] = dist
                for neighbor, attr in g[node].items():
                    weight = attr.get("weight", 1)
                    stack.append((neighbor, dist + weight))
        # Pick the node with maximum distance from 'start'
        farthest = max(distances, key=distances.get)
        return farthest, distances[farthest]

    # Choose an arbitrary starting node
    start = next(iter(graph.nodes))
    node_a, _ = farthest_node(graph, start)
    node_b, _ = farthest_node(graph, node_a)

    # In a tree, the unique shortest path between node_a and node_b is the diameter.
    return nx.shortest_path(graph, source=node_a, target=node_b, weight="weight")


def find_longest_path(centerline_geom):
    """
    Extracts the longest continuous path from a centerline geometry.

    Parameters:
        centerline_geom (LineString or MultiLineString): The centerline geometry
            produced by centerline.geometry.Centerline(item), where item is a Polygon.

    Returns:
        LineString: A LineString representing the longest path (tree diameter).
    """

    def round_point(pt, precision=8):
        """Round a coordinate tuple to avoid floating point issues."""
        return tuple(round(coord, precision) for coord in pt)

    # Normalize input: if it's a single LineString, treat it as a list with one element.
    if centerline_geom.geom_type == "LineString":
        lines = [centerline_geom]
    elif centerline_geom.geom_type == "MultiLineString":
        lines = list(centerline_geom.geoms)
    else:
        raise ValueError(f"Unsupported geometry type: {centerline_geom.geom_type}")

    # Build a graph where nodes are coordinates and each edge connects consecutive points.
    graph = nx.Graph()
    for line in lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            u, v = round_point(coords[i]), round_point(coords[i + 1])
            # Use Euclidean distance as the edge weight.
            weight = ((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2) ** 0.5
            graph.add_edge(u, v, weight=weight)

    diams = [
        LineString(compute_tree_diameter(graph.subgraph(component)))
        for component in list(nx.connected_components(graph))
    ]
    # In case the bug in Centerline produces disconnected centerlines, we need to compute the diameter of each connected component
    # Compute the tree diameter (longest path) of the graph.
    # Note that we just grab the longest one, so the bug is causing us to lose some data here.
    diam_lengths = [d.length for d in diams]
    longest_path = diams[diam_lengths.index(max(diam_lengths))]
    return longest_path


def calculate_centerlines(
    slick_gdf: gpd.GeoDataFrame,
    crs_meters: str,
    close_buffer: int = 2000,
    simplify_tolerance: float = 10.0,  # Tolerance for geometry simplification in meters
):
    """
    From a set of polygons representing oil slick detections, estimate centerlines that go through the detections.
    Inputs:
        slick_gdf: GeoDataFrame of slick detections.
        crs_meters: CRS for slick center in meters.
        close_buffer: Buffer size for cleaning up slick detections.
        simplify_tolerance: Tolerance for simplifying the computed centerlines.
    Returns:
        (dict, float): Tuple containing the GeoJSON representation of the slick centerlines and the aspect ratio factor.
    """
    # clean up the slick detections by dilation followed by erosion
    # this process can merge some polygons but not others, depending on proximity
    slick_closed = (
        slick_gdf.to_crs(crs_meters).buffer(close_buffer).buffer(-close_buffer)
    )

    # split slicks into individual polygons
    slick_closed = slick_closed.explode(ignore_index=True, index_parts=False)

    # Determine candidate slick detections that contribute to the majority of the total area.
    percent_to_keep = 0.95
    slick_closed = slick_closed.iloc[slick_closed.area.argsort()[::-1]]
    cumsum = slick_closed.area.cumsum()
    index_of_min = cumsum.searchsorted(percent_to_keep * cumsum.iloc[-1])
    slick_closed = slick_closed.iloc[: index_of_min + 1].reset_index(drop=True)

    # find a centerline through detections
    slick_cls = []
    for _, item in slick_closed.items():
        # create centerline -> MultiLineString
        polygon_perimeter = item.length  # Perimeter of the polygon
        interp_dist = (
            polygon_perimeter / 1000
        )  # Use a minimum of 1000 points for voronoi calculation
        cl = centerline.geometry.Centerline(item, interpolation_distance=interp_dist)
        longest_path = find_longest_path(cl.geometry)
        longest_path = longest_path.simplify(simplify_tolerance)
        slick_cls.append(longest_path)

    slick_centerline_gdf = gpd.GeoDataFrame(geometry=slick_cls, crs=crs_meters).to_crs(
        "4326"
    )
    slick_centerline_gdf["area"] = slick_closed.geometry.area
    slick_centerline_gdf["length"] = [c.length for c in slick_cls]

    aspect_ratio_factor = compute_aspect_ratio_factor(slick_centerline_gdf, ar_ref=16)
    return json.loads(slick_centerline_gdf.to_json()), aspect_ratio_factor


def compute_aspect_ratio_factor(
    slick_centerlines: gpd.GeoDataFrame, ar_ref=16
) -> float:
    """
    Computes the aspect ratio factor for a given geometry.

    Parameters:
        slick_centerlines (gpd.GeoDataFrame): A GeoDataFrame containing line geometries with a 'length' and 'area' column.
        ar_ref (float, optional): Reference aspect ratio factor. Default is 16.

    Returns:
        float: The computed aspect ratio factor, between 0 and 1, where 0 is a square and 1 is an infinite line
    """

    L = slick_centerlines["length"].values
    A = slick_centerlines["area"].values

    # Centerline Length Weighted Bulk Effective Aspect Ratio
    clwbear = np.average(L**2 / A, weights=L)

    # Note slwbear is between 1 and infinity, so the following transformation moves it between 0 and 1
    arf = 1 - math.exp((1 - clwbear) / ar_ref)  # Aspect Ratio Factor
    return arf
