# -*- coding: utf-8 -*-
"""
Density-reduction helpers built on correlated-disorder structures.

Functions include:
- Loading particle positions from CSV
- Generating a hexagonal lattice
- Building polygonal “areas” around disorder edges
- Selecting particles inside polygons or circles
- Computing area spacing and radius to reach target coverage
"""

from __future__ import annotations

from typing import List, Tuple

from numpy import genfromtxt
import Correlated_disorder_generation as Vor
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.pyplot import figure
from scipy import pi
from numpy import sin, cos
from numpy.lib.scimath import sqrt


def load_positions_files(filename: str, location: str) -> np.ndarray:
    """Load particle positions from a CSV file.

    Args:
        filename: File name without extension.
        location: Directory path where the CSV is stored.

    Returns:
        (N, 2) array with particle coordinates.
    """
    particle_positions = genfromtxt(location + '/' + filename + '.csv', delimiter=',')
    return particle_positions


def define_areas_disorder_edge(
    edge: np.ndarray,
    areas: np.ndarray,
    radius: float,
    circle_prec: int,
) -> List[Polygon]:
    """Create polygons from edge points around given area centers.

    For each center in ``areas``, a circle of radius ``radius`` is sampled at
    ``circle_prec`` points. Edge points inside the circle are collected; the
    first-appearance order of their nearest samples defines a polygon.

    Args:
        edge: (M, 2) array with edge point coordinates.
        areas: (K, 2) array with center coordinates.
        radius: Circle radius around each center.
        circle_prec: Number of samples used to discretize each circle.

    Returns:
        List of shapely Polygons (polygons with <3 vertices are skipped).
    """
    tree_edges = cKDTree(edge)  # nearest-neighbor search on edge points

    # Circle samples and output containers
    limit_circle = np.empty(shape=[circle_prec, 2])
    polygons: List[Polygon] = []

    for p in areas:
        # Parametric circle samples around center p
        for i in range(circle_prec):
            limit_circle[i, 0] = p[0] + radius * cos(i * 2 * pi / circle_prec)
            limit_circle[i, 1] = p[1] + radius * sin(i * 2 * pi / circle_prec)

        # Edge points inside the circle
        d, k_inside = tree_edges.query(p, k=len(edge))
        p_inside_circle = edge[k_inside[(d < radius)]]

        if p_inside_circle.size != 0:
            # For each circle sample, find closest inside-edge point
            tree_inside_circle = cKDTree(p_inside_circle)
            d_lim, k_limit = tree_inside_circle.query(limit_circle, k=1)

            # Preserve first-appearance order among uniques
            p_index, first_index, _ = np.unique(
                k_limit, return_index=True, return_counts=True
            )
            p_index_sorted = p_index[np.argsort(first_index)]
            edge_points = p_inside_circle[p_index_sorted]

            if len(edge_points) > 2:
                polygons.append(Polygon(edge_points))

    return polygons


def generation_hexa(l: float, d: float) -> np.ndarray:
    """Generate a compact hexagonal lattice within a square of side ``l``.

    Args:
        l: Side length of the square domain.
        d: Lattice spacing along x.

    Returns:
        (N, 2) array with hexagonal grid point coordinates.
    """
    hexa = np.asarray([[0.0, 0.0]])

    x1 = np.arange(0, l, d)
    x2 = np.arange(0.5, l, d)
    y = np.arange(0, l, (sqrt(3) / 2) * d)

    for i in range(int(len(y) / 2)):
        for a in range(0, len(x1)):
            point = np.asarray([x1[a], y[2 * i]])
            hexa = np.append(hexa, [point], axis=0)

        for b in range(len(x2)):
            point = np.asarray([x2[b], y[2 * i + 1]])
            hexa = np.append(hexa, [point], axis=0)

    return hexa[1:]


def select_area_points(
    areas: List[Polygon],
    particles: np.ndarray,
    inside: bool = True,
) -> np.ndarray:
    """Filter particles by membership inside a list of polygons.

    Args:
        areas: Polygons defining regions of interest.
        particles: (N, 2) array with particle coordinates.
        inside: If True, return particles inside; otherwise return particles outside.

    Returns:
        Filtered (N', 2) particle array.
    """
    particles_inside_idx: List[int] = []
    for poly in areas:
        for p in range(len(particles)):
            if poly.contains(Point(particles[p])):
                particles_inside_idx.append(p)

    if inside:
        return particles[particles_inside_idx]
    else:
        return np.delete(particles, particles_inside_idx, axis=0)


def select_inside_circle(
    particules: np.ndarray,
    centers: np.ndarray,
    radius_ratio: float = 1.0,
) -> np.ndarray:
    """Select indices of points inside circles centered at ``centers``.

    The circle radius is half the average nearest-neighbor distance between
    centers, divided by ``radius_ratio``.

    Args:
        particules: (N, 2) array with point coordinates to test.
        centers: (K, 2) array with circle centers.
        radius_ratio: Divides the base radius (default 1.0).

    Returns:
        Sorted unique indices (1D) of ``particules`` lying inside ≥1 circle.
    """
    tree_centers = cKDTree(centers)
    d_centers, _ = tree_centers.query(centers, k=2)
    radius = np.average(d_centers[:, 1]) / (2 * radius_ratio)
    print(radius)

    tree_particules = cKDTree(particules)
    d_particules, k_particules = tree_particules.query(
        centers, k=int(len(particules) / len(centers))
    )

    points_inside_circles = np.unique(k_particules[(d_particules < radius)])
    return points_inside_circles


def define_area_distance_and_radius(
    particles_positions: np.ndarray,
    area_distance: float,
    adjust_method: str = 'size',
    original_size: float = 1.0,
    obj_size: float = 0.2,
    obj_d_ave: float = 0.2,
    density_reduction_proportion: float = 0.5,
    keep_inside: bool = True,
) -> Tuple[List[Polygon], float, float]:
    """Compute polygons/radius achieving target spacing and coverage.

    Determines area centers (via correlated disorder) with average spacing
    ~ ``area_distance`` and adjusts polygon radius so the total covered area
    matches ``density_reduction_proportion`` of the (scaled) domain.

    Args:
        particles_positions: (N, 2) particle coordinates.
        area_distance: Target average distance between area centers.
        adjust_method: 'size' uses ``original_size/obj_size`` scaling;
            'distance' uses measured NN distance vs ``obj_d_ave``.
        original_size: Original characteristic size (for 'size' method).
        obj_size: Target characteristic size (for 'size' method).
        obj_d_ave: Target NN distance (for 'distance' method).
        density_reduction_proportion: Desired covered area fraction.
        keep_inside: If True, interior kept; else exterior kept.

    Returns:
        (polygons, average_center_distance, radius).
    """
    particles_tree = cKDTree(particles_positions)
    d, _ = particles_tree.query(particles_positions, k=7)
    d_ave = np.average(d[:, 1:])

    if adjust_method == 'size':
        adjust_value = original_size / obj_size
    elif adjust_method == 'distance':
        adjust_value = d_ave / obj_d_ave
    else:
        adjust_value = 1.0  # fallback

    density_adjust = density_reduction_proportion if keep_inside else 1 - density_reduction_proportion

    # Adjust domain size to targeted object size/distance
    limits = np.asarray(
        [np.max(particles_positions[:, 0]), np.max(particles_positions[:, 1])]
    ) * adjust_value
    N_zones = int((2 / (sqrt(3) * area_distance**2)) * (limits[0] * limits[1]))

    # Place area centers using correlated disorder so spacing ≈ area_distance
    while True:
        areas_c = Vor.make_correlated_disorder(N_zones, 20, 100, perodic_boundaries=True, max_xy=limits, save_to_csv=False)
        masque_tree = cKDTree(areas_c)
        d, _ = masque_tree.query(areas_c, k=2)
        d_ave = np.average(d[:, 1:])
        correc_factor = area_distance / d_ave
        print(correc_factor)
        if 0.99 < correc_factor < 1.01:
            break
        N_zones = int(N_zones / correc_factor**2)

    surface_to_cover = (limits[0] * limits[1]) * density_adjust
    N_edges = N_zones * 20

    # Edge set for polygons
    areas_e = Vor.make_correlated_disorder(N_edges, 20, 100, perodic_boundaries=True, max_xy=limits, save_to_csv=False)

    # Initial radius estimate (equal-area discs heuristic)
    r = sqrt(surface_to_cover / (N_zones * pi))

    # Adjust polygon area to match target total surface
    while True:
        areas_poly = define_areas_disorder_edge(areas_e, areas_c, r, 100)
        tot_area = 0.0
        for a in areas_poly:
            tot_area += a.area
        correction = surface_to_cover / tot_area
        print(correction)
        if 0.99 < correction < 1.01:
            break
        r *= sqrt(correction)

    return areas_poly, float(d_ave), float(r)


def create_areas(
    area_distance: float,
    limits: np.ndarray,
    density_reduction_proportion: float = 0.5,
) -> Tuple[List[Polygon], float, float]:
    """Create polygons with target center spacing and total coverage.

    Args:
        area_distance: Target average distance between area centers.
        limits: Box limits as [max_x, max_y].
        density_reduction_proportion: Desired covered area fraction.

    Returns:
        (polygons, average_center_distance, radius).
    """
    N_zones = int((2 / (sqrt(3) * area_distance**2)) * (limits[0] * limits[1]))

    while True:
        areas_c = Vor.make_correlated_disorder(N_zones, 20, 100, perodic_boundaries=True, max_xy=limits, save_to_csv=False)
        masque_tree = cKDTree(areas_c)
        d, _ = masque_tree.query(areas_c, k=2)
        d_ave = np.average(d[:, 1:])
        correc_factor = area_distance / d_ave
        print(correc_factor)
        if 0.99 < correc_factor < 1.01:
            break
        N_zones = int(N_zones / correc_factor**2)

    surface_to_cover = (limits[0] * limits[1]) * density_reduction_proportion
    N_edges = N_zones * 20
    areas_e = Vor.make_correlated_disorder(N_edges, 20, 100, perodic_boundaries=True, max_xy=limits, save_to_csv=False)
    r = sqrt(surface_to_cover / (N_zones * pi))

    while True:
        areas_poly = define_areas_disorder_edge(areas_e, areas_c, r, 100)
        tot_area = 0.0
        for a in areas_poly:
            tot_area += a.area
        correction = surface_to_cover / tot_area
        print(correction)
        if 0.99 < correction < 1.01:
            break
        r *= sqrt(correction)

    return areas_poly, float(d_ave), float(r)
