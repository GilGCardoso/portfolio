# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:03:00 2020
Updated with type hints, improved docstrings, and logging.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.pyplot import figure
from typing import Tuple

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def select_boundary_points(
    structure_wo_borders: np.ndarray,
    edge_size: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Select points along the edges of the structure and shift them to the opposite side
    to simulate periodic boundaries.
    """
    N = len(structure_wo_borders)

    sorted_x_struc = structure_wo_borders[np.argsort(structure_wo_borders[:, 0])]
    sorted_y_struc = structure_wo_borders[np.argsort(structure_wo_borders[:, 1])]

    edgex_left_to_right = sorted_x_struc[:int(edge_size * N ** 0.5 + 1)]
    edgex_right_to_left = sorted_x_struc[-int(edge_size * N ** 0.5 + 1):]
    edgey_low_to_up = sorted_y_struc[:int(edge_size * N ** 0.5 + 1)]
    edgex_up_to_low = sorted_y_struc[-int(edge_size * N ** 0.5 + 1):]

    x_range = np.ptp(structure_wo_borders[:, 0])
    y_range = np.ptp(structure_wo_borders[:, 1])

    edgex_left_to_right[:, 0] += x_range
    edgex_right_to_left[:, 0] -= x_range
    edgey_low_to_up[:, 1] += y_range
    edgex_up_to_low[:, 1] -= y_range

    return edgex_left_to_right, edgex_right_to_left, edgey_low_to_up, edgex_up_to_low


def structure_w_boundaries(
    structure: np.ndarray,
    return_edges: bool = False
) -> Tuple[np.ndarray, int, Tuple[np.ndarray, ...] | None]:
    """
    Append boundary points to the structure to create a periodic-like structure.
    """
    edges = select_boundary_points(structure)
    struc_w_boundary = structure.copy()
    size_to_remove = len(edges) * len(edges[0])

    for array in edges:
        struc_w_boundary = np.append(struc_w_boundary, array, axis=0)

    if return_edges:
        return struc_w_boundary, size_to_remove, edges
    return struc_w_boundary, size_to_remove, None


def calculate_new_positions(
    structure: np.ndarray,
    limits: np.ndarray,
    mesh_precision: int = 10
) -> np.ndarray:
    """
    Recalculate positions by averaging mesh points nearest to each structure point.
    """
    N = len(structure)
    t_grid = int((mesh_precision * N) ** 0.5)
    ones = np.ones(t_grid ** 2)

    sx, sy = np.mgrid[
        (0.0 - 2 / N ** 0.5):(1.0 + 2 / N ** 0.5) * limits[0]:t_grid * 1j,
        (0.0 - 2 / N ** 0.5):(1.0 + 2 / N ** 0.5) * limits[1]:t_grid * 1j
    ]
    s = np.c_[sx.ravel(), sy.ravel()]

    tree = cKDTree(structure)
    _, k = tree.query(s)

    m = np.bincount(k, weights=ones)
    m[m == 0] = 1

    random_x = np.bincount(k, weights=s[:, 0]) / m
    random_y = np.bincount(k, weights=s[:, 1]) / m

    return np.vstack((random_x, random_y)).T


def correct_edges(structure: np.ndarray) -> np.ndarray:
    """
    Shift structure so that minimum x and y are zero.
    """
    structure[:, 0] -= np.min(structure[:, 0])
    structure[:, 1] -= np.min(structure[:, 1])
    return structure


def make_correlated_disorder(
    N: int,
    mesh_precision: int,
    iterations: int,
    perodic_boundaries: bool = False,
    show_iteration: bool = False,
    show_boundaries: bool = False,
    max_xy: np.ndarray = np.asarray([1, 1]),
    save_to_csv: bool = True
) -> np.ndarray:
    """
    Generate a correlated disorder structure.
    """
    correlated_disorder_structure = np.random.rand(N, 2)
    correlated_disorder_structure[:, 0] *= max_xy[0]
    correlated_disorder_structure[:, 1] *= max_xy[1]

    if perodic_boundaries:
        for i in range(iterations):
            logger.info(f"{100 * i / iterations} %")
            if show_boundaries:
                current_iteration, to_remove, edges = structure_w_boundaries(
                    correlated_disorder_structure, return_edges=True
                )
                if show_iteration:
                    edgex_left, edgex_right, edgey_low, edgex_up = edges
                    figure(figsize=(8, 8))
                    plt.plot(correlated_disorder_structure[:, 0], correlated_disorder_structure[:, 1], '.')
                    plt.plot(edgex_left[:, 0], edgex_left[:, 1], '.')
                    plt.plot(edgex_right[:, 0], edgex_right[:, 1], '.')
                    plt.plot(edgey_low[:, 0], edgey_low[:, 1], '.')
                    plt.plot(edgex_up[:, 0], edgex_up[:, 1], '.')
                    plt.axis('off')
                    plt.show()
            else:
                current_iteration, to_remove, _ = structure_w_boundaries(correlated_disorder_structure)
                if show_iteration:
                    figure(figsize=(8, 8))
                    plt.plot(correlated_disorder_structure[:, 0], correlated_disorder_structure[:, 1], '.')
                    plt.axis('off')
                    plt.show()

            current_iteration = calculate_new_positions(current_iteration, max_xy, mesh_precision)
            correlated_disorder_structure = current_iteration[:-to_remove]
    else:
        for i in range(iterations):
            logger.info(f"{100 * i / iterations} %")
            if show_iteration:
                figure(figsize=(8, 8))
                plt.plot(correlated_disorder_structure[:, 0], correlated_disorder_structure[:, 1], '.')
                plt.axis('off')
                plt.axis('equal')
                plt.show()
            correlated_disorder_structure = calculate_new_positions(
                correlated_disorder_structure, max_xy, mesh_precision
            )

    if save_to_csv:
        np.savetxt("correlated_structure.csv", correct_edges(correlated_disorder_structure), delimiter=",")

    return correct_edges(correlated_disorder_structure)