# -*- coding: utf-8 -*-
"""
Correlated disorder structure generation and density reduction.

Merged from:
  - Structure_generation/Correlated_disorder_generation.py (Created Thu Apr 30 2020)
  - Structure_generation/Density_reduction.py (Created Wed May 20 2020)

@author: Gil
"""

import numpy as np
from numpy import genfromtxt, sin, cos
from numpy.lib.scimath import sqrt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial import cKDTree
from numpy import pi
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# ---------------------------------------------------------------------------
# Correlated disorder generation
# ---------------------------------------------------------------------------

def select_boundary_points(structure_wo_borders, edge_size=1):
    """Takes edge_size(default = 1) times sqrt(total_number of points) from the
    each edge of the structure and moves them to the opposite creating
    ficticinal periodic boundaries"""
    N = len(structure_wo_borders)

    sorted_x_struc = structure_wo_borders[np.argsort(structure_wo_borders[:,0])]
    sorted_y_struc = structure_wo_borders[np.argsort(structure_wo_borders[:,1])]

    edgex_left_to_right = sorted_x_struc[:int(edge_size*N**0.5 + 1)]
    edgex_right_to_left = sorted_x_struc[(N - int(edge_size*N**0.5 + 1)):]
    edgey_low_to_up = sorted_y_struc[:int(edge_size*N**0.5 + 1)]
    edgex_up_to_low = sorted_y_struc[(N - int(edge_size*N**0.5 + 1)):]

    edgex_left_to_right[:, 0] += np.amax(structure_wo_borders[:, 0]) - np.amin(structure_wo_borders[:, 0])
    edgex_right_to_left[:, 0] -= np.amax(structure_wo_borders[:, 0]) - np.amin(structure_wo_borders[:, 0])
    edgey_low_to_up[:, 1] += np.amax(structure_wo_borders[:, 1]) - np.amin(structure_wo_borders[:, 1])
    edgex_up_to_low[:, 1] -= np.amax(structure_wo_borders[:, 1]) - np.amin(structure_wo_borders[:, 1])

    return edgex_left_to_right, edgex_right_to_left, edgey_low_to_up, edgex_up_to_low


def structure_w_boundaries(structure, return_edges=False):
    """Appends arrays in tuple edges to array structure, here we use it to
    append the calculated boundaries to the main structure"""

    edges = select_boundary_points(structure)
    struc_w_boundary = structure
    size_to_remove = len(edges)*len(edges[0])
    for array in edges:
        struc_w_boundary = np.append(struc_w_boundary, array, axis=0)

    if return_edges:
        return struc_w_boundary, size_to_remove, edges
    else:
        return struc_w_boundary, size_to_remove


def calculate_new_positions(structure, limits, mesh_precision=10):
    """Creates a linear 2D mesh. Calculates the points in mesh closest to each
    point in structure, it does the average of the coordinates of all the
    the closest points in mesh. This average becomes the new coordinates for
    the point in structure."""
    N = len(structure)
    t_grid = int((mesh_precision*N)**0.5)
    ones = np.ones(t_grid**2)
    sx, sy = np.mgrid[(0.0 - 2 / N**0.5):(1.0 + 2 / N**0.5)*limits[0]:t_grid * 1j,
                      (0.0 - 2 / N**0.5):(1.0 + 2 / N**0.5)*limits[1]:t_grid * 1j]
    s = np.c_[sx.ravel(), sy.ravel()]
    tree = cKDTree(structure)
    d, k = tree.query(s)
    m = np.bincount(k, weights=ones)
    m[m==0] = 1
    random_x = np.bincount(k, weights=s[:, 0])
    random_y = np.bincount(k, weights=s[:, 1])

    random_x = random_x / m
    random_y = random_y / m

    calculated_structure = np.vstack((random_x, random_y)).T

    return calculated_structure


def correct_edges(structure):

    structure[:,0] = structure[:,0] - np.min(structure[:,0])
    structure[:,1] = structure[:,1] - np.min(structure[:,1])

    return structure


def make_correlated_disorder(N, mesh_precision, iterations, perodic_boundaries=False, show_iteration=False, show_boundaries=False, max_xy=np.asarray([1,1])):
    """Generates a correlated disorder structure and plots the structure
    evolution and boundaries if required.
    The programs plots once for every iteration so it is not recomended for many
    iterations. To be used mostly for testing and check errors"""

    correlated_disorder_structure = np.random.rand(N, 2)
    correlated_disorder_structure[:,0] *= max_xy[0]
    correlated_disorder_structure[:,1] *= max_xy[1]
    print(len(correlated_disorder_structure))

    if perodic_boundaries:
        if show_iteration:
            if show_boundaries:
                for i in range(iterations):
                    print(f"{100*i/iterations} %")
                    current_iteration = correlated_disorder_structure
                    current_iteration, to_remove, edges = structure_w_boundaries(current_iteration, show_boundaries)
                    edgex_left, edgex_right, edgey_low, edgex_up = edges

                    figure(num=None, figsize=(12, 12))
                    plt.plot(correlated_disorder_structure[:,0], correlated_disorder_structure[:,1], '.')
                    plt.plot(edgex_left[:,0], edgex_left[:,1], '.')
                    plt.plot(edgex_right[:,0], edgex_right[:,1], '.')
                    plt.plot(edgey_low[:,0], edgey_low[:,1], '.')
                    plt.plot(edgex_up[:,0], edgex_up[:,1], '.')
                    plt.show()

                    current_iteration = calculate_new_positions(current_iteration, max_xy, mesh_precision)
                    correlated_disorder_structure = current_iteration[:-to_remove]

            if not(show_boundaries):
                for i in range(iterations):
                    print(f"{100*i/iterations} %")
                    current_iteration = correlated_disorder_structure
                    current_iteration, to_remove = structure_w_boundaries(current_iteration)

                    figure(num=None, figsize=(12, 12))
                    plt.plot(correlated_disorder_structure[:,0], correlated_disorder_structure[:,1], '.')
                    plt.show()

                    current_iteration = calculate_new_positions(current_iteration, max_xy, mesh_precision)
                    correlated_disorder_structure = current_iteration[:-to_remove]

        if not(show_iteration):
            for i in range(iterations):
                print(f"{100*i/iterations} %")
                current_iteration = correlated_disorder_structure
                current_iteration, to_remove = structure_w_boundaries(current_iteration)
                current_iteration = calculate_new_positions(current_iteration, max_xy, mesh_precision)
                correlated_disorder_structure = current_iteration[:-to_remove]

    if not(perodic_boundaries):
        if show_iteration:
            for i in range(iterations):
                print(f"{100*i/iterations} %")

                figure(num=None, figsize=(12, 12))
                plt.plot(correlated_disorder_structure[:,0], correlated_disorder_structure[:,1], '.')
                plt.axis('off')
                plt.axis('equal')
                plt.show()

                correlated_disorder_structure = calculate_new_positions(correlated_disorder_structure, max_xy, mesh_precision)

        if not(show_iteration):
            for i in range(iterations):
                print(f"{100*i/iterations} %")
                correlated_disorder_structure = calculate_new_positions(correlated_disorder_structure, max_xy, mesh_precision)

    correlated_disorder_structure = correct_edges(correlated_disorder_structure)

    return correlated_disorder_structure


# ---------------------------------------------------------------------------
# Density reduction
# ---------------------------------------------------------------------------

def load_positions_files(filename, location):

    particle_positions = genfromtxt(location + '/' + filename + '.csv', delimiter=',')

    return particle_positions


def define_areas_disorder_edge(edge, areas, radius, circle_prec):
    """Uses functions for the generation of correlated disorder structures to generate two structures.
    One will be used at the edges of the areas and the other will be used as the center of those areas."""

    tree_edges = cKDTree(edge)

    circle_steps = 100
    limit_circle = np.empty(shape=[circle_prec, 2])
    structure = []
    polygons = []

    for a, p in enumerate(areas):
        for i in range(circle_prec):
            limit_circle[i,0] = p[0] + radius*cos(i*2*pi/circle_prec)
            limit_circle[i,1] = p[1] + radius*sin(i*2*pi/circle_prec)

        d, k_inside = tree_edges.query(p, k=len(edge))
        p_inside_circle = edge[k_inside[(d < radius)]]

        if p_inside_circle.size != 0:
            tree_inside_circle = cKDTree(p_inside_circle)
            d, k_limit = tree_inside_circle.query(limit_circle, k=1)

            p_index, first_index, n_repetition = np.unique(k_limit, return_index=True, return_counts=True)
            p_index_sorted = p_index[np.argsort(first_index)]

            edge_points = p_inside_circle[p_index_sorted]

            if len(edge_points) > 2:
                polygons.append(Polygon(edge_points))

    return polygons


def generation_hexa(l, d):
    """Generates a hexagonal compact structure. approx_N is the approximate numebr of points the structure will have."""

    hexa = np.asarray([[0,0]])

    x1 = np.arange(0, l, d)
    x2 = np.arange(0.5, l, d)
    y = np.arange(0, l, (sqrt(3)/2)*d)

    for i in range(int(len(y)/2)):
        for a in range(0, len(x1)):
            point = np.asarray([x1[a], y[2*i]])
            hexa = np.append(hexa, [point], axis=0)

        for b in range(len(x2)):
            point = np.asarray([x2[b], y[2*i+1]])
            hexa = np.append(hexa, [point], axis=0)

    return hexa[1:]


def select_area_points(areas, particles, inside=True):
    """Return the particles that are inside the areas defined by the "areas" variable."""

    particles_inside = []
    for a in areas:
        for p in range(len(particles)):
            if a.contains(Point(particles[p])):
                particles_inside.append(p)

    if inside:
        particles = particles[particles_inside]

    else:
        particles = np.delete(particles, particles_inside, axis=0)

    return particles


def select_inside_circle(particules, centers, radius_ratio=1):
    """Selects particules inside a circle defined by the centers in the "centers" array and with radius equal to average distance
    between centers divided by 2 times the value of "radius_ratio" """

    tree_centers = cKDTree(centers)
    d_centers, k_centers = tree_centers.query(centers, k=2)
    radius = np.average(d_centers[:,1])/(2*radius_ratio)
    print(radius)

    tree_particules = cKDTree(particules)

    d_particules, k_particules = tree_particules.query(centers, k=int(len(particules)/len(centers)))

    points_inside_circles = np.unique(k_particules[(d_particules < radius)])

    return points_inside_circles


def define_area_distance_and_radius(particles_positions, area_distance, adjust_method='size', original_size=1, obj_size=0.2, obj_d_ave=0.2, density_reduction_proportion=0.5, keep_inside=True):

    particles_tree = cKDTree(particles_positions)

    d, k = particles_tree.query(particles_positions, k=7)

    d_ave = np.average(d[:,1:])

    if adjust_method == 'size':
        adjust_value = original_size/obj_size

    elif adjust_method == 'distance':
        adjust_value = d_ave/obj_d_ave

    if keep_inside:
        density_adjust = density_reduction_proportion

    elif not(keep_inside):
        density_adjust = 1 - density_reduction_proportion

    limits = np.asarray([np.max(particles_positions[:,0]), np.max(particles_positions[:,1])])*adjust_value
    N_zones = int((2/(sqrt(3)*area_distance**2))*(limits[0]*limits[1]))

    while True:

        areas_c = make_correlated_disorder(N_zones, 20, 100, perodic_boundaries=True, max_xy=limits)

        masque_tree = cKDTree(areas_c)

        d, k = masque_tree.query(areas_c, k=2)

        d_ave = np.average(d[:,1:])

        correc_factor = area_distance/d_ave
        print(correc_factor)

        if (correc_factor > 0.99 and correc_factor < 1.01):
            break

        N_zones = int(N_zones/correc_factor**2)

    surface_to_cover = (limits[0]*limits[1])*density_adjust

    N_edges = N_zones*20

    areas_e = make_correlated_disorder(N_edges, 20, 100, perodic_boundaries=True, max_xy=limits)

    r = sqrt((surface_to_cover)/(N_zones*pi))

    while True:

        areas_poly = define_areas_disorder_edge(areas_e, areas_c, r, 100)

        tot_area = 0
        for a in areas_poly:
            tot_area += a.area

        correction = surface_to_cover/tot_area
        print(correction)

        if (correction > 0.99 and correction < 1.01):
            break

        r *= sqrt(correction)

    return areas_poly, d_ave, r


def create_areas(area_distance, limits, density_reduction_proportion=0.5):

    N_zones = int((2/(sqrt(3)*area_distance**2))*(limits[0]*limits[1]))

    while True:

        areas_c = make_correlated_disorder(N_zones, 20, 100, perodic_boundaries=True, max_xy=limits)

        masque_tree = cKDTree(areas_c)

        d, k = masque_tree.query(areas_c, k=2)

        d_ave = np.average(d[:,1:])

        correc_factor = area_distance/d_ave
        print(correc_factor)

        if (correc_factor > 0.99 and correc_factor < 1.01):
            break

        N_zones = int(N_zones/correc_factor**2)

    surface_to_cover = (limits[0]*limits[1])*density_reduction_proportion

    N_edges = N_zones*20

    areas_e = make_correlated_disorder(N_edges, 20, 100, perodic_boundaries=True, max_xy=limits)

    r = sqrt((surface_to_cover)/(N_zones*pi))

    while True:

        areas_poly = define_areas_disorder_edge(areas_e, areas_c, r, 100)

        tot_area = 0
        for a in areas_poly:
            tot_area += a.area

        correction = surface_to_cover/tot_area
        print(correction)

        if (correction > 0.99 and correction < 1.01):
            break

        r *= sqrt(correction)

    return areas_poly, d_ave, r
