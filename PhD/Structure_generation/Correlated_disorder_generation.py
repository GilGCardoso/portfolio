# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:03:00 2020

@author: Gil
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.pyplot import figure


#define grid

def select_boundary_points (structure_wo_borders,edge_size = 1):
    """Takes edge_size(default = 1) times sqrt(total_number of points) from the
    each edge of the structure and moves them to the opposite creating 
    ficticinal periodic boundaries"""
    N = len(structure_wo_borders)
    
    sorted_x_struc = structure_wo_borders[np.argsort(structure_wo_borders[:,0])] # structure array sorted by values of x
    sorted_y_struc = structure_wo_borders[np.argsort(structure_wo_borders[:,1])] # structure array sorted by values of y

    # select value at edge of structure
    edgex_left_to_right = sorted_x_struc[:int(edge_size*N**0.5 + 1)]
    edgex_right_to_left = sorted_x_struc[(N - int(edge_size*N**0.5 + 1)):]
    edgey_low_to_up = sorted_y_struc[:int(edge_size*N**0.5 + 1)]
    edgex_up_to_low = sorted_y_struc[(N - int(edge_size*N**0.5 + 1)):]

    # move boundary points to the oposite side        
    edgex_left_to_right[:, 0] += np.amax(structure_wo_borders[:, 0]) - np.amin(structure_wo_borders[:, 0])
    edgex_right_to_left[:, 0] -= np.amax(structure_wo_borders[:, 0]) - np.amin(structure_wo_borders[:, 0])
    edgey_low_to_up[:, 1] += np.amax(structure_wo_borders[:, 1]) - np.amin(structure_wo_borders[:, 1])
    edgex_up_to_low[:, 1] -= np.amax(structure_wo_borders[:, 1]) - np.amin(structure_wo_borders[:, 1])
    
    return edgex_left_to_right, edgex_right_to_left, edgey_low_to_up, edgex_up_to_low

# create copy of opposite side for periodic like structure
def structure_w_boundaries(structure, return_edges = False):
    """Appends arrays in tuple edges to array structure, here we use it to
    append the calculated boundaries to the main structure"""
    
    edges = select_boundary_points(structure)
    struc_w_boundary = structure
    size_to_remove = len(edges)*len(edges[0])
    for array in edges:
        struc_w_boundary = np.append(struc_w_boundary,array,axis=0)
    
    if return_edges:
       return struc_w_boundary, size_to_remove, edges 
    else:
        return struc_w_boundary, size_to_remove


def calculate_new_positions(structure, limits, mesh_precision = 10):
    """creates a linear 2D mesh. Calculates the points in mesh closest to each
    point in structure, it does the average of the coordinates of all the
    the closest points in mesh. This average becomes the new coordinates for 
    the point in structure."""
    N = len(structure)
    t_grid = int((mesh_precision*N)**0.5)
    ones = np.ones(t_grid**2)
    sx, sy = np.mgrid[(0.0 - 2 / N**0.5):(1.0 + 2 / N**0.5)*limits[0]:t_grid * 1j,
                      (0.0 - 2 / N**0.5):(1.0 + 2 / N**0.5)*limits[1]:t_grid * 1j]
    s = np.c_[sx.ravel(), sy.ravel()]
    tree = cKDTree(structure)  # create tree of closest neighbours
    d, k = tree.query(s)  # get closest neighbours from struc to each grid point
    m = np.bincount(k, weights=ones)  # count how many times each points of struc is a closest neighbourt
    m[m==0]=1
    random_x = np.bincount(k, weights=s[:, 0])  # sum the coordinates of all points of the grid that have the same closest neighbour from struc
    random_y = np.bincount(k, weights=s[:, 1])
    
    random_x = random_x / m  # normalise x coordinate
    random_y = random_y / m
    
    calculated_structure = np.vstack((random_x,random_y)).T
    
    return calculated_structure
    
def correct_edges(structure):
    
    structure[:,0] = structure[:,0] - np.min(structure[:,0])
    structure[:,1] = structure[:,1] - np.min(structure[:,1])
    
    return structure

def make_correlated_disorder(N, mesh_precision, iterations, perodic_boundaries = False, show_iteration = False, show_boundaries = False, max_xy = np.asarray([1,1])):
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
                    current_iteration, to_remove, edges = structure_w_boundaries(current_iteration,show_boundaries)
                    edgex_left, edgex_right, edgey_low, edgex_up = edges
                    
                    #plot structure evolution
                    figure(num=None, figsize=(12, 12))
                    plt.plot(correlated_disorder_structure[:,0], correlated_disorder_structure[:,1], '.')
                    plt.plot(edgex_left[:,0], edgex_left[:,1], '.')
                    plt.plot(edgex_right[:,0], edgex_right[:,1], '.')
                    plt.plot(edgey_low[:,0], edgey_low[:,1], '.')
                    plt.plot(edgex_up[:,0], edgex_up[:,1], '.')
                    #plt.axis('off')
                    #plt.axis('equal')
                    plt.show()
                    
                    current_iteration = calculate_new_positions(current_iteration, max_xy, mesh_precision)
                    correlated_disorder_structure = current_iteration[:-to_remove]
                    
            if not(show_boundaries):
                for i in range(iterations):
                    print(f"{100*i/iterations} %")
                    current_iteration = correlated_disorder_structure
                    current_iteration, to_remove = structure_w_boundaries(current_iteration)
                    
                    #plot structure evolution
                    figure(num=None, figsize=(12, 12))
                    plt.plot(correlated_disorder_structure[:,0], correlated_disorder_structure[:,1], '.')
                    #plt.axis('off')
                    #plt.axis('equal')
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
                    
                #plot structure evolution
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