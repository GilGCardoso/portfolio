# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:29:48 2020

@author: Gil
"""

import os
import skimage
import matplotlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage.measure import label, regionprops, regionprops_table, find_contours
from skimage.morphology import remove_small_holes, remove_small_objects, binary_closing, disk, binary_erosion, binary_dilation, dilation, area_closing
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from scipy.spatial import cKDTree


def fetch_file(folder, filename, invert = False):    

    path = os.path.join(folder, filename)
    im = Image.open(path)
    pix = np.array(im)
    
    # Handle different channel layouts
    if pix.ndim == 2:
        # already grayscale
        img = skimage.img_as_float(pix)
    else:
        # if RGBA, drop alpha
        if pix.shape[-1] == 4:
            pix = pix[..., :3]
        img = rgb2gray(pix) 

    if invert:
        
        img = skimage.util.invert(img)

    return img


def binary_conversion(img, treshold_adjustment, window_size = 15, delta_blur = 0, sigma_weight = 0.2):

    if img is None:
        raise ValueError("Input image is None.")

    ksize = int(delta_blur) if delta_blur is not None else 0
    if ksize >= 2:
        f_img = cv2.blur(img, (ksize, ksize))
    else:
        f_img = img

    if window_size is None or int(window_size) < 3:
        raise ValueError("window_size must be at least 3.")

    threshold = threshold_sauvola(f_img, window_size=int(window_size), k=sigma_weight)
    binary_sauvola = f_img > (threshold + treshold_adjustment)

    plt.figure(num=None, figsize=(18,12), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(binary_sauvola, cmap='gray')
    plt.axis('off')
    plt.show()

    return binary_sauvola


def area_selection(bin_image, min_area, max_area):

    label_img = label(bin_image, connectivity=None)

    props = regionprops_table(label_img, properties = ['area','centroid','perimeter', 'equivalent_diameter', 'major_axis_length', 'solidity'])


    inf_values = props['area'] < min_area
    sup_values = props['area'] > max_area
    extended_shape = props['major_axis_length'] > 1.5*props['equivalent_diameter']
    hollow = props['solidity'] < np.average(props['solidity'])*0.9

    logical_to_keep = np.logical_not(np.logical_or(np.logical_or(np.logical_or(inf_values,sup_values),extended_shape),hollow))

    value_areas_keep = np.asarray(np.where(logical_to_keep))[0,:]
    
    diameter = props['equivalent_diameter'][logical_to_keep].reshape([len(value_areas_keep),1])
    centroid = np.transpose(np.asarray([props['centroid-1'][logical_to_keep],props['centroid-0'][logical_to_keep]]))

    selected_areas = np.append(centroid,diameter,axis = 1)
    
    final_areas = np.isin(label_img, value_areas_keep+1).astype(int)
    
    fig, axs = plt.subplots(1, 2, num=None, figsize=(18,12), dpi=100, facecolor='w', edgecolor='k')
    
    axs[0].imshow(final_areas,cmap = 'gray')
    axs[0].axis('off')
    axs[1].imshow(bin_image - final_areas,cmap = 'gray')
    axs[1].axis('off')
    
    plt.show()
    
    return selected_areas, final_areas, props


def visualise_selected_particules(particules,img):
    
    fig, ax = plt.subplots(num=4, figsize=(18,12), dpi=80, facecolor='w', edgecolor='k')
    ax.imshow(img)

    for p in particules:     
        circle = plt.Circle((p[0],p[1]), radius = p[2]/2, edgecolor = 'black', fill = False)
        ax.add_artist(circle)
    
    plt.axis('off')
    plt.show()


def save_positions_array(filename, location, particule_positions, norm_factor):

    np.savetxt(os.path.join(location, filename + '.csv'), particule_positions[:, :2] * norm_factor, delimiter=',') 
    

def detect_angle_orientation(particles, label):
    
    particles_tree = cKDTree(particles)
    
    d, k = particles_tree.query(particles,k=2)
    
    distance = (particles[k[:,1]] - particles)
    den = np.where(d[:,1] == 0, np.finfo(float).eps, d[:,1])
    ratio = np.clip(distance[:,1] / den, -1.0, 1.0)
    
    angles_degrees = np.arcsin(ratio) * 180/np.pi
    
    #identify quandrant position
    
    #1st quadrant do nothing

    second_and_third_q = distance[:,0] < 0
    angles_degrees[second_and_third_q] = 180 - angles_degrees[second_and_third_q]
    
    fourth_q = np.logical_and(distance[:,0] >= 0, distance[:,1] < 0)
    angles_degrees[fourth_q] = 360 + angles_degrees[fourth_q]
    
    #set between 0 and 60
    angles_degrees[angles_degrees >= 300] = angles_degrees[angles_degrees >= 300] - 300
    angles_degrees[angles_degrees >= 240] = angles_degrees[angles_degrees >= 240] - 240
    angles_degrees[angles_degrees >= 180] = angles_degrees[angles_degrees >= 180] - 180
    angles_degrees[angles_degrees >= 120] = angles_degrees[angles_degrees >= 120] - 120
    angles_degrees[angles_degrees >= 60] = angles_degrees[angles_degrees >= 60] - 60
    
    
    angles_rad = angles_degrees * np.pi/180
    
    vector_point = np.asarray([np.cos(angles_rad),np.sin(angles_rad)]).transpose()
    vector_end_point = particles + (10*vector_point)
    particles_direction = np.append(particles,vector_end_point, axis=1)
    
    return particles_direction, angles_degrees

def detect_angle_variation(particles, angles, treshold):
    if len(particles) == 0:
        return particles

    particles_tree = cKDTree(particles)

    k_neighbors = min(7, len(particles))
    if k_neighbors <= 1:
        return particles

    d, k = particles_tree.query(particles, k=k_neighbors)

    neighbor_idx = k[:, 1:] if k_neighbors > 1 else np.empty((len(particles), 0), dtype=int)
    if neighbor_idx.shape[1] == 0:
        return particles

    diff = np.abs(angles[neighbor_idx] - angles.reshape((len(angles), 1)))

    num_small_diff = np.sum(np.logical_or(diff < treshold, diff > 60 - treshold), axis=1)

    ordered_particules = particles[num_small_diff > 2]

    return ordered_particules

def detect_distance_variation(particles):
    if len(particles) <= 1:
        return particles

    particles_tree = cKDTree(particles)

    d, k = particles_tree.query(particles, k=min(2, len(particles)))
    if d.ndim != 2 or d.shape[1] < 2:
        return particles

    d_median = np.median(d[:, 1])
    print(d_median)

    k2 = min(7, len(particles))
    if k2 <= 1:
        return particles

    d, k = particles_tree.query(particles, k=k2)
    if d.ndim != 2 or d.shape[1] < 2:
        return particles

    particle_dist_average = np.average(d[:, 1:], axis=1)

    ordered_particles = particles[np.logical_and(particle_dist_average > 0.9 * d_median,
                                                particle_dist_average < 1.1 * d_median)]

    return ordered_particles
