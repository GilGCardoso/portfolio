# -*- coding: utf-8 -*-
"""
SEM image processing: load images, threshold, select particle regions,
and extract particle positions and orientations.
"""

import cv2
import skimage
import matplotlib
import numpy as np
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


def fetch_file(folder, file, invert=False):
    """Load an image from disk and convert it to a grayscale float array.

    Parameters
    ----------
    folder : str
        Directory containing the image file.
    file : str
        Filename (including extension).
    invert : bool, optional
        If True, invert pixel intensities before returning (default False).

    Returns
    -------
    numpy.ndarray
        2-D float array of grayscale pixel values in [0, 1].
    """
    im = Image.open(folder + "/" + file)
    pix = np.array(im)
    img = rgb2gray(pix)

    if invert:
        img = skimage.util.invert(img)

    return img


def binary_conversion(img, threshold_adjustment, window_size=15, delta_blur=0, sigma_weight=0.2, show_preview=True):
    """Apply Sauvola adaptive thresholding to produce a binary image.

    Parameters
    ----------
    img : numpy.ndarray
        Grayscale input image.
    threshold_adjustment : float
        Scalar offset added to the Sauvola threshold before binarising.
    window_size : int, optional
        Local neighbourhood size for Sauvola (default 15).
    delta_blur : int, optional
        Box-blur kernel size applied before thresholding; 0 disables blur (default 0).
    sigma_weight : float, optional
        Sauvola k-parameter controlling sensitivity to local variance (default 0.2).
    show_preview : bool, optional
        Display the thresholded image (default True). Set to False for
        non-interactive or batch usage.

    Returns
    -------
    numpy.ndarray
        Boolean binary image (True = foreground).
    """
    f_img = img if delta_blur == 0 else cv2.blur(img, (delta_blur, delta_blur))
    threshold = threshold_sauvola(f_img, window_size=window_size, k=sigma_weight)
    binary_sauvola = f_img > threshold + threshold_adjustment

    if show_preview:
        plt.figure(num=None, figsize=(18,12), dpi=100, facecolor='w', edgecolor='k')
        plt.imshow(binary_sauvola, cmap='gray')
        plt.axis('off')
        plt.show()

    return binary_sauvola


def area_selection(bin_image, min_area, max_area):
    """Filter labelled regions by area, shape, and solidity.

    Regions are discarded if they fall outside [min_area, max_area], are
    elongated (major axis > 1.5× equivalent diameter), or are substantially
    less solid than average (< 90 % of mean solidity).

    Parameters
    ----------
    bin_image : numpy.ndarray
        Boolean binary image produced by :func:`binary_conversion`.
    min_area : int
        Minimum acceptable region area in pixels.
    max_area : int
        Maximum acceptable region area in pixels.

    Returns
    -------
    selected_areas : numpy.ndarray, shape (M, 3)
        Array of [x, y, diameter] for each accepted particle.
    final_areas : numpy.ndarray
        Binary mask containing only the accepted regions.
    props : dict
        Raw regionprops table (all regions, before filtering).
    """
    label_img = label(bin_image, connectivity=None)

    props = regionprops_table(label_img, properties=['area', 'centroid', 'equivalent_diameter', 'major_axis_length', 'solidity'])

    inf_values    = props['area'] < min_area
    sup_values    = props['area'] > max_area
    extended_shape = props['major_axis_length'] > 1.5 * props['equivalent_diameter']
    hollow        = props['solidity'] < np.average(props['solidity']) * 0.9

    logical_to_keep  = ~(inf_values | sup_values | extended_shape | hollow)
    value_areas_keep = np.where(logical_to_keep)[0]

    diameter = props['equivalent_diameter'][logical_to_keep].reshape([len(value_areas_keep), 1])
    centroid = np.transpose(np.asarray([props['centroid-1'][logical_to_keep], props['centroid-0'][logical_to_keep]]))

    selected_areas = np.append(centroid, diameter, axis=1)

    # Lookup-table mask: O(pixels) instead of O(pixels × regions)
    lut = np.zeros(label_img.max() + 1, dtype=bool)
    lut[value_areas_keep + 1] = True
    final_areas = lut[label_img].astype(int)

    fig, axs = plt.subplots(1, 2, num=None, figsize=(18,12), dpi=100, facecolor='w', edgecolor='k')

    axs[0].imshow(final_areas, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title(f'Kept ({len(value_areas_keep)} particles)', fontsize=14)

    axs[1].imshow(bin_image - final_areas, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f'Removed ({int(logical_to_keep.size - len(value_areas_keep))} regions)', fontsize=14)

    plt.tight_layout()
    plt.show()

    return selected_areas, final_areas, props


def visualise_selected_particles(particles, img):
    """Overlay detected particle circles on the original image.

    Parameters
    ----------
    particles : numpy.ndarray, shape (N, 3)
        Array of [x, y, diameter] for each particle (as returned by
        :func:`area_selection`).
    img : numpy.ndarray
        Original grayscale or colour image used as the background.
    """
    fig, ax = plt.subplots(num=4, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    ax.imshow(img)

    for p in particles:
        circle = plt.Circle((p[0], p[1]), radius=p[2] / 2, edgecolor='black', fill=False)
        ax.add_artist(circle)

    plt.axis('off')
    plt.show()


def save_positions_array(filename, location, particle_positions, norm_factor):
    """Save particle (x, y) positions to a CSV file, scaled by norm_factor.

    Parameters
    ----------
    filename : str
        Output filename without extension.
    location : str
        Directory path for the output file.
    particle_positions : numpy.ndarray, shape (N, 3)
        Particle array [x, y, diameter]; only the first two columns are saved.
    norm_factor : float
        Scale factor applied to coordinates before saving (e.g. nm/pixel).
    """
    np.savetxt(location + "/" + filename + '.csv', particle_positions[:, :2] * norm_factor, delimiter=',')


def detect_angle_orientation(particles, label):
    """Compute the orientation angle of each particle relative to its nearest neighbour.

    Angles are mapped into the [0°, 60°) range to exploit hexagonal symmetry.

    Parameters
    ----------
    particles : numpy.ndarray, shape (N, 2)
        Particle (x, y) coordinates.
    label : str
        Identifier string used externally to annotate the result (not used
        internally — kept for API compatibility).

    Returns
    -------
    particles_direction : numpy.ndarray, shape (N, 4)
        Each row is [x, y, end_x, end_y] where the endpoint marks the
        orientation direction (10 px arrow length).
    angles_degrees : numpy.ndarray, shape (N,)
        Orientation angle for each particle in degrees, in [0, 60).
    """
    particles_tree = cKDTree(particles)

    d, k = particles_tree.query(particles, k=2)

    distance = particles[k[:, 1]] - particles
    ratio = distance[:, 1] / d[:, 1]
    angles_degrees = np.arcsin(ratio) * 180 / np.pi

    # Identify quadrant and unwrap to [0°, 360°)
    # 1st quadrant: no correction needed
    second_and_third_q = distance[:, 0] < 0
    angles_degrees[second_and_third_q] = 180 - angles_degrees[second_and_third_q]

    fourth_q = np.logical_and(distance[:, 0] >= 0, distance[:, 1] < 0)
    angles_degrees[fourth_q] = 360 + angles_degrees[fourth_q]

    # Fold into [0°, 60°) using hexagonal symmetry (60° periodicity)
    for cutoff in [300, 240, 180, 120, 60]:
        angles_degrees[angles_degrees >= cutoff] -= cutoff

    angles_rad = angles_degrees * np.pi / 180

    vector_point = np.asarray([np.cos(angles_rad), np.sin(angles_rad)]).transpose()
    vector_end_point = particles + (10 * vector_point)
    particles_direction = np.append(particles, vector_end_point, axis=1)

    return particles_direction, angles_degrees


def detect_angle_variation(particles, angles, threshold):
    """Keep only particles whose nearest neighbours share a similar orientation.

    A particle is retained if more than 2 of its 6 nearest neighbours have an
    orientation angle within ``threshold`` degrees (accounting for 60° symmetry).

    Parameters
    ----------
    particles : numpy.ndarray, shape (N, 2)
        Particle (x, y) coordinates.
    angles : numpy.ndarray, shape (N,)
        Orientation angle for each particle in degrees (as returned by
        :func:`detect_angle_orientation`).
    threshold : float
        Angular tolerance in degrees.

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        Subset of particles with consistent local orientation.
    """
    particles_tree = cKDTree(particles)

    d, k = particles_tree.query(particles, k=7)

    diff = abs(angles[k[:, 1:]] - angles[:].reshape((len(angles), 1)))

    num_small_diff = np.sum(np.logical_or(diff < threshold, diff > 60 - threshold), axis=1)

    return particles[num_small_diff > 2]


def detect_distance_variation(particles):
    """Keep only particles whose average neighbour distance is close to the median.

    Retains particles whose mean distance to their 6 nearest neighbours lies
    within ±10 % of the global nearest-neighbour median, flagging locally
    disordered or isolated particles for removal.

    Parameters
    ----------
    particles : numpy.ndarray, shape (N, 2)
        Particle (x, y) coordinates.

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        Subset of particles with consistent inter-particle spacing.
    """
    particles_tree = cKDTree(particles)

    d, k = particles_tree.query(particles, k=2)
    d_median = np.median(d[:, 1])

    d, k = particles_tree.query(particles, k=7)
    particle_dist_average = np.average(d[:, 1:], axis=1)

    return particles[np.logical_and(
        particle_dist_average > 0.9 * d_median,
        particle_dist_average < 1.1 * d_median,
    )]
