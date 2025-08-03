# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:38:28 2020

@author: Gil
"""

from numpy import genfromtxt
import Correlated_disorder_generation as Vor
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from matplotlib.pyplot import figure
from scipy import pi
from numpy import sin,cos
from numpy.lib.scimath import sqrt


def load_positions_files(filename, location):
    
    particle_positions = genfromtxt(location + '/' + filename + '.csv', delimiter=',')

    return particle_positions

def define_areas_disorder_edge(edge, areas, radius, circle_prec):
    """Uses functions for the generation  of correlated disorder structures to generate two structures.
    One will be used at the edges of the areas and the other will be used as the center of those areas."""
    
    tree_edges = cKDTree(edge)  # create tree of closest neighbours
    
    
    circle_steps = 100 
    limit_circle =np.empty(shape=[circle_prec,2])
    structure = []
    polygons = []
    
    
    for a, p in enumerate(areas):
        for i in range(circle_prec):
            limit_circle[i,0] = p[0] + radius*cos(i*2*pi/circle_prec);  
            limit_circle[i,1] = p[1] + radius*sin(i*2*pi/circle_prec);
        
        d, k_inside = tree_edges.query(p,k=len(edge))
        p_inside_circle =  edge[k_inside[(d < radius)]] #select points inside circle
        
        if p_inside_circle.size != 0:
            tree_inside_circle = cKDTree(p_inside_circle)
            d, k_limit = tree_inside_circle.query(limit_circle,k=1)#select points closest to the limiting 
            
            p_index,first_index,n_repetition = np.unique(k_limit,return_index=True,return_counts=True)
            p_index_sorted = p_index[np.argsort(first_index)]
            #p_index = p_index[n_repetition > circle_prec/10] 
            
            edge_points = p_inside_circle[p_index_sorted]
            
            if len(edge_points) > 2 :
                
                polygons.append(Polygon(edge_points))
                
    return polygons  
    

def generation_hexa(l, d):
    """ Generates a hexagonal compact structure. approx_N is the approximate numebr of points the structure will have."""
            
    
    hexa= np.asarray([[0,0]])
    
    x1 = np.arange(0,l,d)
    x2 = np.arange(0.5,l,d)
    y = np.arange(0,l,(sqrt(3)/2)*d)
    
    for i in range(int(len(y)/2)):             
        for a in range(0,len(x1)):
            point = np.asarray([x1[a],y[2*i]])
            hexa = np.append(hexa,[point],axis=0)
        
        for b in range(len(x2)):
            point = np.asarray([x2[b],y[2*i+1]])
            hexa = np.append(hexa,[point],axis=0)
            
    return hexa[1:]


def select_area_points(areas,particles, inside = True):
    """ Return the particles that are inside the areas defined by the "areas" variable."""
    
    particles_inside = []
    for a in areas:
        for p in range(len(particles)):
            if a.contains(Point(particles[p])):
                particles_inside.append(p) 
                    
    if inside:
        particles =  particles[particles_inside]
    
    else: 
        particles = np.delete(particles,particles_inside,axis = 0)
        
    return particles
    
    
def select_inside_circle(particules,centers,radius_ratio = 1):
    """Selects particules inside a circle defined by the centers in the "centers" array and with radius equal to average distance 
    between centers divided by 2 times the value of "radius_ratio" """
    
    tree_centers = cKDTree(centers)
    d_centers, k_centers = tree_centers.query(centers, k=2)
    radius = np.average(d_centers[:,1])/(2*radius_ratio)
    print(radius)
    
    tree_particules = cKDTree(particules)
    
    d_particules, k_particules = tree_particules.query(centers,k = int(len(particules)/len(centers)))
    
    points_inside_circles  = np.unique(k_particules[(d_particules < radius)])
   
    return points_inside_circles

def define_area_distance_and_radius(particles_positions, area_distance, adjust_method = 'size', original_size = 1, obj_size = 0.2, obj_d_ave = 0.2, density_reduction_proportion = 0.5, keep_inside = True):
    
    particles_tree = cKDTree(particles_positions)
    
    d, k = particles_tree.query(particles_positions,k=7)
    
    d_ave = np.average(d[:,1:])
    
    if adjust_method == 'size':
        
        adjust_value = original_size/obj_size
        
    elif adjust_method == 'distance':
        
        adjust_value = d_ave/obj_d_ave
        
    if keep_inside:
         
         density_adjust = density_reduction_proportion
     
    elif not(keep_inside):
         
         density_adjust = 1 - density_reduction_proportion
        
        
    limits = np.asarray([np.max(particles_positions[:,0]),np.max(particles_positions[:,1])])*adjust_value #adjust structure size to obtain targeted particule size or distance depending on method selected
    N_zones = int((2/(sqrt(3)*area_distance**2))*(limits[0]*limits[1]))
    
    #use correlated disorder to create position of areas for density reduction
    while True:     
    
        areas_c = Vor.make_correlated_disorder(N_zones, 20, 100, perodic_boundaries = True, max_xy= limits)
    
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

    #use correlated disorder to create edges of areas for density reduction      
    areas_e = Vor.make_correlated_disorder(N_edges, 20, 100, perodic_boundaries = True, max_xy= limits)
            
    r = sqrt((surface_to_cover)/(N_zones*pi))

    #iteratively adjust area surface to obtain targeted total surface    
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

def create_areas(area_distance, limits, density_reduction_proportion = 0.5):
    
    N_zones = int((2/(sqrt(3)*area_distance**2))*(limits[0]*limits[1]))
    
    while True:     
    
        areas_c = Vor.make_correlated_disorder(N_zones, 20, 100, perodic_boundaries = True, max_xy= limits)
    
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
            
    areas_e = Vor.make_correlated_disorder(N_edges, 20, 100, perodic_boundaries = True, max_xy= limits)
            
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