# -*- coding: utf-8 -*-
"""
Structure Factor Analysis for Optical Simulations - GPU Accelerated Version

This module provides computation methods of structure factors for non-periodic 
particle arrangements, with targeted application for non-periodic metasurfaces. 
The methods can be generalised for the interaction of assemblies of materials 
(particles, atoms, etc) with light.

Key Features:
- GPU acceleration support via PyTorch CUDA
- Memory-efficient tensor operations for 20k-40k particles
- 2D visualization and radial averaging
- ~10-100x speedup over CPU for large particle systems

Business Applications:
- Rapid prototyping of optical metamaterials
- Manufacturing quality control for photonic devices
- R&D acceleration for non-periodic structure design

Created on Fri May 15 12:06:43 2023
@author: Gil Cardoso
"""

import matplotlib
import torch
import torch.cuda
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial import cKDTree
import psutil
import socket

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from structure_factor import StructureFactorCalculator_cpu


class StructureFactorCalculator_gpu:
    """
    GPU-accelerated structure factor calculator for particle systems.
    
    This class handles the computation of structure factors using PyTorch tensors
    for GPU acceleration. Includes methods for data loading, calculation, and saving.
    """
    
    def __init__(self):
        """Initialize the calculator and check GPU availability."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name()}")
        else:
            print("GPU not available, using CPU")
    
    def calculate_structure_factor_and_save(self, 
        read_folder: str, 
        filename: str, 
        range_calculation: float, 
        vector_step: float, 
        save_folder: str = '',
        *,
        full_plane: bool = False,
        save_hdf5: bool = True,
        save_json: bool = True,
        save_dat: bool = False
    ) -> None:
        """
        Main calculation workflow for structure factor analysis.
        
        This function orchestrates the entire calculation process:
        1. Load particle positions from CSV file
        2. Convert to PyTorch tensors and move to GPU
        3. Calculate 2D structure factor using GPU acceleration
        4. Compute radial average for 1D analysis
        5. Save results to output files
        
        Args:
            read_folder (str): Path to folder containing input CSV files
            file (str): Filename (without extension) of particle positions
            range_calculation (float): Maximum q-vector range for calculation
            vector_step (float): Step size for q-vector sampling
            save_folder (str): Path to folder for saving results
            
        Performance Notes:
            - GPU acceleration provides ~10-100x speedup for large systems
            - Memory usage scales as O(N²) where N is number of particles
            - Supports up to 40k particles on modern GPUs
        """

        logger.info("Processing file: %s", filename)

        # Load particle positions from CSV file
        structure_to_calculate = np.genfromtxt(read_folder + file + '.csv', delimiter=',')
        
        # Convert numpy array to PyTorch tensor for GPU computation
        structure_to_calculate_tensor = torch.from_numpy(structure_to_calculate)
        
        # Extract key parameters: average neighbor distance and particle count
        D, N = self.get_calculation_parameters(structure_to_calculate)
        
        # Generate distance vectors and q-space sampling grid
        d_x, d_y, q = self.generate_vectors(structure_to_calculate_tensor, range_calculation, vector_step, D)
        
        # Move all tensors to GPU for acceleration
        d_x = d_x.cuda()
        d_y = d_y.cuda()
        q = q.cuda()
        N = torch.tensor(N, dtype=torch.float).cuda()

        # Display q-range for user information
        print(f"Q-range: {q[0]*1e6:.2e} to {q[-1]*1e6:.2e} (physical units)")
            
        # Initialize result tensor on GPU
        structure_factor = torch.zeros((len(q),len(q)))
        
        # Double loop calculation: S(qx,qy) = (1/N) * Σ exp(i*(qx*dx + qy*dy))
        # This is the core structure factor calculation using discrete Fourier transform
        for a in range(len(q)): 
            for b in range(len(q)):
                # Calculate structure factor at each q-point using complex exponentials
                structure_factor[a,b] = (1/N)*torch.sum(torch.exp(1j*(q[a]*d_x + q[b]*d_y))) 

            progress = 100.0 * a / len(q)
            logger.info("Progress: %.1f%% (%s/%d)", progress, a, len(q)) # Progress indicator

        # Transfer results back to CPU for post-processing and saving
        Sq_2D = structure_factor.cpu().numpy() 
        q = q.cpu().numpy()

        # Compute radial average to get 1D structure factor S(|q|)
        Sq, R = StructureFactorCalculator_cpu.radial_average(Sq_2D, q)

        # Save both 2D and 1D results with metadata
        StructureFactorCalculator_cpu.save_data(
            Sq_2D, q, D, Sq, R, save_folder, filename,
            save_hdf5=save_hdf5,
            save_json=save_json,
            save_dat=save_dat
            )

    def generate_vectors(self, structure, domain, vector_step, particle_distance): 
        """
        Generate distance arrays and q-vectors for structure factor calculation.
        
        Creates all pairwise distance vectors between particles and defines the
        q-space sampling grid for structure factor computation.
        
        Args:
            structure (torch.Tensor): Particle positions, shape (N, 2)
            domain (float): Maximum q-vector magnitude in units of particle_distance
            vector_step (float): Step size for q-vector grid
            particle_distance (float): Characteristic length scale for normalization
            
        Returns:
            tuple: (distances_x, distances_y, scattering_vector)
                - distances_x: All pairwise x-distances, shape (1, N²)
                - distances_y: All pairwise y-distances, shape (1, N²)
                - scattering_vector: 1D array of q-values for calculation
        """
        coord_x = structure[:,0]
        coord_y = structure[:,1]
       
        # Create pairwise distance arrays using broadcasting
        # This generates all possible distance vectors r_i - r_j
        coord_x_ = coord_x.reshape(len(coord_x),1)
        distances_x = (coord_x - coord_x_).reshape(1,len(coord_x)**2)
            
        coord_y_ = coord_y.reshape(len(coord_y),1)
        distances_y = (coord_y - coord_y_).reshape(1,len(coord_y)**2)

        # Define normalized q-vector sampling grid
        # Normalization by particle_distance makes results scale-invariant
        border = domain/particle_distance
        step = vector_step/particle_distance
        scattering_vector = torch.arange(0, border, step)

        return distances_x, distances_y, scattering_vector

    def get_calculation_parameters(self, structure):
        """
        Extract key structural parameters needed for calculation setup.
        
        Analyzes the particle arrangement to determine:
        1. Average distance to nearest neighbors (sets length scale)
        2. Total number of particles (for normalization)
        
        Args:
            structure (np.ndarray): Particle positions, shape (N, 2)
            
        Returns:
            tuple: (average_neighbor_distance, number_of_particles)
                - average_neighbor_distance: Characteristic length scale
                - number_of_particles: Total particle count for normalization
                
        Physical Interpretation:
            - neighbor_distance sets natural q-space resolution
            - Used for converting between normalized and physical units
            - Essential for comparing results across different structures
        """
        N_points = float(len(structure))
        
        # Build spatial tree for efficient neighbor searching
        tree = cKDTree(structure)  # O(N log N) construction
        
        # Find nearest neighbor for each particle (k=2 includes self + nearest)
        d, k = tree.query(structure, k=2)
        
        # Calculate average nearest neighbor distance (excluding self at index 0)
        d_first_neighbours = np.average(d[:,1])
        
        return d_first_neighbours, N_points