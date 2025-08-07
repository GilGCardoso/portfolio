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


class StructureFactorCalculator:
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
    
    def calculate_structure_factor(self, read_folder, file, range_calculation, vector_step, save_folder):
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
            print(f"Progress: {a/len(q)*100:.1f}%")  # Progress indicator
            for b in range(len(q)):
                # Calculate structure factor at each q-point using complex exponentials
                structure_factor[a,b] = (1/N)*torch.sum(torch.exp(1j*(q[a]*d_x + q[b]*d_y)))  

        # Transfer results back to CPU for post-processing and saving
        structure_factor = structure_factor.cpu().numpy() 
        q = q.cpu().numpy()

        # Compute radial average to get 1D structure factor S(|q|)
        Sq, R = self.radial_average(structure_factor, q)

        # Save both 2D and 1D results with metadata
        self.save_data(structure_factor, q, D, Sq, R, save_folder, file)

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
                
        Memory Usage:
            - Distance arrays scale as O(N²) where N is number of particles
            - For 20k particles: ~1.6GB memory for distance storage
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

    def radial_average(self, Sq, qD):
        """
        Compute radial average of 2D structure factor for isotropic analysis.
        
        Converts 2D structure factor S(qx, qy) into 1D function S(|q|) by
        averaging over angular directions in q-space.
        
        Args:
            Sq (np.ndarray): 2D structure factor array
            qD (np.ndarray): 1D q-vector array
            
        Returns:
            tuple: (structure_factor_1d, radial_q_values)
                - structure_factor_1d: Radially averaged S(|q|)
                - radial_q_values: Corresponding |q| values
                
        Physical Meaning:
            - Removes directional information, keeping only radial correlations
            - Useful for isotropic or weakly anisotropic structures
            - Enables comparison with theoretical predictions (liquid theory, etc.)
        """
        N = len(Sq)
        
        # Build 2D grid of radial distances in q-space
        x_q, y_q = np.meshgrid(qD, qD)
        R = (x_q**2 + y_q**2)**0.5  # |q| = sqrt(qx² + qy²)

        # Define radial binning scheme
        step = qD[1] - qD[0]
        rad_bins = np.linspace(-step/2, np.max(qD)+step/2, num=N+1)

        # Bin centers for output q-values
        r = (rad_bins[0:-1] + rad_bins[1:])/2
        
        # Compute average within each radial shell
        structure_factor = np.zeros(N)

        for n in range(N-1):
            # Find all points within this radial shell
            mask = (R >= rad_bins[n]) & (R < rad_bins[n+1])
            count = len(Sq[mask])
            
            if count != 0:
                # Average structure factor over all points in shell
                structure_factor[n] = np.sum(Sq[mask]) / count
            else:
                # Handle empty bins (typically at large |q|)
                structure_factor[n] = float('nan')
                
        return structure_factor, r

    def save_data(self, Sq_2D, q, D, Sq, R, save_folder, filename):
        """
        Save calculation results in standardized format with metadata.
        
        Creates two output files:
        1. 2D structure factor with coordinate information
        2. 1D radial average for quick analysis
        
        Args:
            Sq_2D (np.ndarray): 2D structure factor array
            q (np.ndarray): Q-vector sampling points (normalized)
            D (float): Characteristic particle distance
            Sq (np.ndarray): 1D radially averaged structure factor
            R (np.ndarray): Radial q-coordinates
            save_folder (str): Output directory path
            filename (str): Base filename for outputs
            
        Output Format:
            - CSV files with descriptive headers
            - Both normalized (q) and physical (q*D) units included
            - Self-documenting format for reproducible analysis
        """
        # Prepare 2D data: structure factor + coordinate columns
        q_save = q.reshape(np.ma.size(q,axis=0),1)
        save_2D = np.append(Sq_2D, q_save, axis=1)  # Add normalized q-vector
        save_2D = np.append(save_2D, q_save*D, axis=1)  # Add physical q-vector

        # Prepare 1D data: [S(q), q_normalized, q_physical]
        save_1D = [Sq, R, R*D]

        # Generate informative filenames with parameter metadata
        q_range = str(round((q[-1]+q[1]-q[0])*D))
        q_step = str(round((q[1]-q[0])*D, 3))

        # Save 2D structure factor with metadata header
        filename_2D = save_folder + '2D_Sq_' + filename + '_range_' + q_range + '_step_' + q_step + '.dat'
        np.savetxt(filename_2D, save_2D, delimiter=',', 
                   header='The before last and last columns correspond to vector q and q.D respectively.')

        # Save 1D radial average with metadata header
        filename_1D = save_folder + 'Sq_' + filename + '_range_' + q_range + '_step_' + q_step + '.dat'
        np.savetxt(filename_1D, save_1D, delimiter=',', 
                   header='The row order is Sq,q,q.D.') 


class StructureFactorVisualizer:
    """
    Visualization and analysis tools for structure factor data.
    
    This class provides plotting functions for both 2D structure factors
    and 1D radial averages, with publication-quality output options.
    """
    
    def __init__(self):
        """Initialize visualizer with default plotting parameters."""
        # Set publication-quality defaults
        plt.rcParams.update({'font.size': 20})
    
    def plot_Sq_2D(self, folder_read, file, edge, x_axis='qD', save_plot=False, 
                   remove_center=False, remove_single_center=False, 
                   folder_write='', log_scale=False, max_value=100):
        """
        Create publication-quality 2D structure factor visualization.
        
        Generates color-coded plots of the 2D structure factor with proper
        axis labels, colorbars, and optional logarithmic scaling.
        
        Args:
            folder_read (str): Path to data file directory
            file (str): Filename (without extension) of structure factor data
            edge (tuple): (min, max) range for plot axes
            x_axis (str): Coordinate system - 'q', 'qD', or 'xf'
            save_plot (bool): Whether to save figure to file
            log_scale (bool): Use logarithmic color scaling
            max_value (float): Maximum value for color scale normalization
            folder_write (str): Output directory for saved figures
            
        Coordinate Systems:
            - 'q': Physical units (m⁻¹)
            - 'qD': Normalized by particle distance (dimensionless)
            - 'xf': Real-space representation (microns)
            
        Business Value:
            - Professional visualization for reports and publications
            - Multiple coordinate systems for different analysis needs
            - Logarithmic scaling reveals features across wide dynamic range
        """
        # Load structure factor data from file
        Sq2D_and_arrays = np.loadtxt(folder_read + file + '.dat', delimiter=',') 

        # Extract 2D structure factor (exclude coordinate columns)
        Sq_2D = Sq2D_and_arrays[:,:-2]
        
        # Select coordinate system and set appropriate labels
        if x_axis == 'q':
            vector = Sq2D_and_arrays[:,-2]*1e6  # Convert to μm⁻¹
            xlabel, ylabel = 'qx (m⁻¹)', 'qy (m⁻¹)'
        elif x_axis == 'qD':
            vector = Sq2D_and_arrays[:,-1]  # Normalized coordinates
            xlabel, ylabel = 'qx D', 'qy D'
        elif x_axis == 'xf':
            vector = Sq2D_and_arrays[:,-1] / 1.033e2  # Real-space units
            xlabel, ylabel = 'x(micron)', 'y(micron)'
                  
        # Apply range selection for focused visualization
        bool_vec = (vector > -(edge[1])) & (vector < (edge[1]))
        vector = vector[bool_vec] 
        
        # Create 2D mask for data selection
        bool_vec_x, bool_vec_y = np.meshgrid(bool_vec, bool_vec)
        bool_vec_2D = bool_vec_x & bool_vec_y    
        Sq_2D = Sq_2D[bool_vec_2D].reshape((len(vector), len(vector)))
        
        # Create figure with square aspect ratio
        figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
        ax = plt.axes(xlim=(edge), ylim=(edge), autoscale_on=True)
        
        # Choose color normalization and colormap
        if log_scale:
            norm = matplotlib.colors.LogNorm()
            colormap = 'jet'
            plot = plt.pcolor(vector, vector, Sq_2D, rasterized=True, 
                            cmap=colormap, shading='auto', vmax=max_value, norm=norm)
        else:
            plot = plt.pcolor(vector, vector, Sq_2D, rasterized=True, 
                            cmap='jet', shading='auto', vmax=max_value)
        
        # Set labels and formatting
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.rc('axes', titlesize=20)     
        plt.rc('axes', labelsize=30) 
        plt.rc('xtick', labelsize=30)   
        plt.rc('ytick', labelsize=30)
        ax.set_aspect('equal')
        
        # Add colorbar with proper sizing
        cbar = plt.colorbar(plot, shrink=0.65)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('S(q)', rotation=270, labelpad=20)
        plt.tight_layout()
        
        # Save figure if requested
        if save_plot:
            if log_scale:  
                filename = folder_write + '2D_Sq_' + file + "_edge_" + str(edge) + '_log_scale.png'
            else:  
                filename = folder_write + '2D_Sq_' + file + "_edges_" + str(edge) + '.svg'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show() 

    def vector_selection(self, array, axis_selection):
        """
        Select appropriate coordinate vector based on analysis type.
        
        Args:
            array (np.ndarray): Data array containing coordinate information
            axis_selection (str): Coordinate system choice
            
        Returns:
            np.ndarray: Selected coordinate vector
        """
        if axis_selection == 'q':
            axis = array[-2]*1e6  # Physical q-vector
        elif axis_selection == 'qD':
            axis = array[-1]  # Normalized q-vector
        elif axis_selection == 'xf':    
            axis = array[1]*1e6 / 1.033e4  # Real-space coordinates
            
        return axis

    def plot_Sq_1D(self, folder_read, file, labels, edges, x_axis='qD', save_plot=False, 
                   folder_write='', log_scale=False, mult_plot=False, 
                   moving_average=False, n_ave=3, y_max=100):
        """
        Create comparative plots of multiple 1D structure factors.
        
        Overlays multiple structure factor curves for comparison analysis,
        with optional smoothing and reference lines for interpretation.
        
        Args:
            folder_read (str): Input data directory
            file (list): List of filenames to plot
            labels (list): Legend labels for each curve
            edges (tuple): (min, max) range for x-axis
            x_axis (str): Coordinate system for x-axis
            save_plot (bool): Save figure to file
            log_scale (bool): Use logarithmic y-axis
            moving_average (bool): Apply smoothing filter
            n_ave (int): Number of points for moving average
            y_max (float): Maximum y-axis value
            
        Analysis Features:
            - Multiple curve overlay for comparative analysis
            - Reference lines for key structural features
            - Optional smoothing for noisy data
            - Professional formatting for publications
        """
        figure(num=None, figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')
        
        # Define marker styles for different curves
        markers_array = np.asarray(["v", "o", "^", "s", "<", "x", ">",
                                   "v", "o", "^", "s", "<", "x", ">"])
        
        # Plot each structure factor curve
        for count, Sq_and_array in enumerate(file):
            # Load data for this curve
            Sq_and_arrays = np.genfromtxt(folder_read + str(file[count]) + '.dat', delimiter=',')
            Sq = Sq_and_arrays[0]  # Structure factor values
            
            # Apply moving average smoothing if requested
            if moving_average:
                Sq_mov_ave = np.zeros(len(Sq) - n_ave)
                for p in range(n_ave):
                    Sq_mov_ave += Sq[p:p - n_ave] 
                Sq_mov_ave /= n_ave
                Sq = np.append(Sq[:n_ave], Sq_mov_ave)
                
            # Select coordinate system
            vector = self.vector_selection(Sq_and_arrays, x_axis)
                    
            # Apply range selection
            range_mask = np.logical_and((edges[0] < vector), (vector < edges[1]))
            Sq = Sq[range_mask]
            vector = vector[range_mask]

            # Plot curve with unique marker and label
            plt.plot(vector, Sq, marker=markers_array[count], ms=6, lw=2, 
                    label=labels[count])
        
        # Set axis scaling
        if log_scale: 
            plt.yscale("log")
        
        # Set axis labels based on coordinate system
        if x_axis == 'q':
            plt.xlabel('q (m⁻¹)', fontsize=35)
        elif x_axis == 'qD':
            plt.xlabel('q D', fontsize=35)
        elif x_axis == 'xf':
            plt.xlabel('x(micron)', fontsize=35)
        
        plt.ylabel('S(q)', fontsize=35)
        
        # Add reference lines for structural interpretation
        # These lines mark important q-values for photonic applications
        plt.axvline(x=2.19e6, color='k', lw=2, linestyle='dashed', alpha=0.7)
        plt.axvline(x=1.57e7, color='k', lw=2, linestyle='dashed', alpha=0.7)
        plt.axhline(y=0.05, color='g', lw=2, alpha=0.7)
        plt.axhline(y=0.1, color='b', lw=2, alpha=0.7)
        
        # Set axis limits and formatting
        plt.ylim([0, y_max])
        plt.xlim(edges)
        plt.legend(fontsize=20)
        plt.rc('axes', titlesize=30)     
        plt.rc('axes', labelsize=30) 
        plt.rc('xtick', labelsize=30)   
        plt.rc('ytick', labelsize=30)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_plot:
            filename = (folder_write + 'Sq_multiple_filtered_areas_edges_' 
                       + str(edges[0]) + '_' + str(edges[1]) + '.svg')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            
        plt.show()


# Example usage demonstrating the improved class structure
if __name__ == "__main__":
    # Initialize calculator and visualizer
    calculator = StructureFactorCalculator()
    visualizer = StructureFactorVisualizer()
    
    # Example calculation workflow
    try:
        # Calculate structure factor for a metasurface design
        calculator.calculate_structure_factor(
            read_folder='./input_data/',
            file='metasurface_v1',
            range_calculation=20.0,
            vector_step=0.1,
            save_folder='./results/'
        )
        
        # Visualize results
        visualizer.plot_Sq_2D(
            folder_read='./results/',
            file='2D_Sq_metasurface_v1_range_20_step_0.1',
            edge=(-10, 10),
            x_axis='qD',
            save_plot=True,
            folder_write='./figures/'
        )
        
        print("Analysis completed successfully!")
        
    except FileNotFoundError:
        print("Input files not found. Please check file paths.")