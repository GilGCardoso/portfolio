    # -*- coding: utf-8 -*-
"""
Structure Factor Analysis for Optical Simulations

This module provides computation methods of structure factors for non-periodic particle arrangements, 
with targeted application at for non-periodic metasurfaces. The methods can be generalised for the interaction of
asemblies of materials (particles, atoms, etc) with light.

Key Features:
- Memory-adaptive algorithms (iterative, matrix, matrix-by-parts)
- Handles 20k-40k particles efficiently
- 2D visualization and radial averaging

Created on Fri May 15 12:06:43 2020

@author: Gil Cardoso
"""

import matplotlib
import numpy as np
from typing import Tuple, Optional, Sequence
from matplotlib.pyplot import figure
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import psutil
import socket
import logging
import os
import h5py
import json
from datetime import datetime


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- StructureFactorCalculator ---
class StructureFactorCalculator_cpu:

    # --- Initialization ---
    def __init__(self, memory_fraction: float = 2/3, dtype: np.dtype = np.float32) -> None:
        """
        Initialize the calculator.
        
        Args:
            memory_fraction: Fraction of available memory to use (default: 2/3)
        """
        self.memory_fraction = memory_fraction         # unchanged behavior
        self.dtype = dtype                  # new: controls numeric precision


    # --- Input & geometry helpers ---
    def get_calculation_parameters(self, structure: np.ndarray) -> Tuple[float, int]:
        """Compute basic parameters from the input structure.

        Args:
            structure (np.ndarray): Array of particle positions, shape (N, 2).

        Returns:
            Tuple[float, int]: (average first-neighbor distance D, number of particles N)
        """
        N_points = len(structure)
        tree = cKDTree(structure)  # create tree of closest neighbours
        d, k = tree.query(structure, k=2)
        d_first_neighbours = np.average(d[:, 1])

        logger.info(f"Structure analysis: {N_points} particles, "
                    f"avg neighbor distance: {d_first_neighbours:.6f}")

        return d_first_neighbours, N_points

    def generate_vectors(
        self,
        structure: np.ndarray,
        domain: float,
        vector_step: float,
        particle_distance: float,
        full_plane: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates the arrays of particle distance and the scattering vector necessary for the calculation of the structure factor.
        Args:
            structure: Particle positions array, shape (N, 2)
            domain: Maximum q-vector magnitude in units of particle_distance
            vector_step: Step size for q-vector grid
            particle_distance: Characteristic length scale for normalization
            full_plane: If True, q-vector covers symmetric range (full plane); if False, covers first quadrant only.
            
        Returns:
            tuple: (x_distances, y_distances, q_vector_grid)
                - x_distances: All pairwise x-distances, shape (1, N²)
                - y_distances: All pairwise y-distances, shape (1, N²)  
                - q_vector_grid: 1D array of q-values for calculation"""
        coord_x = structure[:,0]
        coord_y = structure[:,1]

        #create distance array for x an y   
        coord_x_ = coord_x.reshape(len(coord_x),1)
        coord_y_ = coord_y.reshape(len(coord_y),1)
        # after computing distances_x, distances_y
        distances_x = (coord_x - coord_x_).reshape(1, len(coord_x)**2).astype(self.dtype)
        distances_y = (coord_y - coord_y_).reshape(1, len(coord_y)**2).astype(self.dtype)

        # set q vector
        border = domain / particle_distance
        step = vector_step / particle_distance
        if full_plane:
            # symmetric range covering the full qx–qy plane
            scat_vector = np.arange(-border, border, step, dtype=self.dtype)
        else:
            # original behavior: first quadrant only
            scat_vector = np.arange(0, border, step, dtype=self.dtype)

        return distances_x, distances_y, scat_vector

    def select_calculation_method(self, vector: np.ndarray, q: np.ndarray) -> str:
        """Select the calculation method depeding on memory of the computer, the maximum memory is half of the memory of the pc"""
        max_size_array = memory_fraction*(psutil.virtual_memory().total/vector.itemsize)
        
        if max_size_array < np.size(vector,axis=1)*len(q):
            selector = "iterative"
        
        elif max_size_array > (np.size(vector,axis=1)*len(q)**2):
            selector = "matrix"
        else:
            selector = "matrix_by_parts"
            
        return selector

    # --- Matrix generation ---
    def generate_calculation_matrix(self, d_x: np.ndarray, d_y: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_x = d_x.reshape(np.ma.size(d_x, axis=1), 1)
        d_y = d_y.reshape(np.ma.size(d_y, axis=1), 1)

        mult_x = d_x * q
        mult_y = d_y*q
                        
        mult_y = mult_y.reshape(np.ma.size(mult_y,axis=0),np.ma.size(mult_y,axis=1),1)
        matrix_dq_y = np.swapaxes(mult_y,0,1)
                        
        mult_x = mult_x.reshape(np.ma.size(mult_x,axis=0),np.ma.size(mult_x,axis=1),1)
        mult_x = np.swapaxes(mult_x,0,1)
        matrix_dq_x = np.swapaxes(mult_x,0,2) 
        
        return matrix_dq_x, matrix_dq_y

    # --- Core calculation methods ---
    def iterative_method(self, d_x: np.ndarray, d_y: np.ndarray, q: np.ndarray, N: int) -> np.ndarray:
        structure_factor = np.zeros((len(q), len(q)), dtype=self.dtype)
        for a in range(len(q)):
            print(str(a/len(q)*100)+"%")
            for b in range(len(q)):
                structure_factor[a,b] = (1/N)*np.sum(np.exp(1j*(q[a]*d_x + q[b]*d_y)))
        return structure_factor

    def matrix_method(self, d_x: np.ndarray, d_y: np.ndarray, q: np.ndarray, N: int) -> np.ndarray:
        """
        Compute the structure factor using the full matrix approach.

        Parameters:
        - d_x (np.ndarray): Pairwise x-distance vector (shape (1, N^2)).
        - d_y (np.ndarray): Pairwise y-distance vector (shape (1, N^2)).
        - q (np.ndarray): 1D q-vector grid.
        - N (int): Number of particles.

        Returns:
        - np.ndarray: Structure factor array of shape (len(q), len(q)).
        """
        # Build matrices from distance vectors and q-grid
        m_x, m_y = self.generate_calculation_matrix(d_x, d_y, q)
        # Direct computation (identical to previous behavior)
        structure_factor = (1 / N) * np.sum(np.cos(m_x + m_y), axis=1)
        return structure_factor

    def matrix_by_parts_method(self, d_x: np.ndarray, d_y: np.ndarray, q: np.ndarray, N: int) -> np.ndarray:
        """
        Compute the structure factor in blocks to reduce memory usage.

        Parameters:
        - d_x (np.ndarray): Pairwise x-distance vector (shape (1, N^2)).
        - d_y (np.ndarray): Pairwise y-distance vector (shape (1, N^2)).
        - q (np.ndarray): 1D q-vector grid.
        - N (int): Number of particles.

        Returns:
        - np.ndarray: Structure factor array of shape (len(q), len(q)).
        """
        # Build matrices from distance vectors and q-grid
        m_x, m_y = self.generate_calculation_matrix(d_x, d_y, q)
        q_length = len(q)
        structure_factor = np.zeros((q_length, q_length), dtype=self.dtype)

        # Determine maximum rows per block based on available memory
        max_size_array = (2 / 3) * (psutil.virtual_memory().total / m_x.itemsize)
        rows_total = np.ma.size(m_y, axis=0)
        cols = np.ma.size(m_y, axis=1)
        max_dim = int(max_size_array / (cols * q_length))
        # Log chunk configuration
        logger.info(
            f"Matrix-by-parts configuration: d_length={cols}, q_length={q_length}, max_rows_per_chunk={max_dim}"
        )

        count = 1
        while rows_total > max_dim * count:
            start = max_dim * (count - 1)
            end = max_dim * count
            logger.info(
                f"Processing chunk {count}: rows={end - start} (indices {start}:{end}), cols={cols}"
            )
            structure_factor[start:end, :] = (1 / N) * np.sum(
                np.cos(m_x + m_y[start:end, :]), axis=1
            )
            progress = (1 - ((rows_total - end) / rows_total)) * 100
            print(f"{progress}%")
            count += 1

        # Final chunk
        start = max_dim * (count - 1)
        logger.info(
            f"Processing final chunk: rows={rows_total - start} (indices {start}:{rows_total}), cols={cols}"
        )
        structure_factor[start:, :] = (1 / N) * np.sum(
            np.cos(m_x + m_y[start:, :]), axis=1
        )

        return structure_factor

    # --- High-level orchestration ---
    def direct_Sq_calculation(self, structure: np.ndarray, domain: float, vector_step: float, full_plane: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Compute the 2D structure factor S(qx, qy) using a memory‑adaptive method.

        Pipeline (no behavior change):
        1) Derive the characteristic distance D and particle count from `structure`.
        2) Build pairwise distance vectors (d_x, d_y) and the 1D q-grid.
        3) Select a method based on available memory: 'iterative', 'matrix', or 'matrix_by_parts'.
           • For the matrix methods, the q–distance matrices are generated *inside* the method.

        Parameters
        ----------
        structure : array-like, shape (N, 2)
            Particle positions.
        domain : float
            Maximum extent used to generate the q-grid (passed to `generate_vectors`).
        vector_step : float
            Step size for the q-grid (passed to `generate_vectors`).
        full_plane : bool, optional
            If True, q-vector covers symmetric range (full qx–qy plane); if False, first quadrant only.

        Returns
        -------
        Sq_2D : np.ndarray, shape (len(q), len(q))
            2D structure factor.
        D : float
            Characteristic length scale used to non-dimensionalize q.
        q : np.ndarray, shape (len(q),)
            1D q-grid corresponding to the axes of `Sq_2D`.
        """
        D, Ptot = self.get_calculation_parameters(structure)
        d_x, d_y, q = self.generate_vectors(structure, domain, vector_step, D, full_plane=full_plane)
        method = self.select_calculation_method(d_x, q)

        logging.info(f"Selected structure factor calculation method: {method}")
        logging.info(f"q-range: {np.min(q) * 1e6:.2f} µm⁻¹ to {np.max(q) * 1e6:.2f} µm⁻¹")

        if method == 'iterative':
            Sq_2D = self.iterative_method(d_x, d_y, q, Ptot)

        elif method == 'matrix':
            Sq_2D = self.matrix_method(d_x, d_y, q, Ptot)

        elif method == 'matrix_by_parts':
            Sq_2D = self.matrix_by_parts_method(d_x, d_y, q, Ptot)

        else:
            raise ValueError(f"Unknown method selected: {method}")

        return Sq_2D, D, q
    
    

    # --- Post-processing ---
    def radial_average(self, Sq: np.ndarray, qD: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the radial average of a 2D structure factor array.

        Parameters:
        - Sq (np.ndarray): 2D array representing the structure factor.
        - qD (np.ndarray): 1D array of q-vector magnitudes corresponding to Sq.

        Returns:
        - structure_factor (np.ndarray): 1D array of radial averaged values.
        - r (np.ndarray): Radial bin midpoints corresponding to the averages.
        """

        N = len(Sq)

        # Create meshgrid of qD values to calculate radial distances
        x_q, y_q = np.meshgrid(qD, qD)
        R = np.sqrt(x_q**2 + y_q**2)

        # Define radial bin edges with half-step offset for proper binning
        step = qD[1] - qD[0]
        rad_bins = np.linspace(-step / 2, np.max(qD) + step / 2, num=N + 1)
        r = 0.5 * (rad_bins[:-1] + rad_bins[1:])  # Bin midpoints for x-axis

        # Initialize array to hold radial averages
        structure_factor = np.empty(N)

        # Compute mean of Sq values within each radial bin
        for i in range(N):
            mask = (R >= rad_bins[i]) & (R < rad_bins[i + 1])
            values = Sq[mask]

            if values.size > 0:
                structure_factor[i] = np.mean(values)
            else:
                structure_factor[i] = np.nan  # Assign NaN if no values in bin

        return structure_factor, r

    # --- I/O and pipeline orchestration ---
    def save_data(self, Sq_2D: np.ndarray, q: np.ndarray, D: float, Sq: np.ndarray, R: np.ndarray, save_folder: str, filename: str, *, save_hdf5: bool = True, save_json: bool = True, save_dat: bool = False) -> None:
        """
        Save results to HDF5 and JSON (metadata). Optionally also save legacy .dat files.

        Parameters:
        - Sq_2D: 2D array for Sq data
        - q: 1D array of q values
        - D: scaling factor
        - Sq: 1D array for Sq data
        - R: 1D array for R data
        - save_folder: folder path to save files
        - filename: base filename string
        - save_hdf5 (bool): Save HDF5 file with datasets (default True)
        - save_json (bool): Save JSON metadata file (default True)
        - save_dat (bool): Also save legacy .dat files (default False)
        """

        # Ensure save_folder ends with separator and exists
        if save_folder and not save_folder.endswith(os.sep):
            save_folder += os.sep
        if save_folder and not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)

        # Reshape q to a column vector
        q_col = q.reshape(-1, 1)

        # Prepare 2D data by stacking Sq_2D, q, and q scaled by D
        save_2D = np.hstack((Sq_2D, q_col, q_col * D))

        # Prepare 1D data by stacking Sq, R, and R scaled by D as rows
        save_1D = np.vstack((Sq, R, R * D))

        # Calculate qD min/max and step for filename metadata
        q_step = q[1] - q[0]
        q_range = q[-1] + q_step
        qD = q * D
        qD_min = float(np.min(qD))
        qD_max = float(np.max(qD))
        dqD = float(q_step * D)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')

        # Base file name (Option B without method, no Nq/D in name)
        base = f"sq2d-{filename}-qD{qD_min:.3f}-{qD_max:.3f}-dq{dqD:.3f}-t{ts}"

        # --- HDF5 save ---
        if save_hdf5:
            h5_path = f"{save_folder}{base}.h5"
            with h5py.File(h5_path, 'w') as h5:
                # Datasets
                dset_sq2d = h5.create_dataset('Sq_2D_with_q_columns', data=save_2D)
                dset_sq1d = h5.create_dataset('Sq_1D_rows', data=save_1D)
                h5.create_dataset('q', data=q)
                h5.create_dataset('R', data=R)
                h5.create_dataset('q_values_1D', data=R)
                h5.create_dataset('Sq', data=Sq)
                # Attributes
                h5.attrs['D'] = float(D)
                h5.attrs['q_step'] = float(q_step)
                h5.attrs['q_range'] = float(q_range)
                h5.attrs['created'] = datetime.utcnow().isoformat() + 'Z'
                h5.attrs['description'] = (
                    'Sq_2D_with_q_columns has last two columns = q and q·D; '
                    'Sq_1D_rows rows = [Sq, R, R·D]'
                )

        # --- JSON metadata save ---
        if save_json:
            meta = {
                'filename': filename,
                'created_utc': datetime.utcnow().isoformat() + 'Z',
                'D': float(D),
                'q_step': float(q_step),
                'q_range': float(q_range),
                'qD_min': qD_min,
                'qD_max': qD_max,
                'dqD': dqD,
                'filename_base': base,
                'shapes': {
                    'Sq_2D': list(Sq_2D.shape),
                    'Sq': int(np.size(Sq)),
                    'R': int(np.size(R)),
                    'q_values_1D': int(np.size(R)),
                    'q': int(np.size(q))
                }
            }
            json_path = f"{save_folder}{base}.json"
            with open(json_path, 'w') as jf:
                json.dump(meta, jf, indent=2)

        # --- Optional legacy .dat save ---
        if save_dat:
            file_2d = f"{save_folder}{base}.2d.dat"
            file_1d = f"{save_folder}{base}.1d.dat"
            np.savetxt(
                file_2d, save_2D, delimiter=',',
                header='The last two columns are q and q·D respectively.'
            )
            np.savetxt(
                file_1d, save_1D, delimiter=',',
                header='Rows: Sq, q, q·D.'
            )
        
    def calculate_structure_factor_and_save(
        self, 
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
        Load structural data from CSV, perform Sq calculations, and save outputs.

        Parameters:
        - read_folder (str): Path to the folder containing the input CSV file.
        - filename (str): Name of the CSV file (without extension).
        - range_calculation: Range parameter for direct_Sq_calculation().
        - vector_step: Step size for direct_Sq_calculation().
        - save_folder (str, optional): Folder to save the output files. Defaults to ''.
        - full_plane (bool, optional): If True, compute q from -border to +border; else 0 to border. Defaults to False.
        - save_hdf5 (bool, optional): Save HDF5 results (default True).
        - save_json (bool, optional): Save JSON metadata (default True).
        - save_dat (bool, optional): Also save legacy .dat files (default False).
        """
        
        print(f"Processing file: {filename}")
        
        # Load structural data from CSV
        filepath = os.path.join(read_folder, f"{filename}.csv")
        structure = np.genfromtxt(filepath, delimiter=',').astype(self.dtype)

        # Perform direct 2D Sq calculation
        Sq_2D, D, q = self.direct_Sq_calculation(
            structure, range_calculation, vector_step, full_plane=full_plane
        )
        
        # Calculate radial average (1D Sq and R arrays)
        Sq, R = self.radial_average(Sq_2D, q)
        
        # Save both 2D and 1D results
        self.save_data(
            Sq_2D, q, D, Sq, R, save_folder, filename,
            save_hdf5=save_hdf5,
            save_json=save_json,
            save_dat=save_dat
        )

# --- StructureFactorVisualizer ---
class StructureFactorVisualizer:
    # --- Helper: vector selection ---
    def vector_selection(self, array: np.ndarray, axis_selection: str) -> np.ndarray:
        if axis_selection == 'q':
            axis = array[-2]*1e6
        elif axis_selection == 'qD':
            axis = array[-1]
        elif axis_selection == 'xf':    
            axis = array[1]*1e6
            axis /= 1.033e4     
        return axis

    # --- Plotting: 2D structure factor ---
    def plot_Sq_2D(self, data_folder: str, filename: str, edge: Sequence[float], x_axis: str = 'qD', save_plot: bool = False, folder_write: str = '', log_scale: bool = False, *, 
                   auto_contrast: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = 'jet', show: bool = True) -> Tuple[Figure, Axes]:
        # Load from HDF5 produced by save_data
        with h5py.File(os.path.join(data_folder, filename + '.h5'), 'r') as f:
            Sq2D_and_arrays = f['Sq_2D_with_q_columns'][:]

        # Separate grid and axis vector (last two columns are q and qD)
        Sq_2D = Sq2D_and_arrays[:, :-2]
        if x_axis == 'q':
            vector = Sq2D_and_arrays[:, -2] * 1e6
        elif x_axis == 'qD':
            vector = Sq2D_and_arrays[:, -1]
        elif x_axis == 'xf':
            vector = Sq2D_and_arrays[:, -1] / 1.033e2
        else:
            # Fallback to qD if unknown axis
            vector = Sq2D_and_arrays[:, -1]

        # Crop to requested window (symmetric around 0 with edge[1])
        mask_1d = (vector > -edge[1]) & (vector < edge[1])
        vector = vector[mask_1d]
        mask_2d = np.logical_and.outer(mask_1d, mask_1d)
        Sq_2D = Sq_2D[mask_2d].reshape((len(vector), len(vector)))

        # Font sizes
        plt.rcParams.update({'font.size': 20})

        # Auto-contrast percentiles if requested
        if auto_contrast and vmin is None and vmax is None:
            vmin = float(np.nanpercentile(Sq_2D, 1))
            vmax = float(np.nanpercentile(Sq_2D, 99))

        # Plot
        fig = figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
        ax = plt.axes(xlim=edge, ylim=edge, autoscale_on=True)

        # Always use imshow for plotting
        extent = [vector.min(), vector.max(), vector.min(), vector.max()]
        if log_scale:
            from matplotlib.colors import LogNorm
            norm = LogNorm()
            img = plt.imshow(
                Sq_2D, extent=extent, origin='lower', aspect='equal', cmap=cmap,
                norm=norm
            )
        else:
            img = plt.imshow(
                Sq_2D, extent=extent, origin='lower', aspect='equal', cmap=cmap,
                vmin=vmin, vmax=vmax
            )

        # Labels (q and qD use identical labels in current code)
        if x_axis in ('q', 'qD'):
            plt.xlabel('qx (m\u207B\u00B9)')
            plt.ylabel('qy (m\u207B\u00B9)')
        elif x_axis == 'xf':
            plt.xlabel('x(micron)')
            plt.ylabel('y(micron)')

        # Style and colorbar
        plt.rc('axes', titlesize=20)
        plt.rc('axes', labelsize=30)
        plt.rc('xtick', labelsize=30)
        plt.rc('ytick', labelsize=30)
        ax.set_aspect('equal')
        cbar = plt.colorbar(img, shrink=0.65)
        cbar.set_label('S(q)')
        cbar.ax.tick_params(labelsize=20)
        plt.tight_layout()

        # Optional save (filenames unchanged)
        if save_plot:
            if log_scale:
                plt.savefig(folder_write + '/2D_Sq_' + filename + "_edge_" + str(edge) + '_log_scale.png')
            else:
                plt.savefig(folder_write + '/2D_Sq_' + filename + "_edges_" + str(edge) + '.svg')

        if show:
            plt.show()
        return fig, ax

    # --- Plotting: 1D structure factor ---
    def plot_Sq_1D(self, data_folder: str, filename: Sequence[str], labels: Sequence[str], edges: Sequence[float], x_axis: str = 'qD', save_plot: bool = False, folder_write: str = '', log_scale: bool = False, mult_plot: bool = False, moving_average: bool = False, n_ave: int = 3, y_max: float = 100, *, auto_ylim: bool = False, show: bool = True) -> Tuple[Figure, Axes]:
        # Figure and axis
        fig = figure(num=None, figsize=(18, 10), dpi=100, facecolor='w', edgecolor='k')
        ax = plt.gca()

        # Consistent styling once (not in loop)
        plt.rc('legend', fontsize=15)
        plt.rc('axes', titlesize=30)
        plt.rc('axes', labelsize=30)
        plt.rc('xtick', labelsize=30)
        plt.rc('ytick', labelsize=30)

        markers_array = np.asarray(["v", "o", "^", "s", "<", "x", ">", "v", "o", "^", "s", "<", "x", ">"])

        for count, base in enumerate(filename):
            # Load from HDF5 produced by save_data
            with h5py.File(os.path.join(data_folder, str(base) + '.h5'), 'r') as f:
                Sq = f['Sq'][:]
                if 'q_values_1D' in f:
                    q_values_1D = f['q_values_1D'][:]
                else:
                    q_values_1D = f['R'][:]
                D = float(f.attrs['D']) if 'D' in f.attrs else 1.0

            # Build minimal array so vector_selection works unchanged: rows [dummy, q_values_1D, q_values_1D*D]
            Sq_and_arrays = np.vstack((np.zeros_like(q_values_1D), q_values_1D, q_values_1D * D))

            # Optional smoothing
            if moving_average and n_ave > 1:
                kernel = np.ones(n_ave) / n_ave
                Sq = np.convolve(Sq, kernel, mode='same')

            # Axis selection and cropping
            vector = self.vector_selection(Sq_and_arrays, x_axis)
            mask = (edges[0] < vector) & (vector < edges[1])
            vector = vector[mask]
            Sq = Sq[mask]

            # Plot
            ax.plot(vector, Sq, marker=markers_array[count], ms=6, lw=2, label=labels[count])

        # Scales and labels
        if log_scale:
            ax.set_yscale('log')

        if x_axis == 'q':
            ax.set_xlabel('q (m\u207B\u00B9)')
        elif x_axis == 'qD':
            ax.set_xlabel('q (m\u207B\u00B9)')
        elif x_axis == 'xf':
            ax.set_xlabel('x(micron)')

        ax.set_ylabel('S(q)')

        # Reference lines
        ax.axvline(x=2.19e6, color='k', lw=2, linestyle='dashed')
        ax.axvline(x=2.22e7, color='k', lw=2, linestyle='dashed')
        ax.axvline(x=1.57e7, color='k', lw=2, linestyle='dashed')
        ax.axhline(y=0.05, color='g', lw=2)
        ax.axhline(y=0.1, color='b', lw=2)

        # y-limits and optional auto y scaling
        ax.set_ylim(0, y_max)
        if auto_ylim and not log_scale:
            lines = ax.get_lines()
            if lines:
                ydata = np.concatenate([ln.get_ydata() for ln in lines])
                ydata = ydata[np.isfinite(ydata)]
                if ydata.size > 0:
                    ax.set_ylim(0, float(np.nanmax(ydata) * 1.05))

        # x-limits
        ax.set_xlim(edges)

        ax.legend()
        plt.tight_layout()

        # Optional save
        if save_plot:
            plt.tight_layout()
            plt.savefig(folder_write + 'Sq_' + 'multiple_filtered_areas' + "_edges_" + str(edges[0]) + '_' + str(edges[1]) + '.svg')

        if show:
            plt.show()
        return fig, ax