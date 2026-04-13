    # -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:06:43 2020

@author: Gil
"""

import matplotlib
import torch
import torch.cuda
import numpy as np
import psutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial import cKDTree
from pathlib import Path
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)

_BYTES_PER_F32 = 4
_CPU_RESERVE = 512 * 1024 * 1024
# Headroom for allocator fragmentation and small unaccounted tensors.
# GPU: cuda.mem_get_info is accurate, so we can trust most of it.
# CPU: psutil.available is an optimistic OS-level number (includes caches,
# ignores Windows commit limits and fragmentation) — halve it to stay safe.
_SAFETY_GPU = 0.85
_SAFETY_CPU = 0.5


def _free_bytes(device):
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free * _SAFETY_GPU)
    avail = psutil.virtual_memory().available - _CPU_RESERVE
    return max(0, int(avail * _SAFETY_CPU))


def _bytes_full_matrix(N, nq):
    # Peak tensors live simultaneously during `cos(a + b)`:
    #   pre_qb   = (1, nq, N²)  from q_b * d_y
    #   pre_qa   = (nq, 1, N²)  from q_a * d_x
    #   add_out  = (nq, nq, N²) sum of the two broadcasts  ← slab
    #   cos_out  = (nq, nq, N²) cos of add_out             ← slab
    # Reduction output (nq, nq) is negligible.
    N2 = N * N
    slab = nq * nq * N2 * _BYTES_PER_F32
    pre = 2 * nq * N2 * _BYTES_PER_F32
    return 2 * slab + pre


def _bytes_chunked(N, nq, chunk):
    # Same shape accounting as _bytes_full_matrix but qx is sliced to `chunk` rows:
    #   pre_qb  = (1, nq, N²)      — full q along second axis, NOT shrunk by chunk
    #   pre_qa  = (chunk, 1, N²)
    #   add_out = (chunk, nq, N²)  ← chunk_slab
    #   cos_out = (chunk, nq, N²)  ← chunk_slab
    # The (1, nq, N²) term dominates at chunk=1 — omitting it under-estimates by ~50%.
    N2 = N * N
    chunk_slab = chunk * nq * N2 * _BYTES_PER_F32
    pre_qb = nq * N2 * _BYTES_PER_F32
    pre_qa = chunk * N2 * _BYTES_PER_F32
    return 2 * chunk_slab + pre_qb + pre_qa


def _select_method(N, nq, device):
    budget = _free_bytes(device)
    if _bytes_full_matrix(N, nq) <= budget:
        return "matrix"
    if _bytes_chunked(N, nq, chunk=1) <= budget:
        return "matrix_by_parts"
    return "iterative"


def _matrix(d_x, d_y, q, N):
    q_a = q.reshape(-1, 1, 1)
    q_b = q.reshape(1, -1, 1)
    with tqdm(total=1, desc="matrix", unit="op") as pbar:
        out = (1.0 / N) * torch.sum(
            torch.cos(q_a * d_x.reshape(1, 1, -1) + q_b * d_y.reshape(1, 1, -1)),
            dim=2,
        )
        pbar.update(1)
    return out


_OOM_ERRORS = (RuntimeError,)
if hasattr(torch, "cuda") and hasattr(torch.cuda, "OutOfMemoryError"):
    _OOM_ERRORS = (RuntimeError, torch.cuda.OutOfMemoryError)


def _is_oom(exc):
    msg = str(exc).lower()
    return "out of memory" in msg or "not enough memory" in msg


def _matrix_by_parts(d_x, d_y, q, N, device):
    nq = q.numel()
    n_particles = int(round(d_x.numel() ** 0.5))
    out = torch.empty((nq, nq), dtype=torch.float32, device=device)
    budget = _free_bytes(device)
    chunk = 1
    while chunk + 1 <= nq and _bytes_chunked(n_particles, nq, chunk + 1) <= budget:
        chunk += 1

    start = 0
    pbar = tqdm(total=nq, desc="matrix_by_parts", unit="qx")
    while start < nq:
        end = min(nq, start + chunk)
        try:
            q_a = q[start:end].reshape(-1, 1, 1)
            q_b = q.reshape(1, -1, 1)
            out[start:end] = (1.0 / N) * torch.sum(
                torch.cos(q_a * d_x.reshape(1, 1, -1) + q_b * d_y.reshape(1, 1, -1)),
                dim=2,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            pbar.update(end - start)
            start = end
        except _OOM_ERRORS as exc:
            if not _is_oom(exc):
                pbar.close()
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if chunk == 1:
                logger.warning("OOM at chunk=1; falling back to iterative for remaining rows")
                out[start:] = _iterative_rows(d_x, d_y, q[start:], q, N, device)
                pbar.update(nq - start)
                break
            chunk = max(1, chunk // 2)
            logger.warning("OOM in matrix_by_parts; halving chunk to %d", chunk)
    pbar.close()
    return out


def _iterative_rows(d_x, d_y, q_rows, q_all, N, device):
    """Iterative computation restricted to a contiguous slice of qx rows."""
    nr = q_rows.numel()
    nq = q_all.numel()
    block = torch.empty((nr, nq), dtype=torch.float32, device=device)
    for a in tqdm(range(nr), desc="iterative (fallback)", unit="qx"):
        for b in range(nq):
            block[a, b] = (1.0 / N) * torch.sum(torch.cos(q_rows[a] * d_x + q_all[b] * d_y))
    return block


def _iterative(d_x, d_y, q, N):
    nq = q.numel()
    out = torch.empty((nq, nq), dtype=torch.float32, device=d_x.device)
    for a in tqdm(range(nq), desc="iterative", unit="qx"):
        for b in range(nq):
            out[a, b] = (1.0 / N) * torch.sum(torch.cos(q[a] * d_x + q[b] * d_y))
    return out


def calculate_structure_factor(read_folder, file, range_calculation, vector_step, save_folder, device="cpu"):

    if device == "cpu":
        torch_device = torch.device("cpu")
    elif device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available on this machine. Re-run with "
                "device='cpu' (the default) to use the CPU implementation."
            )
        torch_device = torch.device("cuda")
    else:
        raise ValueError(f"device must be 'cpu' or 'gpu', got {device!r}")

    structure_to_calculate = np.genfromtxt(str(Path(read_folder) / f'{file}.csv'), delimiter=',')
    structure_to_calculate_tensor = torch.from_numpy(structure_to_calculate).to(torch.float32)

    D, N = get_calculation_parameters(structure_to_calculate)
    d_x, d_y, q = generate_vectors(structure_to_calculate_tensor, range_calculation, vector_step, D)

    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    d_x = d_x.to(torch_device).to(torch.float32)
    d_y = d_y.to(torch_device).to(torch.float32)
    q = q.to(torch_device).to(torch.float32)
    N_tensor = torch.tensor(N, dtype=torch.float32, device=torch_device)

    n_particles = int(round(d_x.numel() ** 0.5))
    method = _select_method(n_particles, q.numel(), torch_device)
    print(f"Structure factor method: {method} "
          f"(N={n_particles}, nq={q.numel()}, device={torch_device})")

    if method == "matrix":
        structure_factor = _matrix(d_x, d_y, q, N_tensor)
    elif method == "matrix_by_parts":
        structure_factor = _matrix_by_parts(d_x, d_y, q, N_tensor, torch_device)
    else:
        structure_factor = _iterative(d_x, d_y, q, N_tensor)

    logger.info("Structure factor calculation complete")

    structure_factor = structure_factor.cpu().numpy()
    q = q.cpu().numpy()

    Sq, R = radial_average(structure_factor, q)

    save_data(structure_factor, q, D, Sq, R, save_folder, file)


def generate_vectors(structure, domain, vector_step, particule_distance):
    """Generates the vector used for the calculation of the structure factor.
    distances_x, distances_y : vectors that contains the distances between all points along the x anf y axis respectivly
    scat_vector : the scatterring vector used for the calculation"""

    coord_x = structure[:,0]
    coord_y = structure[:,1]

    #create distance array for x an y
    coord_x_ = coord_x.reshape(len(coord_x),1)
    distances_x = (coord_x - coord_x_).reshape(1,len(coord_x)**2)

    coord_y_ = coord_y.reshape(len(coord_y),1)
    distances_y= (coord_y - coord_y_).reshape(1,len(coord_y)**2)

    #set q vector
    border = domain/particule_distance;
    step = vector_step/particule_distance;
    scat_vector = torch.arange(0,border,step)

    return distances_x, distances_y, scat_vector

def get_calculation_parameters(structure):
    """return the parameters that are necessary for the calculation
    N_points : The amount of points in the structure
    d_first_neighbours : The average distance between first neighbours"""

    N_points = float(len(structure))
    tree = cKDTree(structure)  # create tree of closest neighbours
    d, k = tree.query(structure,k=2)
    d_first_neighbours = np.average(d[:,1])

    return d_first_neighbours, N_points

def radial_average(Sq,qD):
    """
    Compute radial average of 2D structure factor using vectorized binning.
    O(N log N) complexity instead of O(N²) with masking.
    """
    N = len(Sq)
    #build matrix of radial distances
    x_q,y_q = np.meshgrid(qD,qD)
    R  = (x_q**2 + y_q**2)**0.5

    #array for radial bins
    step = qD[1]-qD[0]
    rad_bins = np.linspace(-step/2,np.max(qD)+step/2,num=N+1)

    #mid points for each bins to be used as x axis
    r = (rad_bins[0:-1]+rad_bins[1:])/2

    #calculate radial average using vectorized binning
    bin_indices = np.digitize(R.flatten(), rad_bins) - 1

    structure_factor = np.zeros(N)
    for n in range(N-1):
        mask = bin_indices == n
        if np.any(mask):
            structure_factor[n] = np.mean(Sq.flatten()[mask])
        else:
            structure_factor[n] = float('nan')

    return structure_factor, r

def save_data(Sq_2D, q, D, Sq, R, save_folder, filename):
    # Ensure we save real-valued arrays (take absolute value for complex data)
    q_save = q.reshape(np.ma.size(q, axis=0), 1)

    Sq_2D_real = np.abs(Sq_2D)
    # Vectorize: use column_stack instead of sequential append calls
    save_2D = np.column_stack((Sq_2D_real, q_save, q_save * D))

    # Prepare 1D save array (rows: Sq, R, R*D) as real values
    save_1D = np.vstack((np.real(Sq), np.real(R), np.real(R * D)))

    # Build filenames with pathlib for cross-platform compatibility
    range_str = str(round((q[-1] + q[1] - q[0]) * D))
    step_str = str(round((q[1] - q[0]) * D, 3))

    save_folder_path = Path(save_folder)

    file_2d = save_folder_path / f'2D_Sq_{filename}_range_{range_str}_step_{step_str}.dat'
    file_1d = save_folder_path / f'Sq_{filename}_range_{range_str}_step_{step_str}.dat'

    np.savetxt(str(file_2d), save_2D, delimiter=',',
               header='The before last and last columns correspond to vector q and q.D respectively.')

    np.savetxt(str(file_1d), save_1D, delimiter=',',
               header='The row order is Sq,q,q.D.')


def plot_Sq_2D (folder_read, file, edge, x_axis = 'qD', save_plot = False,
                folder_write ='', log_scale = False, max_value = 100):

    Sq2D_and_arrays = np.loadtxt(folder_read + file + '.dat', delimiter=',' )

    Sq_2D = Sq2D_and_arrays[:,:-2]

    if x_axis == 'q':
        vector = Sq2D_and_arrays[:,-2]*1e6

    elif x_axis == 'qD':
        vector =   Sq2D_and_arrays[:,-1]

    elif x_axis == 'xf':
        vector =   Sq2D_and_arrays[:,-1]
        vector /= 1.033e2

    #Sq_2D = remove_center_2D(Sq_2D,vector,edge[0])

    bool_vec = (vector > -(edge[1])) & (vector < (edge[1]))
    vector = vector[bool_vec]
    bool_vec_x,bool_vec_y = np.meshgrid(bool_vec,bool_vec)
    bool_vec_2D  = bool_vec_x & bool_vec_y
    Sq_2D = Sq_2D[bool_vec_2D].reshape((len(vector),len(vector)))


    plt.rcParams.update({'font.size': 20})

    #plot 2D structure factor and save figure
    figure(num=None, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    ax = plt.axes(xlim=(edge), ylim=(edge), autoscale_on=True)


    if log_scale:
        plot = plt.pcolor( vector , vector, Sq_2D, rasterized=True, cmap='jet', shading='auto', vmax= max_value, norm=matplotlib.colors.LogNorm())

    else :
        plot = plt.pcolor( vector , vector, Sq_2D, rasterized=True, cmap='jet', shading='auto', vmax= max_value)
    #plt.title('2D structure factor')

    if x_axis == 'q':
        plt.xlabel('qx (m\u207B\u00B9)')
        plt.ylabel('qy (m\u207B\u00B9)')

    elif x_axis == 'qD':
        plt.xlabel('qx (m\u207B\u00B9)')
        plt.ylabel('qy (m\u207B\u00B9)')

    elif x_axis == 'xf':
        plt.xlabel('x(micron)')
        plt.ylabel('y(micron)')

    ax.set_aspect('equal')
    #plt.xlim(0, 4.5e7)   # set the xlim to left, right
    #plt.ylim(0, 4.5e7)
    cbar = plt.colorbar(plot, shrink = 0.65)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()

    if save_plot:

        if log_scale:
            plt.savefig( folder_write + '2D_Sq_' + file + "_edge_" + str(edge) + '_log_scale.png')

        else:
            plt.savefig( folder_write + '2D_Sq_' + file + "_edges_" + str(edge) + '.svg')

    plt.show()


def vector_selection(array, axis_selection):

    if axis_selection == 'q':
        axis = array[-2]*1e6

    elif axis_selection == 'qD':
        axis = array[-1]

    elif axis_selection == 'xf':
        axis = array[1]*1e6
        axis /= 1.033e4

    return axis


def plot_Sq_1D(folder_read, file, labels, edges, x_axis = 'qD', save_plot = False, folder_write ='', log_scale = False, moving_average = False, n_ave = 3, y_max = 100):

    # Configure matplotlib once before plotting
    plt.rcParams.update({'font.size': 20})
    plt.rc('legend', fontsize=20)
    plt.rc('axes', titlesize=30)
    plt.rc('axes', labelsize=30)
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    figure(num=None, figsize=(15, 10), dpi=100, facecolor='w', edgecolor='k')

    markers_array = np.asarray(["v", "o", "^", "s", "<", "x", ">","v", "o", "^", "s", "<", "x", ">"])
    for count, Sq_and_array in enumerate(file):

        Sq_and_arrays = np.genfromtxt(folder_read + str(file[count]) + '.dat', delimiter=',')

        Sq = Sq_and_arrays[0]


        if moving_average:

            Sq_mov_ave = np.zeros(len(Sq) - n_ave)

            for p in range(len(Sq) - n_ave):
                Sq_mov_ave[p] = np.mean(Sq[p:p + n_ave])

            Sq = np.append(Sq[:n_ave], Sq_mov_ave)

        vector = vector_selection(Sq_and_arrays,x_axis)

        Sq = Sq[(np.logical_and((edges[0] < vector), (vector < edges[1])))]
        vector = vector[(np.logical_and((edges[0] < vector), (vector < edges[1])))]


        plt.plot(vector,Sq, marker = markers_array[count], ms=6, lw=2, label= labels[count])


    if log_scale:
        plt.yscale("log")

    if x_axis == 'q':
        plt.xlabel('q (m\u207B\u00B9)', fontsize=35)

    elif x_axis == 'qD':
        plt.xlabel('q (m\u207B\u00B9)', fontsize=35)

    elif x_axis == 'xf':
        plt.xlabel('x(micron)', fontsize=35)

    plt.ylabel('S(q)', fontsize=35)
    plt.axvline(x=2.19e6, color='k', lw = 2, linestyle='dashed')
    plt.axvline(x=1.57e7, color='k', lw = 2, linestyle='dashed')
    plt.axhline(y=0.05, color='g', lw = 2)
    plt.axhline(y=0.1, color='b', lw = 2)
    plt.ylim([0,y_max])
    plt.xlim(edges)
    plt.legend()
    plt.tight_layout

    if save_plot:
        plt.tight_layout
        plt.savefig( folder_write + 'Sq_' + 'multiple_filtered_areas' + "_edges_" + str(edges[0]) + '_' + str(edges[1]) + '.svg')
