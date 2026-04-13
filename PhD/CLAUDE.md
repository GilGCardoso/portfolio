# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD research toolkit for simulating optical properties of non-periodic particle structures via structure factor S(q) calculations. Handles 20k–40k particle positions with dual CPU/GPU implementations.

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Running Calculations

Import directly from the `phd_tools` package:

```python
from phd_tools.structure_factor import calculate_structure_factor

calculate_structure_factor(
    read_folder='data/',
    file='test_grid_31x31',   # filename without .csv extension
    range_calculation=10,
    vector_step=1,
    save_folder='notebooks/results/'
)
```

Or run the relevant notebook in `notebooks/`.

## Repository Layout

```
phd_tools/          # Python package — import from here
  sem.py            # SEM image → particle positions (Sauvola threshold + skimage)
  structure_gen.py  # Correlated disorder generation + density reduction
  structure_factor.py  # S(q) calculation, radial average, save, plot (PyTorch)
notebooks/
  sem_detection.ipynb         # SEM machine vision workflow
  structure_generation.ipynb  # Structure generation workflow
  structure_factor.ipynb      # S(q) calculation and plotting workflow
  results/                    # .dat output files written here at runtime
data/               # Input CSV files (particle coordinates, no header, x,y columns)
archive/
  structure_factor_cupy.py    # Legacy cupy implementation (CUDA-only, not maintained)
```

## Architecture

### Data format

- **Input**: CSV, no header, two columns — x and y particle coordinates.
- **Output** `.dat` files:
  - `2D_Sq_<name>_range_<R>_step_<s>.dat` — 2D S(q) matrix with q and q·D as last two columns.
  - `Sq_<name>_range_<R>_step_<s>.dat` — three rows: S(q), q, q·D.

### Structure factor pipeline (`phd_tools/structure_factor.py`)

1. **`get_calculation_parameters`** — average nearest-neighbour distance D and particle count N via `cKDTree`.
2. **`generate_vectors`** — pairwise distance arrays (d_x, d_y) and scattering vector q (normalised by D).
3. **Adaptive calculation** — `_select_method` picks one of three tiers based on available memory (`psutil` on CPU, `torch.cuda.mem_get_info` on GPU) and tensor sizes:
   - `matrix` — full broadcast `(nq, nq, N²)`, fastest when it fits.
   - `matrix_by_parts` — chunks over qx rows; halves chunk on runtime OOM.
   - `iterative` — per-cell loop for the tightest budgets.
   Uses real `cos` (S(q) is real-valued) in float32. Safety factors: `_SAFETY_GPU=0.85`, `_SAFETY_CPU=0.5` (CPU is more conservative — `psutil.available` overstates usable RAM). `device='cpu'` or `'gpu'` is an explicit arg.
4. **`radial_average`** — bins 2D S(q) into radial shells via `np.digitize`.
5. **`save_data`** — writes both `.dat` files.

### Correlated disorder generation (`phd_tools/structure_gen.py`)

`make_correlated_disorder` iteratively relaxes N random 2D points by replacing each with the centroid of its Voronoi cell (grid approximation in `calculate_new_positions`). Periodic boundary copies can be prepended before each step and removed afterward. `define_area_distance_and_radius` / `create_areas` build disordered polygonal regions for density reduction.

## Known Issues / Gotchas

- `plot_Sq_2D` and `plot_Sq_1D` expect `file` **without** the `.dat` extension — they append it internally. Passing the full filename causes a double-extension `FileNotFoundError`.
- `binary_conversion` in `sem.py` calls `cv2.blur` but `cv2` (OpenCV) is not listed in `requirements.txt` and not imported at the top of the file — install `opencv-python` separately if using that function.
- The cupy archive (`archive/structure_factor_cupy.py`) requires `cupy` and a CUDA GPU; it is kept for reference only.
