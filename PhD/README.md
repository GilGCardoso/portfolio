# High-Performance Optical Simulation for non periodic structures

## Problem Statement
Simulate optical properties of non-periodic structures using particle positions through structure factor calculations. Challenge: Handle 20k-40k particles with limited computational resources.

## Technical Solution
- **Algorithm**: Discretized Fourier Transform for structure factor calculation
- **Performance**: Dual implementation (CPU/GPU) for optimal resource utilization
- **Scalability**: Memory-efficient processing up to 40k particles on 32GB RAM
- **Adaptive memory management**: automatic selection between three compute tiers (full-matrix, chunked, iterative) based on available RAM/VRAM — same API runs on a laptop or a workstation without code changes
- **User feedback**: prints selected tier and shows a progress bar during the calculation

## Key Achievements
- 🚀 **Performance**: ?? speedup with GPU implementation
- 📊 **Scale**: Successfully processed structures with 40,000+ particles
- 🔧 **Optimization**: Memory-efficient algorithms for resource-constrained environments
- 💻 **Portability**: Cross-platform CPU/GPU implementations

## Business Impact
- **Materials Design**: Rapid prototyping of optical materials
- **Manufacturing**: Process optimization for photonic devices
- **R&D Efficiency**: Reduced simulation time for non-periodic structures