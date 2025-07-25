# FDTDX Documentation

![image](img/logo.png)

FDTDX is a high-performance framework for electromagnetic simulations and inverse design of photonic devices. Built on JAX, it provides GPU-accelerated FDTD (Finite-Difference Time-Domain) simulations with automatic differentiation capabilities.

## Installation

Install FDTDX using pip:

```bash
pip install fdtdx  # Basic CPU-Installation
pip install fdtdx[cuda12]  # GPU-Acceleration (Highly Recommended!)
pip install fdtdx[rocm]   # AMD-GPU (only python<=3.12)
```

For development installation, clone the repository and install in editable mode:

```bash
git clone https://github.com/ymahlau/fdtdx
cd fdtdx
pip install -e .
```

## Key Features

### High Performance Computing
- Native GPU acceleration through JAX
- Multi-GPU scaling for large simulations 
- Memory-efficient time-reversal implementation
- Optimized for large-scale inverse design
- Flexible boundary conditions with PML support

## Guides

- [Object Placement Guide](tutorials/object_placement.md) - Learn how to position and configure simulation objects
- [Materials Guide](tutorials/materials.md) - Learn how to use materials in FDTDX
- [Fabrication Constraints](tutorials/parameter_mapping.md) - Learn how to use the parameter mapping API to include fabrication constraints
- [Interface Compression](tutorials/interface_compression.md) - Learn how to use The compresion API to compute gradients with reversible autodiff
- See the examples folder for complete scripts for inverse design in FDTDX
- More guides will follow shortly

## Citation

If you find this repository helpful for your work, please consider citing:

```bibtex
@article{schubert2025quantized,
  title={Quantized inverse design for photonic integrated circuits},
  author={Schubert, Frederik and Mahlau, Yannik and Bethmann, Konrad and Hartmann, Fabian and Caspary, Reinhard and Munderloh, Marco and Ostermann, J{\"o}rn and Rosenhahn, Bodo},
  journal={ACS omega},
  volume={10},
  number={5},
  pages={5080--5086},
  year={2025},
  publisher={ACS Publications}
}
```

