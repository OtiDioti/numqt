# NumQT - Numerical Quantum Toolkit

NumQT is a Python package designed for numerical simulations in quantum mechanics. It provides a comprehensive set of tools for solving quantum mechanical problems using finite difference methods as well as spectral techniques.

## Features

- **Flexible mesh generation** with support for 1D, 2D, and 3D systems
- **Canonical operators** (position, momentum, and derivatives) in multiple dimensions
- **Efficient Hamiltonian construction** 
- **Eigenvalue solvers** for finding energy states and wave functions
- **Rich visualization tools** for plotting wave functions and probability densities
- **Support for different basis functions** and spectral methods

## Installation

You can install NumQT  manually:

```bash
git clone https://github.com/OtiDioti/numqt.git
cd numqt
pip install -e .
```

## Dependencies

NumQT requires the following Python packages:

- NumPy
- SciPy
- Matplotlib
- Plotly (for 3D visualizations)

## Quick Start

Here's a minimal example to solve the quantum harmonic oscillator problem:

```python
from numqt import *

# Define mesh parameters
Lx = 1  # Characteristic length
xbounds = (-15*Lx, 15*Lx)  # Domain boundaries
dx = 0.1  # Grid spacing
nx = int(np.abs(xbounds[1] - xbounds[0]) / dx)

# Create mesh
mesh_obj = Mesh(dims=1, xbounds=xbounds, nx=nx)

# Physical parameters
hbar = 1
m = 1  # Mass
wx = hbar / (m * Lx**2)  # Frequency

# Get canonical operators
operators = canonic_ops(mesh_obj, ops_to_compute=["p2", "x2"], additional_subspaces=None, hbar=1)
px2 = operators.get_ops()["p2"]
x2 = operators.get_ops()["x2"]

# Create Hamiltonian
H = Hamiltonian(px2 / (2*m) + 0.5 * m * x2 * wx**2, mesh_obj)

# Solve for eigenvalues and eigenfunctions
k = 10  # Number of states to compute
energies, wavefunctions = H.solve(k)

# Plot ground state wavefunction
H.plot(0)
```

## Core Classes

### Mesh

The `Mesh` class is the foundation of NumQT, providing a discretized representation of space for quantum mechanical calculations.

```python
mesh = Mesh(dims=1, xbounds=(-10, 10), nx=200)
```

Parameters:
- `dims`: Dimensionality of the problem (1, 2, or 3)
- `xbounds`, `ybounds`, `zbounds`: Domain boundaries in each dimension
- `nx`, `ny`, `nz`: Number of grid points in each dimension

Methods:
- `generate_mesh()`: Creates the mesh grid
- `get_grids()`: Returns mesh coordinates
- `get_spacings()`: Returns grid spacing
- `visualize_grid()`: Displays the mesh grid
- `visualize_slice()`: Shows a slice of the grid

### Canonical Operators

The `canonic_ops` class constructs quantum mechanical operators like position, momentum, and their derivatives.

```python
operators = canonic_ops(mesh_obj, ops_to_compute=["p", "p2", "x", "x2"], hbar=1)
```

Parameters:
- `mesh`: A Mesh object
- `ops_to_compute`: List of operators to construct ("p", "p2", "x", "x2")
- `basis`: Optional basis functions for spectral methods
- `additional_subspaces`: For tensor product spaces
- `hbar`: Planck's constant

Methods:
- `get_ops()`: Returns constructed operators as a dictionary

### Hamiltonian

The `Hamiltonian` class represents quantum mechanical Hamiltonians and solves the time-independent Schrödinger equation.

```python
H = Hamiltonian(px2 / (2*m) + 0.5 * m * x2 * wx**2, mesh_obj)
```

Parameters:
- `H`: Hamiltonian operator (sparse matrix)
- `mesh`: Mesh object
- `basis`: Optional basis for spectral methods
- `other_subspaces_dims`: Dimensions of additional subspaces

Methods:
- `solve(k)`: Finds k lowest energy eigenstates
- `plot(wf_index)`: Visualizes an eigenfunction
- `visualize_slice()`: Displays a slice of 2D/3D wavefunctions

## Contributing

Contributions to NumQT are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use NumQT in your research, please cite:

```
@software{numqt,
  author = {NumQT Developers},
  title = {NumQT: Numerical Quantum Toolkit},
  url = {https://github.com/username/numqt},
  year = {2023},
}
```

## Acknowledgments

NumQT was inspired by various numerical techniques in quantum mechanics and benefits from the robust numerical capabilities provided by NumPy and SciPy.
