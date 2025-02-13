# Numerical Quantum Theory (numpt)

numpt is a Python package for constructing adaptive, non-uniform meshes and generating quantum mechanical operators on these grids. It also includes routines for creating spin operators (spin‑1/2 and spin‑3/2) that can be embedded in higher-dimensional Hilbert spaces via tensor products. This tool is ideal for researchers and developers working on numerical quantum mechanics, computational physics, and related fields.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Adaptive Mesh Generation](#adaptive-mesh-generation)
  - [Canonical Operators](#canonical-operators)
  - [Spin Operators](#spin-operators)
  - [Visualization](#visualization)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Features

- **Adaptive Mesh Generation**:  
  Create non-uniform 1D, 2D, or 3D grids with smooth transitions in resolution using user-defined spacing functions. No more sharp boundaries between high- and low-resolution regions.

- **Canonical Operators**:  
  Construct quantum mechanical operators (e.g., momentum \( p \), kinetic energy \( p^2 \), position \( x \), and potential \( x^2 \)).

- **Spin Operators**:  
  Build spin‑3/2 and spin‑1/2 operators (with their squared counterparts) and embed them into tensor product spaces if extra degrees of freedom (e.g., spatial dimensions) are provided.

- **Flexible Tensor Embedding**:  
  Operators are automatically “embedded” into the full Hilbert space only for the extra dimensions specified by the user.

- **Modular & Extensible**:  
  Designed for integration with quantum simulation workflows. Easily extendable and customizable to fit your research needs.

## Installation

### Requirements

- Python 3.6+
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- (Optional) [Matplotlib](https://matplotlib.org/) for 1D/2D visualization
- (Optional) [Plotly](https://plotly.com/python/) and [scikit-image](https://scikit-image.org/) for 3D visualization

### Installing from Source

Clone the repository and install dependencies:

```bash
git clone https://github.com/OtiDioti/numpt.git
cd numpt
pip install -r requirements.txt
```

For development, you can install in editable mode:

```bash
pip install -e .
```

## Usage

### Adaptive Mesh Generation

The package includes a `Mesh` class that lets you generate adaptive grids. For example, to create a 1D adaptive mesh with smooth spacing:

```python
from mesh import Mesh

# Define a spacing function for x:
dx_func = lambda x: 0.05 if 0.4 <= x <= 0.6 else 0.2

# Create a 1D mesh over [0, 1]
mesh = Mesh(
    dims=1,
    xbounds=(0, 1),
    initial_points=10,
    dx_max=0.2,
    dx_func=dx_func,
    max_iter=20
)
```

For 2D/3D meshes, provide the additional bounds and spacing functions (dy_func, dz_func) as needed.

### Canonical Operators

The `canonic_ops` class builds discretized quantum operators on the adaptive mesh with selectable order of accuracy. For instance, to construct operators with third‑order accuracy:

```python
from canonic_ops import canonic_ops

# Create canonical operators with third-order accuracy:
ops = canonic_ops(hbar=1, order=3)
p_operator = ops.p(mesh.mesh_x)     # momentum operator in x
p2_operator = ops.p2(mesh.mesh_x)    # kinetic energy operator in x
x_operator = ops.x(mesh.mesh_x)       # position operator in x
x2_operator = ops.x2(mesh.mesh_x)     # x^2 operator
```

If working in 2D or 3D, you can similarly build operators for each coordinate.

### Spin Operators

The package includes functions for generating spin operators. For a spin‑3/2 system:

```python
from spin_ops import spin32

# Generate spin-3/2 operators and embed them in an extra spatial Hilbert space of dimension 50:
J0, Jx, Jy, Jz, Jx2, Jy2, Jz2 = spin32(dimx=50, hbar=1)
```

For a spin‑1/2 system:

```python
from spin_ops import spin12

# Generate spin-1/2 operators (Pauli matrices) without extra embedding:
S0, Sx, Sy, Sz, Sx2, Sy2, Sz2 = spin12(hbar=1)
```

If extra dimensions (dimx, dimy, or dimz) are provided, the functions automatically embed the operators via the Kronecker product.

### Visualization

The package supports visualization of both the adaptive mesh and the computed operators:

- **1D/2D Visualization**:  
  Use Matplotlib to plot grid points or operator eigenfunctions.
  
  ```python
  mesh.visualize_grid()  # Visualizes the 1D or 2D grid.
  ```

- **3D Visualization**:  
  Use Plotly along with scikit-image’s marching cubes algorithm to display 3D isosurfaces.

  ```python
  from hamiltonian import Hamiltonian
  
  # Construct your Hamiltonian sparse matrix H...
  ham = Hamiltonian(H, mesh)
  energies, wavefunctions = ham.solve(k=5)
  ham.plot(wf_index=0)
  ```

## Examples

Visit the `examples/` directory for detailed usage examples:
- Adaptive mesh generation in 1D, 2D, and 3D.
- Construction and diagonalization of Hamiltonians.
- Embedding and visualizing spin operators.

## Contributing

Contributions are welcome! To contribute:
- Fork the repository.
- Create a feature branch.
- Write tests and update documentation.
- Submit a pull request.

Please adhere to the existing code style and include sufficient documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The open-source community for providing robust libraries such as NumPy, SciPy, Matplotlib, Plotly, and scikit-image.
