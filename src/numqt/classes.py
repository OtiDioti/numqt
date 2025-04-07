import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, eye, kron
from .utils import iterative_kron
from skimage import measure
import plotly.graph_objects as go
from scipy.sparse.linalg import eigsh

class Mesh:
    def __init__(self, 
                 dims: int,
                 xbounds: tuple,
                 ybounds: tuple = None,
                 zbounds: tuple = None,
                 nx: int = 50,
                 ny: int = None,
                 nz: int = None):
        """
        Constructs a uniform structured grid in 1D, 2D, or 3D.

        Parameters
        ----------
        dims : int
            Dimensionality of the grid. Must be 1, 2, or 3.
        xbounds, ybounds, zbounds : tuple
            Endpoints of the domain in x, y, and z. (Unused bounds are ignored for dims < 3.)
        nx, ny, nz : int
            Number of grid points along x, y, and z axes (including endpoints).
            If ny or nz are not provided, they default to nx.
        """
        # Validate the dimensionality
        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3.")

        # Validate domain bounds
        if xbounds is None:
            raise ValueError("xbounds must be provided.")
        if dims >= 2 and ybounds is None:
            raise ValueError("For dims>=2, ybounds must be provided.")
        if dims == 3 and zbounds is None:
            raise ValueError("For dims==3, zbounds must be provided.")

        # Set default values for grid points if not provided
        if ny is None:
            ny = nx
        if nz is None:
            nz = nx

        # Save parameters
        self.dims = dims
        self.xbounds = xbounds
        self.ybounds = ybounds if dims >= 2 else None
        self.zbounds = zbounds if dims == 3 else None
        
        # Unpack domain bounds for convenience
        self.x0, self.x1 = xbounds
        if dims >= 2:
            self.y0, self.y1 = ybounds
        if dims == 3:
            self.z0, self.z1 = zbounds
            
        # Generate uniform grid based on the dimensionality
        self.generate_mesh(nx, ny, nz)
        
        # Store grid sizes
        self.Nx = len(self.mesh_x)
        self.Ny = len(self.mesh_y) if dims >= 2 else 0
        self.Nz = len(self.mesh_z) if dims == 3 else 0

    def generate_mesh(self, nx, ny, nz):
        """Creates uniform grids for each dimension."""
        # Create x grid
        self.mesh_x = np.linspace(self.x0, self.x1, nx)
        
        # Create y grid if needed
        if self.dims >= 2:
            self.mesh_y = np.linspace(self.y0, self.y1, ny)
        else:
            self.mesh_y = None
            
        # Create z grid if needed
        if self.dims == 3:
            self.mesh_z = np.linspace(self.z0, self.z1, nz)
        else:
            self.mesh_z = None
            
    def get_grids(self):
        """
        Returns the grid arrays.

        Returns
        -------
        For dims==1: mesh_x
        For dims==2: mesh_x, mesh_y
        For dims==3: mesh_x, mesh_y, mesh_z
        """
        if self.dims == 1:
            return self.mesh_x
        elif self.dims == 2:
            return self.mesh_x, self.mesh_y
        elif self.dims == 3:
            return self.mesh_x, self.mesh_y, self.mesh_z
    
    def get_spacings(self):
        """
        Returns the grid spacings for each dimension.
        
        Returns
        -------
        For dims==1: dx
        For dims==2: dx, dy
        For dims==3: dx, dy, dz
        """
        dx = (self.x1 - self.x0) / (self.Nx - 1)
        
        if self.dims == 1:
            return dx
        elif self.dims == 2:
            dy = (self.y1 - self.y0) / (self.Ny - 1)
            return dx, dy
        elif self.dims == 3:
            dy = (self.y1 - self.y0) / (self.Ny - 1)
            dz = (self.z1 - self.z0) / (self.Nz - 1)
            return dx, dy, dz

    # ----------------------------------------------------------
    # Visualization Methods
    # ----------------------------------------------------------

    def visualize_grid(self, marker_size=10, elev=30, azim=45, alpha=0.1):
        """
        Visualizes the grid.
        
        - In 1D, displays the grid points on a line.
        - In 2D, plots the full tensor product of the 1D grids.
        - In 3D, shows a 3D scatter plot of the tensor product.
        """
        if self.dims == 1:
            plt.figure(figsize=(8, 2))
            plt.scatter(self.mesh_x, np.zeros_like(self.mesh_x), s=marker_size, c='b', alpha=alpha)
            plt.xlabel('x')
            plt.title("1D Grid Visualization")
            plt.yticks([])
            plt.show()
        elif self.dims == 2:
            X, Y = np.meshgrid(self.mesh_x, self.mesh_y, indexing='ij')
            plt.figure(figsize=(8, 6))
            plt.scatter(X.flatten(), Y.flatten(), s=marker_size, c='b', alpha=alpha)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("2D Grid Visualization")
            plt.show()
        elif self.dims == 3:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            # Create the full 3D tensor product of grid points.
            X, Y, Z = np.meshgrid(self.mesh_x, self.mesh_y, self.mesh_z, indexing='ij')
            ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), s=marker_size, c='b', alpha=alpha)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=elev, azim=azim)
            plt.title("3D Grid Visualization")
            plt.show()
    
    def visualize_slice(self, slice_axis: str, slice_value: float, marker_size=20, alpha=0.7, c="b"):
        """
        Visualizes a slice of the grid.
        
        In 2D, the slice will be a 1D set of points;
        in 3D, the slice will be a 2D plane of points.
        (Slicing is not defined for 1D grids.)
        
        Parameters
        ----------
        slice_axis : str
            The axis along which to take the slice ('x', 'y', or 'z').
        slice_value : float
            The coordinate value at which to slice.
        """
        slice_axis = slice_axis.lower()
        if self.dims == 1:
            raise ValueError("Slicing is not applicable for 1D grids.")
        elif self.dims == 2:
            if slice_axis == 'x':
                idx = np.argmin(np.abs(self.mesh_x - slice_value))
                slice_coord = self.mesh_x[idx]
                # Fixed x, vary y.
                Y = self.mesh_y
                X = np.full_like(Y, slice_coord)
                plt.figure(figsize=(8, 4))
                plt.scatter(X, Y, s=marker_size, c=c, alpha=alpha)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f"Slice at x = {slice_coord:.2f} (closest to {slice_value})")
            elif slice_axis == 'y':
                idx = np.argmin(np.abs(self.mesh_y - slice_value))
                slice_coord = self.mesh_y[idx]
                # Fixed y, vary x.
                X = self.mesh_x
                Y = np.full_like(X, slice_coord)
                plt.figure(figsize=(8, 4))
                plt.scatter(X, Y, s=marker_size, c=c, alpha=alpha)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f"Slice at y = {slice_coord:.2f} (closest to {slice_value})")
            else:
                raise ValueError("For 2D grids, slice_axis must be 'x' or 'y'.")
            plt.grid(True)
            plt.show()
        elif self.dims == 3:
            if slice_axis == 'x':
                idx = np.argmin(np.abs(self.mesh_x - slice_value))
                slice_coord = self.mesh_x[idx]
                # Build the 2D grid in y-z.
                Y, Z = np.meshgrid(self.mesh_y, self.mesh_z, indexing='ij')
                plt.figure(figsize=(8, 6))
                plt.scatter(Y.flatten(), Z.flatten(), s=marker_size, c=c, alpha=alpha)
                plt.xlabel('y')
                plt.ylabel('z')
                plt.title(f"Slice at x = {slice_coord:.2f} (closest to {slice_value})")
            elif slice_axis == 'y':
                idx = np.argmin(np.abs(self.mesh_y - slice_value))
                slice_coord = self.mesh_y[idx]
                # Build the 2D grid in x-z.
                X, Z = np.meshgrid(self.mesh_x, self.mesh_z, indexing='ij')
                plt.figure(figsize=(8, 6))
                plt.scatter(X.flatten(), Z.flatten(), s=marker_size, c=c, alpha=alpha)
                plt.xlabel('x')
                plt.ylabel('z')
                plt.title(f"Slice at y = {slice_coord:.2f} (closest to {slice_value})")
            elif slice_axis == 'z':
                idx = np.argmin(np.abs(self.mesh_z - slice_value))
                slice_coord = self.mesh_z[idx]
                # Build the 2D grid in x-y.
                X, Y = np.meshgrid(self.mesh_x, self.mesh_y, indexing='ij')
                plt.figure(figsize=(8, 6))
                plt.scatter(X.flatten(), Y.flatten(), s=marker_size, c=c, alpha=alpha)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f"Slice at z = {slice_coord:.2f} (closest to {slice_value})")
            else:
                raise ValueError("slice_axis must be one of 'x', 'y', or 'z'.")
            plt.grid(True)
            plt.show()

# ----------------------------------------------------------
# ----------------------------------------------------------
# Canonic Operators
# ----------------------------------------------------------
# ----------------------------------------------------------

class canonic_ops:
    def __init__(self, mesh, additional_subspaces=None, hbar=1):
        """
        Constructs the discretized quantum mechanical operators for position and momentum
        using finite difference methods on a uniform grid (only the interior points are used).

        Parameters
        ----------
        mesh : object
            Must have an attribute "dims" (equal to 1, 2, or 3) and a get_grids() method
            returning the 1D grid arrays. For dims==1, get_grids() returns a 1D array;
            for dims==2, it returns (mesh_x, mesh_y); for dims==3, (mesh_x, mesh_y, mesh_z).
        additional_subspaces : list, optional
            A list with the structure [position, I_second, I_third, ...] used to embed the operator
            into a tensor product space. (If not provided, the operators are returned as is.)
        hbar : float, optional
            Planck’s constant (defaults to 1).
        """
        self.hbar = hbar
        self.dim = mesh.dims
        self.additional_subspaces = additional_subspaces

        # Get the grid(s) from the mesh and compute the number of interior points.
        if self.dim == 1:
            self.mesh_x = mesh.get_grids()  # expected 1D array
            self.Nx = len(self.mesh_x) - 2
        elif self.dim == 2:
            self.mesh_x, self.mesh_y = mesh.get_grids()
            self.Nx = len(self.mesh_x) - 2
            self.Ny = len(self.mesh_y) - 2
        elif self.dim == 3:
            self.mesh_x, self.mesh_y, self.mesh_z = mesh.get_grids()
            self.Nx = len(self.mesh_x) - 2
            self.Ny = len(self.mesh_y) - 2
            self.Nz = len(self.mesh_z) - 2

        # Pre-construct identity matrices on the interior grid.
        self.Ix = eye(self.Nx, format='csr')
        if self.dim >= 2:
            self.Iy = eye(self.Ny, format='csr')
        if self.dim == 3:
            self.Iz = eye(self.Nz, format='csr')

        # Generate operators for each coordinate direction.
        if self.dim == 1:
            self.px    = self.p(self.mesh_x)
            self.px2   = self.p2(self.mesh_x)
            self.x_op  = self.x(self.mesh_x)
            self.x2_op = self.x2(self.mesh_x)
        elif self.dim == 2:
            self.px    = self.p(self.mesh_x)
            self.py    = self.p(self.mesh_y)
            self.px2   = self.p2(self.mesh_x)
            self.py2   = self.p2(self.mesh_y)
            self.x_op  = self.x(self.mesh_x)
            self.y_op  = self.x(self.mesh_y)
            self.x2_op = self.x2(self.mesh_x)
            self.y2_op = self.x2(self.mesh_y)
        elif self.dim == 3:
            self.px    = self.p(self.mesh_x)
            self.py    = self.p(self.mesh_y)
            self.pz    = self.p(self.mesh_z)
            self.px2   = self.p2(self.mesh_x)
            self.py2   = self.p2(self.mesh_y)
            self.pz2   = self.p2(self.mesh_z)
            self.x_op  = self.x(self.mesh_x)
            self.y_op  = self.x(self.mesh_y)
            self.z_op  = self.x(self.mesh_z)
            self.x2_op = self.x2(self.mesh_x)
            self.y2_op = self.x2(self.mesh_y)
            self.z2_op = self.x2(self.mesh_z)

    def get_ops(self):
        """
        Returns the constructed operators in a dictionary.
        For dims==1, only the x-operators are returned.
        For dims==2, x- and y-operators are returned.
        For dims==3, x-, y-, and z-operators are returned.
        """
        if self.dim == 1:
            ops = {
                "p":  self.px,
                "p2": self.px2,
                "x":  self.x_op,
                "x2": self.x2_op
            }
        elif self.dim == 2:
            ops = {
                "p":  [self.px, self.py],
                "p2": [self.px2, self.py2],
                "x":  [self.x_op, self.y_op],
                "x2": [self.x2_op, self.y2_op]
            }
        elif self.dim == 3:
            ops = {
                "p":  [self.px, self.py, self.pz],
                "p2": [self.px2, self.py2, self.pz2],
                "x":  [self.x_op, self.y_op, self.z_op],
                "x2": [self.x2_op, self.y2_op, self.z2_op]
            }
        return ops

    def p(self, mesh):
        """
        Constructs the discretized momentum operator (p = -i*hbar*d/dx) using a
        2-point central difference stencil (2nd-order accurate) on a uniform grid.
    
        Parameters
        ----------
        mesh : ndarray
            1D array for the grid in the given dimension.
        other_subspaces : list, optional
            If provided, should be of the form [position, I2, I3, ...] and the operator will be
            embedded in the tensor product via iterative_kron.
        """
        dx = np.abs(mesh[1] - mesh[0])
        N = len(mesh) - 2  # only use interior points
    
        # Simple central difference: df/dx ≈ [f(x+dx) - f(x-dx)]/(2*dx)
        offsets = np.array([-1, 1])
        coeffs = np.array([-1, 1], dtype=complex)
        prefactor = -1j * self.hbar / (2 * dx)
        diagonals = [
            prefactor * coeff * np.ones(N - abs(offset), dtype=complex)
            for offset, coeff in zip(offsets, coeffs)
        ]
        p_mat = diags(diagonals, offsets, shape=(N, N), format='csr')
    
        if self.additional_subspaces is not None:
            pos = self.additional_subspaces[0]
            kron_list = self.additional_subspaces[1:].copy()
            kron_list.insert(pos, p_mat)
            return iterative_kron(kron_list)
        else:
            return p_mat
    
    def p2(self, mesh):
        """
        Constructs the discretized squared momentum operator (p² = -ħ² d²/dx²)
        using a 3-point central difference stencil (2nd-order accurate) on a uniform grid.
    
        Parameters
        ----------
        mesh : ndarray
            1D array for the grid in the given dimension.
        other_subspaces : list, optional
            If provided, should be of the form [position, I2, I3, ...] to embed the operator
            into a tensor product space.
        """
        dx = np.abs(mesh[1] - mesh[0])
        N = len(mesh) - 2  # interior points only
    
        # Standard 3-point stencil for second derivative:
        # d²f/dx² ≈ [f(x-dx) - 2*f(x) + f(x+dx)]/(dx²)
        offsets = np.array([-1, 0, 1])
        coeffs = np.array([1, -2, 1], dtype=float)
        prefactor = -self.hbar**2 / (dx**2)
        diagonals = [
            prefactor * coeff * np.ones(N - abs(offset), dtype=float)
            for offset, coeff in zip(offsets, coeffs)
        ]
        p2_mat = diags(diagonals, offsets, shape=(N, N), format='csr')
    
        if self.additional_subspaces is not None:
            pos = self.additional_subspaces[0]
            kron_list = self.additional_subspaces[1:].copy()
            kron_list.insert(pos, p2_mat)
            return iterative_kron(kron_list)
        else:
            return p2_mat
    
    def x(self, mesh):
        """
        Constructs the discretized position operator
        (defined on interior points only).
    
        Parameters
        ----------
        mesh : ndarray
            1D grid array.
        other_subspaces : list, optional
            If provided, should be of the form [position, I2, I3, ...] to embed the operator
            into a tensor product space.
        """
        x_interior = mesh[1:-1]
        X = diags(x_interior, 0, format='csr')
    
        if self.additional_subspaces is not None:
            pos = self.additional_subspaces[0]
            kron_list = self.additional_subspaces[1:].copy()
            kron_list.insert(pos, X)
            return iterative_kron(kron_list)
        else:
            return X
    
    def x2(self, mesh):
        """
        Constructs the discretized x² operator.
    
        Parameters
        ----------
        mesh : ndarray
            1D grid array.
        other_subspaces : list, optional
            If provided, should be of the form [position, I2, I3, ...] to embed the operator
            into a tensor product space.
        """
        x_interior = mesh[1:-1]
        X2 = diags(x_interior**2, 0, format='csr')
    
        if self.additional_subspaces is not None:
            pos = self.additional_subspaces[0]
            kron_list = self.additional_subspaces[1:].copy()
            kron_list.insert(pos, X2)
            return iterative_kron(kron_list)
        else:
            return X2
    def project(self, basis):
        """
        Projects the canonical operators (position, momentum, and their squares) onto a new basis set.
        
        For 1D, the `basis` parameter should be a tuple:
            (fn, x0, x1, dx, N)
        For 2D or 3D, `basis` can be provided either as:
            - A single tuple, in which case the same basis is used for all directions, or
            - A dictionary with keys "x", "y", (and "z") where each value is a tuple
              (fn, x0, x1, dx, N) for the corresponding direction.
              
        The projection is performed using the provided functions:
          - p(fn, x0, x1, dx, N, sparse)
          - p2(fn, x0, x1, dx, N, sparse)
          - x(fn, x0, x1, dx, N, sparse)
          - x2(fn, x0, x1, dx, N, sparse)
        
        These functions are assumed to be defined in the global scope.
        The projected operators are computed via discretized integration and are always returned as sparse matrices.
        
        Parameters
        ----------
        basis : tuple or dict
            Basis information. Either a single tuple (fn, x0, x1, dx, N) to be used for all directions,
            or a dict with keys "x", "y", (and "z") for higher dimensions.
        
        Returns
        -------
        dict
            A dictionary of projected operators. For 1D:
                {"p": p_proj, "p2": p2_proj, "x": x_proj, "x2": x2_proj}
            For 2D:
                {"p": [p_x_proj, p_y_proj],
                 "p2": [p2_x_proj, p2_y_proj],
                 "x": [x_x_proj, x_y_proj],
                 "x2": [x2_x_proj, x2_y_proj]}
            For 3D, similarly with an entry for z.
        """
        # Helper to extract basis info for a given direction.
        # If basis is a tuple, then use it for all directions.
        def get_basis(direction):
            if isinstance(basis, dict):
                return basis.get(direction)
            else:
                return basis

        # For each direction, retrieve the basis info.
        if self.dim == 1:
            basis_x = get_basis("x")
            # Unpack basis info: (fn, x0, x1, dx, N)
            fn_x, x0_x, x1_x, dx_x, N_x = basis_x
            p_proj  = self.project_p(self.px, fn_x, x0_x, x1_x, dx_x, N_x)
            p2_proj = self.project_p2(self.px2, fn_x, x0_x, x1_x, dx_x, N_x)
            x_proj  = self.project_x(self.x, fn_x, x0_x, x1_x, dx_x, N_x)
            x2_proj = self.project_x2(self.x2, fn_x, x0_x, x1_x, dx_x, N_x)
            proj_ops = {"p": p_proj, "p2": p2_proj, "x": x_proj, "x2": x2_proj}

        elif self.dim == 2:
            basis_x = get_basis("x")
            basis_y = get_basis("y")
            # If basis_y is None, use basis_x for both directions.
            if basis_y is None:
                basis_y = basis_x

            fn_x, x0_x, x1_x, dx_x, N_x = basis_x
            fn_y, x0_y, x1_y, dx_y, N_y = basis_y

            p_proj_x  = self.project_p(self.px, fn_x, x0_x, x1_x, dx_x, N_x)
            p_proj_y  = self.project_p(self.py, fn_y, x0_y, x1_y, dx_y, N_y)
            p2_proj_x = self.project_p2(self.px2, fn_x, x0_x, x1_x, dx_x, N_x)
            p2_proj_y = self.project_p2(self.py2, fn_y, x0_y, x1_y, dx_y, N_y)
            x_proj_x  = self.project_x(self.x, fn_x, x0_x, x1_x, dx_x, N_x)
            x_proj_y  = self.project_x(self.y, fn_y, x0_y, x1_y, dx_y, N_y)
            x2_proj_x = self.project_x2(self.x2, fn_x, x0_x, x1_x, dx_x, N_x)
            x2_proj_y = self.project_x2(self.y2, fn_y, x0_y, x1_y, dx_y, N_y)

            proj_ops = {
                "p":  [p_proj_x, p_proj_y],
                "p2": [p2_proj_x, p2_proj_y],
                "x":  [x_proj_x, x_proj_y],
                "x2": [x2_proj_x, x2_proj_y]
            }

        elif self.dim == 3:
            basis_x = get_basis("x")
            basis_y = get_basis("y")
            basis_z = get_basis("z")
            if basis_y is None:
                basis_y = basis_x
            if basis_z is None:
                basis_z = basis_x

            fn_x, x0_x, x1_x, dx_x, N_x = basis_x
            fn_y, x0_y, x1_y, dx_y, N_y = basis_y
            fn_z, x0_z, x1_z, dx_z, N_z = basis_z
            
            p_proj_x  = self.project_p(self.px, fn_x, x0_x, x1_x, dx_x, N_x)
            p_proj_y  = self.project_p(self.py, fn_y, x0_y, x1_y, dx_y, N_y)
            p_proj_z  = self.project_p(self.pz, fn_z, x0_z, x1_z, dx_z, N_z)
            p2_proj_x = self.project_p2(self.px2, fn_x, x0_x, x1_x, dx_x, N_x)
            p2_proj_y = self.project_p2(self.py2, fn_y, x0_y, x1_y, dx_y, N_y)
            p2_proj_z = self.project_p2(self.pz2, fn_z, x0_z, x1_z, dx_z, N_z)
            x_proj_x  = self.project_x(self.x, fn_x, x0_x, x1_x, dx_x, N_x)
            x_proj_y  = self.project_x(self.y, fn_y, x0_y, x1_y, dx_y, N_y)
            x_proj_z  = self.project_x(self.z, fn_z, x0_z, x1_z, dx_z, N_z)
            x2_proj_x = self.project_x2(self.x2, fn_x, x0_x, x1_x, dx_x, N_x)
            x2_proj_y = self.project_x2(self.y2, fn_y, x0_y, x1_y, dx_y, N_y)
            x2_proj_z = self.project_x2(self.z2, fn_z, x0_z, x1_z, dx_z, N_z)

            proj_ops = {
                "p":  [p_proj_x, p_proj_y, p_proj_z],
                "p2": [p2_proj_x, p2_proj_y, p2_proj_z],
                "x":  [x_proj_x, x_proj_y, x_proj_z],
                "x2": [x2_proj_x, x2_proj_y, x2_proj_z]
            }
        else:
            raise ValueError("Unsupported dimension: {}".format(self.dim))

        return proj_ops



    
    def project_p(self, p_matrix, fn, x0, x1, dx, N):
        """
        Compute the matrix representation of the momentum operator in an eigenstate basis.
    
        This function projects the momentum operator (defined in the position basis using a 7-point
        central difference scheme) onto an eigenstate basis. The momentum operator in the position basis is:
        
        .. math::
            \hat{p} = -i \frac{d}{dx} \quad (\hbar = 1)
    
        and its 7-point discretization is given by:
        
        .. math::
            f'(x) \approx \frac{1}{60h} \left[f(x-3h) - 9f(x-2h) + 45f(x-h) - 45f(x+h) + 9f(x+2h) - f(x+3h)\right].
    
        The matrix elements in the eigenbasis are computed as:
        
        .. math::
            P_{mn} = dx \sum_{j} \psi_m^*(x_j) \left[\hat{p}\,\psi_n\right](x_j)
    
        where :math:`\psi_n(x)` is the discretized eigenstate given by ``fn(n, x0, x1, dx)``.
    
        Parameters
        ----------
        fn : callable
            Function that returns the discretized eigenstate for a given index.
            It should accept parameters (n, x0, x1, dx) and return a 1D NumPy array.
        x0 : float
            Lower bound of the spatial domain.
        x1 : float
            Upper bound of the spatial domain.
        dx : float
            Grid spacing.
        N : int
            Number of eigenstates (and the dimension of the discretized position space).
    
        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            An (N × N) complex matrix representing the momentum operator in the eigenstate basis.
        """
        
        # Precompute the eigenstate basis.
        # Each column of Psi corresponds to an eigenstate (assumed discretized on the same grid).
        Psi = np.array([fn(n, x0, x1, dx) for n in range(N)]).T  # Shape: (grid_points, N)
        
        # Compute the matrix elements via discretized integration (using dx as weight).
        p_mat = dx * Psi.conjugate().T @ (p_matrix @ Psi)
        return p_mat
    
    def project_p2(self, p2_matrix, fn, x0, x1, dx, N):
        """
        Compute the matrix representation of the momentum-squared operator in an eigenstate basis.
    
        This function projects the momentum-squared operator (constructed in the position basis using a 7-point
        central difference scheme) onto an eigenstate basis. The momentum-squared operator is defined as:
        
        .. math::
            \hat{p}^2 = -\frac{d^2}{dx^2} \quad (\hbar = 1)
    
        with its 7-point discretization given by:
        
        .. math::
            f''(x) \approx \frac{1}{180h^2} \left[-f(x-3h) + 12f(x-2h) - 39f(x-h) + 56f(x) - 39f(x+h) + 12f(x+2h) - f(x+3h)\right].
    
        The matrix elements are computed as:
        
        .. math::
            (P^2)_{mn} = dx \sum_{j} \psi_m^*(x_j) \left[\hat{p}^2\,\psi_n\right](x_j)
    
        Parameters
        ----------
        fn : callable
            Function that returns the discretized eigenstate for a given index.
        x0 : float
            Lower bound of the spatial domain.
        x1 : float
            Upper bound of the spatial domain.
        dx : float
            Grid spacing.
        N : int
            Number of eigenstates (and the dimension of the discretized position space).
    
    
        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            An (N × N) matrix representing the momentum-squared operator in the eigenstate basis.
        """
        
        # Precompute the eigenstate basis.
        Psi = np.array([fn(n, x0, x1, dx) for n in range(N)]).T
        
        # Compute the matrix elements via discretized integration.
        p2_mat = dx * Psi.conjugate().T @ (p2_matrix @ Psi)
        return p2_mat
    
    def project_x(self, x_matrix, fn, x0, x1, dx, N):
        """
        Compute the matrix representation of the position operator in an eigenstate basis.
    
        The position operator in the position basis is diagonal with entries corresponding to the spatial
        grid points. This function projects the operator onto the eigenstate basis using:
        
        .. math::
            X_{mn} = dx \sum_{j} \psi_m^*(x_j)\, x_j\, \psi_n(x_j)
    
        Parameters
        ----------
        fn : callable
            Function that returns the discretized eigenstate for a given index.
        x0 : float
            Lower bound of the spatial domain.
        x1 : float
            Upper bound of the spatial domain.
        dx : float
            Grid spacing.
        N : int
            Number of eigenstates (and the dimension of the discretized position space).
    
        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            An (N × N) matrix representing the position operator in the eigenstate basis.
        """
         
        # Precompute the eigenstate basis.
        Psi = np.array([fn(n, x0, x1, dx) for n in range(N)]).T
        
        # Compute the matrix elements via discretized integration.
        x_mat = dx * Psi.conjugate().T @ (x_matrix @ Psi)
        return x_mat
    
    def project_x2(self, x2_matrix, fn, x0, x1, dx, N):
        """
        Compute the matrix representation of the squared position operator in an eigenstate basis.
    
        This function projects the squared position operator, defined in the position basis as
        :math:`\hat{x}^2`, onto an eigenstate basis. The operator in the position basis is given by:
    
        .. math::
            (\hat{x}^2)_{ij} = x_i^2 \, \delta_{ij},
    
        where the grid points :math:`x_i` are defined on the interval (x0, x1) with spacing `dx`.
        The matrix elements in the eigenbasis are computed as:
    
        .. math::
            (X^2)_{mn} = dx \sum_{j} \psi_m^*(x_j) \, x_j^2 \, \psi_n(x_j)
    
        where :math:`\psi_n(x)` is the discretized eigenstate provided by ``fn(n, x0, x1, dx)``.
    
        Parameters
        ----------
        fn : callable
            Function that returns the discretized eigenstate for a given index.
            It should accept parameters (n, x0, x1, dx) and return a 1D NumPy array.
        x0 : float
            Lower bound of the spatial domain.
        x1 : float
            Upper bound of the spatial domain.
        dx : float
            Grid spacing between adjacent points.
        N : int
            Number of eigenstates (and the dimension of the discretized position space).
    
        Returns
        -------
        numpy.ndarray or scipy.sparse.spmatrix
            An (N × N) matrix representing the squared position operator in the eigenstate basis.
        """

        
        # Precompute the eigenstate basis.
        # Each column corresponds to an eigenstate (discretized on the same grid).
        Psi = np.array([fn(n, x0, x1, dx) for n in range(N)]).T  # Shape: (grid_points, N)
        
        # Compute the matrix elements via discretized integration.
        x2_mat = dx * Psi.conjugate().T @ (x2_matrix @ Psi)
        return x2_mat





# ----------------------------------------------------------
# ----------------------------------------------------------
# Hamiltonian
# ----------------------------------------------------------
# ----------------------------------------------------------

class Hamiltonian:
    def __init__(self, H, mesh, other_subspaces_dims=None):
        """
        Parameters
        ----------
        H : sparse matrix
            The Hamiltonian operator defined on the interior points (after applying Dirichlet BC).
        mesh : object
            A mesh object that contains the grid information and has an attribute "dims" (1, 2, or 3).
            It is assumed that the mesh object also stores the full 1D grids as self.mesh_x, etc.,
            and the number of grid points as self.Nx (and self.Ny, self.Nz where appropriate),
            and that the bounds are unpacked as self.x0, self.x1, etc.
        other_subspaces : list[int], optional
            List of integers. Will be used to determine how to reshape the obtained wavefunctions.
        """
        self.H = H
        self.mesh = mesh  # the mesh object containing grid information
        self.energies = None
        self.wavefunctions = None
        self.other_subspaces_dims = other_subspaces_dims
        if self.other_subspaces_dims: # if other_subspaces_dims is NOT None
            if any(not isinstance(dim, int) for dim in self.other_subspaces_dims):
                raise ValueError("other_subspaces_dims must be a list of integers")

    def solve(self, k):
        """
        Diagonalizes the Hamiltonian and stores the energies and reshaped wavefunctions.

        Parameters
        ----------
        k : int
            The number of eigenvalues/eigenfunctions to compute.
            
        The method computes:
            energies, eigenvectors = eigsh(H, k=k, which="SM")
        and then reshapes each eigenvector according to the mesh dimensionality. The interior
        grid in each coordinate has length (N-2) due to Dirichlet boundary conditions.
        Finally, it orders the eigenpairs in increasing order of energy.
        """
        # Diagonalize the sparse Hamiltonian.
        energies, eigvecs = eigsh(self.H, k=k, which="SM")
        # Order the eigenvalues and eigenvectors in increasing order.
        order = np.argsort(energies)
        energies = energies[order]
        eigvecs = eigvecs[:, order]
        
        # Determine the shape to which each eigenvector must be reshaped.
        if self.mesh.dims == 1:
            if not self.other_subspaces_dims: # if other_subspaces_dims is None
                new_shape = (self.mesh.Nx - 2,)
            else:
                new_shape = tuple(self.other_subspaces_dims) + (self.mesh.Nx - 2,)
        elif self.mesh.dims == 2:
            if not self.other_subspaces_dims: # if other_subspaces_dims is None
                new_shape = (self.mesh.Nx - 2, self.mesh.Ny - 2)
            else:
                 new_shape = tuple(self.other_subspaces_dims) + (self.mesh.Nx - 2, self.mesh.Ny - 2)
        elif self.mesh.dims == 3:
            if not self.other_subspaces_dims: # if other_subspaces_dims is None
                new_shape = (self.mesh.Nx - 2, self.mesh.Ny - 2, self.mesh.Nz - 2)
            else:
                new_shape = tuple(self.other_subspaces_dims) + (self.mesh.Nx - 2, self.mesh.Ny - 2, self.mesh.Nz - 2)
        else:
            raise ValueError("Mesh dimensionality must be 1, 2, or 3.")
            
        # Reshape each eigenvector (note: eigvecs.T has shape (k, N_interior)).
        wavefunctions = [vec.reshape(new_shape) for vec in eigvecs.T]
        
        if self.other_subspaces_dims: # if other_subspaces_dims is NOT None
            to_be_traced_over = len(self.other_subspaces_dims)
            for _ in range(to_be_traced_over):
                wavefunctions = [np.sum(vec, axis=0) for vec in wavefunctions] # Tracing over non-orbital degrees of freedom        
            
        densities = [np.abs(wavefunction)**2 for wavefunction in wavefunctions]
        # Normalization
        wavefunctions = [wavefunction / np.sqrt(np.sum(density)) for wavefunction,density in zip(wavefunctions, densities)]
        densities = [np.abs(wavefunction)**2 for wavefunction in wavefunctions]
        
        self.energies = energies
        self.wavefunctions = wavefunctions
        self.densities = densities
        
        return energies, wavefunctions

    def plot(self, 
             wf_index:int =0, 
             c:str = "b", 
             cmap: str = "RdYlBu_r",
             n_surfaces:int = 20):
        """
        Plots the probability density of the specified eigenfunction.
        
        Parameters
        ----------
        wf_index : int, optional
            Index of the eigenfunction (wavefunction) to plot (default is 0).
        c : str, optional
            color for 1d line plots
        cmap : str, optional
            color for 2d and 3d plots
        n_surfaces : int, optional
            number of isosurfaces to plot in 3d plots
            
        For 1D, a line plot is created; for 2D, an image/contour plot is generated;
        for 3D, the method uses Plotly with marching cubes to display multiple isosurfaces.
        """
        # Make sure the eigenfunctions have been computed.
        if self.densities is None:
            raise RuntimeError("No eigenfunctions found. Call .solve(k) first.")
            
        # Compute the probability density.
        density = self.densities[wf_index]
        dims = self.mesh.dims

        if dims == 1:
            # For 1D, use the interior grid points.
            x = self.mesh.mesh_x[1:-1]
            plt.figure()
            plt.plot(x, density, '-', c = c)
            plt.xlabel("x")
            plt.ylabel("Probability Density")
            plt.title(f"Wavefunction {wf_index} Probability Density (1D)")
            plt.show()

        elif dims == 2:
            # Extract interior grid points
            x = self.mesh.mesh_x[1:-1]
            y = self.mesh.mesh_y[1:-1]
            
            # Determine the larger dimension
            nx, ny = density.shape
            if nx > ny:
                # Pad the smaller dimension (y) with zeros
                pad_size = (nx - ny) // 2
                density_padded = np.pad(density, ((0, 0), (pad_size, pad_size)), mode="constant")
                extent = [x[0], x[-1], y[0] - pad_size * (y[1] - y[0]), y[-1] + pad_size * (y[1] - y[0])]
            elif ny > nx:
                # Pad the smaller dimension (x) with zeros
                pad_size = (ny - nx) // 2
                density_padded = np.pad(density, ((pad_size, pad_size), (0, 0)), mode="constant")
                extent = [x[0] - pad_size * (x[1] - x[0]), x[-1] + pad_size * (x[1] - x[0]), y[0], y[-1]]
            else:
                # Already square
                density_padded = density
                extent = [x[0], x[-1], y[0], y[-1]]
        
            # Plot heatmap with equal aspect ratio
            plt.figure()
            plt.imshow(density_padded.T, extent=extent, origin="lower", aspect="equal", cmap=cmap)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Wavefunction {wf_index} Probability Density (2D)")
            plt.colorbar(label="Probability Density")
            plt.show()

        elif dims == 3:
            # For 3D, use Plotly and marching cubes.
            import plotly.graph_objects as go
            from skimage import measure

            # Retrieve grid bounds.
            x0, x1 = self.mesh.x0, self.mesh.x1
            y0, y1 = self.mesh.y0, self.mesh.y1
            z0, z1 = self.mesh.z0, self.mesh.z1
            
            # Determine the number of points along each axis (full grid).
            # The interior grid (where the eigenfunction is defined) has size (N-2).
            dimx = self.mesh.Nx - 2
            dimy = self.mesh.Ny - 2
            dimz = self.mesh.Nz - 2
            
            # Create coordinate arrays for the interior points.
            # (Assuming uniform spacing for the purpose of marching cubes.)
            xvals_full = np.linspace(x0, x1, self.mesh.Nx)
            yvals_full = np.linspace(y0, y1, self.mesh.Ny)
            zvals_full = np.linspace(z0, z1, self.mesh.Nz)
            xvals = xvals_full[1:-1]
            yvals = yvals_full[1:-1]
            zvals = zvals_full[1:-1]
            
            # Determine global min and max of the density for normalization.
            global_min = np.min(density)
            global_max = np.max(density)
            
            # Define several isosurface levels.
            iso_levels = np.linspace(global_min * 1.001, global_max * 0.999, n_surfaces)
            
            # Approximate spacing between grid points (assumes uniformity).
            spacing = (xvals[1] - xvals[0], yvals[1] - yvals[0], zvals[1] - zvals[0])
            
            # Initialize the Plotly figure.
            fig = go.Figure()
            
            # Loop over iso-levels to add isosurfaces.
            for level in iso_levels:
                verts, faces, _, _ = measure.marching_cubes(density, level, spacing=spacing)
                
                # Normalize and rescale vertices to match the physical grid range.
                verts[:, 0] = verts[:, 0] * (x1 - x0) / (xvals[-1] - xvals[0]) + x0
                verts[:, 1] = verts[:, 1] * (y1 - y0) / (yvals[-1] - yvals[0]) + y0
                verts[:, 2] = verts[:, 2] * (z1 - z0) / (zvals[-1] - zvals[0]) + z0
                
                # Normalize color intensity based on the current iso-level.
                norm_intensity = (np.full(verts.shape[0], level) - global_min) / (global_max - global_min)
                
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    intensity=norm_intensity,
                    colorscale=cmap,
                    opacity=0.5,
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Density Level", tickvals=[0, 1],
                                  ticktext=[f"{global_min:.2e}", f"{global_max:.2e}"])
                ))
            
            # Update layout with visible axes.
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="X", 
                        tickvals=xvals[::5], 
                        ticktext=[f"{x:.2f}" for x in xvals[::5]],
                        range=(min(x0, y0, z0), max(x1, y1, z1)),
                        tickfont=dict(color='white'),
                    ),
                    yaxis=dict(
                        title="Y", 
                        tickvals=yvals[::5], 
                        ticktext=[f"{y:.2f}" for y in yvals[::5]],
                        range=(min(x0, y0, z0), max(x1, y1, z1)),
                        tickfont=dict(color='white'),
                    ),
                    zaxis=dict(
                        title="Z", 
                        tickvals=zvals[::2], 
                        ticktext=[f"{z:.2f}" for z in zvals[::2]],
                        range=(min(x0, y0, z0), max(x1, y1, z1)),
                        tickfont=dict(color='white'),
                    ),
                    bgcolor='rgb(0, 0, 0)'
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            
            fig.show()

        else:
            raise ValueError("Mesh dimensionality must be 1, 2, or 3.")




