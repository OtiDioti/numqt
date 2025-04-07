import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import kron
from .utils import iterative_kron
from scipy.sparse import diags
from skimage import measure
import plotly.graph_objects as go
from scipy.sparse.linalg import eigsh

class Mesh:
    def __init__(self, 
                 dims: int,
                 xbounds: tuple,
                 ybounds: tuple = None,
                 zbounds: tuple = None,
                 initial_points: int = 20,
                 dx_max: float = None,
                 dy_max: float = None,
                 dz_max: float = None,
                 dx_func: callable = None,
                 dy_func: callable = None,
                 dz_func: callable = None,
                 max_iter: int = 10):
        """
        Constructs an adaptive structured grid in 1D, 2D, or 3D.

        The grid is initially defined uniformly (with `initial_points` along each axis)
        over the specified intervals. Then, the grid is refined adaptively:
          - For each cell (an interval in 1D, rectangle in 2D, or box in 3D), its center is computed.
          - The allowed spacing at that cell center is determined by a function. For example,
            for the x-direction: allowed_dx = dx_func(x_center). If no dx_func is provided,
            the allowed spacing is constant and equal to dx_max.
          - If the cell’s size along a given axis exceeds the allowed spacing, the midpoint
            along that axis is added.
          - This process is repeated until no cell is refined (or until max_iter iterations).

        Parameters
        ----------
        dims : int
            Dimensionality of the grid. Must be 1, 2, or 3.
        xbounds, ybounds, zbounds : tuple
            Endpoints of the domain in x, y, and z. (Unused bounds are ignored for dims < 3.)
        initial_points : int, optional
            Number of grid points to start with along each axis (including endpoints).
        dx_max, dy_max, dz_max : float
            Global maximum allowed spacing along x, y, and z (used as default if no function is provided).
        dx_func, dy_func, dz_func : callable, optional
            Functions that return the allowed spacing at a given coordinate.
            For example, dx_func(x) should return the allowed spacing at position x.
            If not provided, a constant function returning dx_max (etc.) is used.
        max_iter : int, optional
            Maximum number of refinement iterations.
        """
        # Validate the dimensionality.
        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3.")

        # Validate domain bounds.
        if xbounds is None:
            raise ValueError("xbounds must be provided.")
        if dims >= 2 and ybounds is None:
            raise ValueError("For dims>=2, ybounds must be provided.")
        if dims == 3 and zbounds is None:
            raise ValueError("For dims==3, zbounds must be provided.")

        # Validate essential grid parameters.
        if dx_max is None:
            raise ValueError("dx_max must be provided for the x-direction.")
        if dims >= 2 and dy_max is None:
            raise ValueError("dy_max must be provided for dims>=2.")
        if dims == 3 and dz_max is None:
            raise ValueError("dz_max must be provided for dims==3.")

        # If no allowed-spacing function is provided, use a constant function.
        if dx_func is None:
            dx_func = lambda x: dx_max
        if dims >= 2:
            if dy_func is None:
                dy_func = lambda y: dy_max
        if dims == 3:
            if dz_func is None:
                dz_func = lambda z: dz_max

        # Save basic parameters.
        self.dims = dims
        self.xbounds = xbounds
        self.ybounds = ybounds if dims >= 2 else None
        self.zbounds = zbounds if dims == 3 else None

        self.initial_points = initial_points
        self.dx_max = dx_max
        self.dy_max = dy_max if dims >= 2 else None
        self.dz_max = dz_max if dims == 3 else None

        self.dx_func = dx_func
        self.dy_func = dy_func if dims >= 2 else None
        self.dz_func = dz_func if dims == 3 else None

        self.max_iter = max_iter

        # Unpack domain bounds for convenience.
        self.x0, self.x1 = xbounds
        if dims >= 2:
            self.y0, self.y1 = ybounds
        if dims == 3:
            self.z0, self.z1 = zbounds

        # Generate the grid based on the dimensionality.
        self.generate_mesh()

        self.Nx = len(self.mesh_x)
        self.Ny = len(self.mesh_y) if dims >= 2 else 0
        self.Nz = len(self.mesh_z) if dims == 3 else 0

    def generate_mesh(self):
        """Creates the adaptive grid based on the specified dimensionality and allowed-spacing functions."""
        if self.dims == 1:
            # 1D grid only along x.
            self.mesh_x = np.linspace(self.x0, self.x1, self.initial_points)
            for it in range(self.max_iter):
                cell_refined = False
                new_x = []
                for i in range(len(self.mesh_x) - 1):
                    x0 = self.mesh_x[i]
                    x1 = self.mesh_x[i+1]
                    x_center = (x0 + x1) / 2.0
                    allowed_dx = self.dx_func(x_center)
                    if (x1 - x0) > allowed_dx:
                        new_x.append(x_center)
                        cell_refined = True
                if not cell_refined:
                    break
                self.mesh_x = np.sort(np.unique(np.concatenate([self.mesh_x, new_x])))

        elif self.dims == 2:
            # 2D grid along x and y.
            self.mesh_x = np.linspace(self.x0, self.x1, self.initial_points)
            self.mesh_y = np.linspace(self.y0, self.y1, self.initial_points)
            for it in range(self.max_iter):
                cell_refined = False
                new_x = []
                new_y = []
                for i in range(len(self.mesh_x) - 1):
                    for j in range(len(self.mesh_y) - 1):
                        x0 = self.mesh_x[i]
                        x1 = self.mesh_x[i+1]
                        y0 = self.mesh_y[j]
                        y1 = self.mesh_y[j+1]
                        x_center = (x0 + x1) / 2.0
                        y_center = (y0 + y1) / 2.0
                        allowed_dx = self.dx_func(x_center)
                        allowed_dy = self.dy_func(y_center)
                        if (x1 - x0) > allowed_dx:
                            new_x.append(x_center)
                            cell_refined = True
                        if (y1 - y0) > allowed_dy:
                            new_y.append(y_center)
                            cell_refined = True
                if not cell_refined:
                    break
                self.mesh_x = np.sort(np.unique(np.concatenate([self.mesh_x, new_x])))
                self.mesh_y = np.sort(np.unique(np.concatenate([self.mesh_y, new_y])))

        elif self.dims == 3:
            # 3D grid along x, y, and z.
            self.mesh_x = np.linspace(self.x0, self.x1, self.initial_points)
            self.mesh_y = np.linspace(self.y0, self.y1, self.initial_points)
            self.mesh_z = np.linspace(self.z0, self.z1, self.initial_points)
            for it in range(self.max_iter):
                cell_refined = False
                new_x = []
                new_y = []
                new_z = []
                for i in range(len(self.mesh_x) - 1):
                    for j in range(len(self.mesh_y) - 1):
                        for k in range(len(self.mesh_z) - 1):
                            x0 = self.mesh_x[i]
                            x1 = self.mesh_x[i+1]
                            y0 = self.mesh_y[j]
                            y1 = self.mesh_y[j+1]
                            z0 = self.mesh_z[k]
                            z1 = self.mesh_z[k+1]
                            x_center = (x0 + x1) / 2.0
                            y_center = (y0 + y1) / 2.0
                            z_center = (z0 + z1) / 2.0
                            allowed_dx = self.dx_func(x_center)
                            allowed_dy = self.dy_func(y_center)
                            allowed_dz = self.dz_func(z_center)
                            if (x1 - x0) > allowed_dx:
                                new_x.append(x_center)
                                cell_refined = True
                            if (y1 - y0) > allowed_dy:
                                new_y.append(y_center)
                                cell_refined = True
                            if (z1 - z0) > allowed_dz:
                                new_z.append(z_center)
                                cell_refined = True
                if not cell_refined:
                    break
                self.mesh_x = np.sort(np.unique(np.concatenate([self.mesh_x, new_x])))
                self.mesh_y = np.sort(np.unique(np.concatenate([self.mesh_y, new_y])))
                self.mesh_z = np.sort(np.unique(np.concatenate([self.mesh_z, new_z])))
                
    
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

class canonic_ops():
    def __init__(self, mesh, additional_subspaces=None, hbar=1):
        """
        Constructs the discretized quantum mechanical operators for position and momentum,
        adapting the generated operators to the dimensionality of mesh3D.
        
        If mesh3D.dims == 1 then only x, p, x², and p² are generated.
        For dims == 2 and 3 the corresponding y and z operators are also built.
        
        Parameters
        ----------
        mesh : object
            A mesh object that must have an attribute "dims" (equal to 1, 2, or 3) and a 
            get_grids() method returning the 1D grid arrays. For dims==1, get_grids() should return
            (mesh_x,), for dims==2 (mesh_x, mesh_y), and for dims==3 (mesh_x, mesh_y, mesh_z).
        additional_subspaces : list, optional
            If provided, these extra Hilbert spaces will be incorporated via tensor products.
        hbar : float, optional
            Planck's constant (defaults to 1).
        """
        self.hbar = hbar
        self.dim = mesh.dims

        # Get the grid(s) from the mesh.
        if self.dim == 1:
            mesh_x = mesh.get_grids()  # single 1D array
        elif self.dim == 2:
            mesh_x, mesh_y = mesh.get_grids()
        elif self.dim == 3:
            mesh_x, mesh_y, mesh_z = mesh.get_grids()

        # Construct identity matrices for the interior nodes.
        # (Assume that the operators act on interior points only, i.e. mesh[1:-1].)
        I_x = sparse.eye(len(mesh_x) - 2)
        if self.dim >= 2:
            I_y = sparse.eye(len(mesh_y) - 2)
        if self.dim == 3:
            I_z = sparse.eye(len(mesh_z) - 2)

        # Set up the "additional subspaces" lists for tensor products.
        if additional_subspaces is None:
            if self.dim == 1:
                additional_x = [0]
            elif self.dim == 2:
                additional_x = [0, I_y]
                additional_y = [1, I_x]
            elif self.dim == 3:
                additional_x = [0, I_y, I_z]
                additional_y = [1, I_x, I_z]
                additional_z = [2, I_x, I_y]
        else:
            if self.dim == 1:
                additional_x = [len(additional_subspaces) + 0] + [identity for identity in additional_subspaces]
            elif self.dim == 2:
                additional_x = [len(additional_subspaces) + 0] + [identity for identity in additional_subspaces] + [I_y]
                additional_y = [len(additional_subspaces) + 1] + [identity for identity in additional_subspaces] + [I_x]
            elif self.dim == 3:
                additional_x = [len(additional_subspaces) + 0] + [identity for identity in additional_subspaces] + [I_y, I_z]
                additional_y = [len(additional_subspaces) + 1] + [identity for identity in additional_subspaces] + [I_x, I_z]
                additional_z = [len(additional_subspaces) + 2] + [identity for identity in additional_subspaces] + [I_x, I_y]

        # Generate only the operators needed for the given dimensionality.
        if self.dim == 1:
            self.px    = self.p(mesh_x, additional_x)
            self.px2   = self.p2(mesh_x, additional_x)
            self.x_op  = self.x(mesh_x, additional_x)
            self.x2_op = self.x2(mesh_x, additional_x)
        elif self.dim == 2:
            self.px    = self.p(mesh_x, additional_x)
            self.py    = self.p(mesh_y, additional_y)
            self.px2   = self.p2(mesh_x, additional_x)
            self.py2   = self.p2(mesh_y, additional_y)
            self.x_op  = self.x(mesh_x, additional_x)
            self.y_op  = self.x(mesh_y, additional_y)
            self.x2_op = self.x2(mesh_x, additional_x)
            self.y2_op = self.x2(mesh_y, additional_y)
        elif self.dim == 3:
            self.px    = self.p(mesh_x, additional_x)
            self.py    = self.p(mesh_y, additional_y)
            self.pz    = self.p(mesh_z, additional_z)
            self.px2   = self.p2(mesh_x, additional_x)
            self.py2   = self.p2(mesh_y, additional_y)
            self.pz2   = self.p2(mesh_z, additional_z)
            self.x_op  = self.x(mesh_x, additional_x)
            self.y_op  = self.x(mesh_y, additional_y)
            self.z_op  = self.x(mesh_z, additional_z)
            self.x2_op = self.x2(mesh_x, additional_x)
            self.y2_op = self.x2(mesh_y, additional_y)
            self.z2_op = self.x2(mesh_z, additional_z)

    def get_ops(self):
        """
        Returns the operators in a dictionary.
        
        For dims==1, only the x-operators are returned.
        For dims==2, x- and y-operators are returned.
        For dims==3, x-, y-, and z-operators are returned.
        """
        if self.dim == 1:
            outpt = {
                "p":  self.px,
                "p2": self.px2,
                "x":  self.x_op,
                "x2": self.x2_op
            }
        elif self.dim == 2:
            outpt = {
                "p":  [self.px, self.py],
                "p2": [self.px2, self.py2],
                "x":  [self.x_op, self.y_op],
                "x2": [self.x2_op, self.y2_op]
            }
        elif self.dim == 3:
            outpt = {
                "p":  [self.px, self.py, self.pz],
                "p2": [self.px2, self.py2, self.pz2],
                "x":  [self.x_op, self.y_op, self.z_op],
                "x2": [self.x2_op, self.y2_op, self.z2_op]
            }
        return outpt

    def p(self, mesh, other_subspaces=None):
        """
        Constructs the discretized momentum operator (p = -i*hbar*d/dx) using a 7-point
        central difference stencil approximation (6th-order accurate) on a uniform grid.
        
        The 7-point stencil for approximating the first derivative is given by:
        
            f'(x) ≈ [ f(x-3h) - 9f(x-2h) + 45f(x-h) - 45f(x+h) + 9f(x+2h) - f(x+3h) ] / (60h)
        
        Multiplying the derivative by -i*hbar gives the momentum operator.
        
        Parameters
        ----------
        mesh : ndarray
            1D mesh array encoding information for the given dimension's grid.
        other_subspaces : list, optional
            If provided, a list whose first element is an integer specifying the position
            at which the momentum operator should be inserted in a tensor (Kronecker) product.
            The remaining elements should be the identity operators for the other Hilbert spaces.
            For example, to have p act on the first factor in a two-space product, you might use:
                other_subspaces = [0, I_second]
        
        Returns
        -------
        scipy.sparse.spmatrix or numpy.ndarray
            An (N × N) matrix representation of the momentum operator.
        """
        dx = np.abs(mesh[i] - mesh[i - 1])
        #N = N_full - 2
        # Offsets for the 7-point stencil (skip the central point since its coefficient is zero)
        offsets = np.array([-3, -2, -1, 1, 2, 3])
        # Stencil coefficients corresponding to the 7-point central difference formula:
        # [1, -9, 45, -45, 9, -1]
        coeffs = np.array([1, -9, 45, -45, 9, -1], dtype=complex)
        
        # The prefactor includes the momentum operator constants (-i * hbar) and the derivative factor (1/(60h))
        prefactor = -1j * self.hbar / (60 * dx)
        
        # Build the diagonals for the sparse matrix using the stencil coefficients
        diagonals = [
            prefactor * coeff * np.ones(self.N - abs(offset), dtype=complex)
            for offset, coeff in zip(offsets, coeffs)
        ]
        
        # Construct the sparse matrix with the given diagonals and offsets
        p_mat = sparse.diags(diagonals, offsets, shape=(self.N, self.N), format='csr')
        
        # If other subspaces are provided, insert the momentum operator into the tensor product
        if other_subspaces is not None:
            pos = other_subspaces[0]
            kron_list = other_subspaces[1:]
            kron_list.insert(pos, p_mat)
            return iterative_kron(kron_list)
        else:
            return p_mat

    def p2(self, mesh, other_subspaces=None):
        """
        Constructs the discretized squared momentum operator (p² = -ħ² d²/dx²) using a 7-point 
        central difference stencil (6th-order accurate) on a uniform grid.
    
        The 7-point stencil for approximating the second derivative is given by:
    
            f''(x) ≈ [ -f(x-3h) + 12f(x-2h) - 39f(x-h) + 56f(x) - 39f(x+h) + 12f(x+2h) - f(x+3h) ] / (180 h²)
    
        Multiplying the second derivative by -ħ² yields the squared momentum operator.
    
        Parameters
        ----------
        mesh : ndarray
            1D mesh array encoding information for the given dimension's grid.
        other_subspaces : list, optional
            If provided, a list whose first element is an integer specifying the position 
            at which the squared momentum operator should be inserted in a tensor (Kronecker) product.
            The remaining elements should be the identity operators for the other Hilbert spaces.
            For example, to have p² act on the first factor in a two-space product, you might use:
                other_subspaces = [0, I_second]
    
        Returns
        -------
        scipy.sparse.spmatrix or numpy.ndarray
            An (N × N) matrix representation of the squared momentum operator.
        """
        dx = np.abs(mesh[i] - mesh[i - 1])
        #N = N_full - 2
        # Offsets for the 7-point stencil
        offsets = np.array([-3, -2, -1, 0, 1, 2, 3])
        # Corresponding coefficients for the 7-point finite difference formula
        coeffs = np.array([-1, 12, -39, 56, -39, 12, -1], dtype=float)
        
        # The prefactor includes the -ħ² factor and the derivative discretization factor
        prefactor = -self.hbar**2 / (180 * dx**2)
        
        # Build the diagonals using the stencil coefficients
        diagonals = [
            prefactor * coeff * np.ones(self.N - abs(offset), dtype=float)
            for offset, coeff in zip(offsets, coeffs)
        ]
        
        # Construct the sparse matrix with the given diagonals and offsets
        p2_mat = sparse.diags(diagonals, offsets, shape=(self.N, self.N), format='csr')
        
        # If other subspaces are provided, insert the squared momentum operator into the tensor product
        if other_subspaces is not None:
            pos = other_subspaces[0]
            kron_list = other_subspaces[1:]
            kron_list.insert(pos, p2_mat)
            return iterative_kron(kron_list)
        else:
            return p2_mat

    
   
    
    
    
    def x(self, mesh, other_subspaces=None):
        """
        Constructs the discretized position operator on a nonuniform mesh.
        The operator is multiplicative; we define it on the interior nodes (mesh[1:-1]).
        Parameters
        ----------
        mesh : ndarray
            1D mesh array encoding information for the given dimension's grid.
        other_subspaces : list, optional
            If provided, a list whose first element is an integer specifying the position 
            at which the momentum operator should be inserted in a tensor (Kronecker) product.
            The remaining elements should be the identity operators for the other Hilbert spaces.
            For example, to have p act on the first factor in a two-space product, you might use:
                other_subspaces = [0, I_second]
        """
        x_interior = mesh[1:-1]
        X = diags(x_interior, 0, format='csr')
        
        if other_subspaces is not None:
            pos = other_subspaces[0]
            kron_list = other_subspaces[1:]
            kron_list.insert(pos, X)
            return iterative_kron(kron_list)
        else:
            return X
    
    def x2(self, mesh, other_subspaces=None):
        """
        Constructs the discretized x^2 operator on a nonuniform mesh.
        Parameters
        ----------
        mesh : ndarray
            1D mesh array encoding information for the given dimension's grid.
        other_subspaces : list, optional
            If provided, a list whose first element is an integer specifying the position 
            at which the momentum operator should be inserted in a tensor (Kronecker) product.
            The remaining elements should be the identity operators for the other Hilbert spaces.
            For example, to have p act on the first factor in a two-space product, you might use:
                other_subspaces = [0, I_second]
        """
        x_interior = mesh[1:-1]
        X2 = diags(x_interior**2, 0, format='csr')
        if other_subspaces is not None:
            pos = other_subspaces[0]
            kron_list = other_subspaces[1:]
            kron_list.insert(pos, X2)
            return iterative_kron(kron_list)
        else:
            return X2









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




