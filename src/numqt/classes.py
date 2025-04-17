import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import diags, eye, kron, coo_matrix, isspmatrix_csr, isspmatrix_csc
from .utils import iterative_kron
from skimage import measure
import plotly.graph_objects as go
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
from scipy.sparse import lil_matrix, csr_matrix
from concurrent.futures import ThreadPoolExecutor

        
class Mesh:
    def __init__(self, 
                 dims: int,
                 xbounds: tuple,
                 ybounds: tuple = None,
                 zbounds: tuple = None,
                 nx: int = 50,
                 ny: int = None,
                 nz: int = None):
        if dims not in [1, 2, 3]:
            raise ValueError("dims must be 1, 2, or 3.")

        if xbounds is None:
            raise ValueError("xbounds must be provided.")
        if dims >= 2 and ybounds is None:
            raise ValueError("For dims>=2, ybounds must be provided.")
        if dims == 3 and zbounds is None:
            raise ValueError("For dims==3, zbounds must be provided.")

        if ny is None:
            ny = nx
        if nz is None:
            nz = nx

        self.dims = dims
        self.xbounds = xbounds
        self.ybounds = ybounds if dims >= 2 else None
        self.zbounds = zbounds if dims == 3 else None

        self.x0, self.x1 = xbounds
        if dims >= 2:
            self.y0, self.y1 = ybounds
        if dims == 3:
            self.z0, self.z1 = zbounds

        self.generate_mesh(nx, ny, nz)

        self.Nx = len(self.mesh_x)
        self.Ny = len(self.mesh_y) if dims >= 2 else 0
        self.Nz = len(self.mesh_z) if dims == 3 else 0

    def generate_mesh(self, nx, ny, nz):
        self.mesh_x = np.linspace(self.x0, self.x1, nx)
        self.mesh_y = np.linspace(self.y0, self.y1, ny) if self.dims >= 2 else None
        self.mesh_z = np.linspace(self.z0, self.z1, nz) if self.dims == 3 else None

    @property
    def total_meshes(self):
        if self.dims == 1:
            return self.mesh_x
        elif self.dims == 2:
            return np.meshgrid(self.mesh_x, self.mesh_y, indexing='ij')
        elif self.dims == 3:
            return np.meshgrid(self.mesh_x, self.mesh_y, self.mesh_z, indexing='ij')

    def get_grids(self):
        if self.dims == 1:
            return self.mesh_x
        elif self.dims == 2:
            return self.mesh_x, self.mesh_y
        elif self.dims == 3:
            return self.mesh_x, self.mesh_y, self.mesh_z

    def get_spacings(self):
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

    def visualize_grid(self, marker_size=10, elev=30, azim=45, alpha=0.1):
        if self.dims == 1:
            plt.figure(figsize=(8, 2))
            plt.scatter(self.total_meshes, np.zeros_like(self.total_meshes), s=marker_size, c='b', alpha=alpha)
            plt.xlabel('x')
            plt.title("1D Grid Visualization")
            plt.yticks([])
            plt.grid(True)
            plt.show()

        elif self.dims == 2:
            X, Y = self.total_meshes
            plt.figure(figsize=(8, 6))
            plt.scatter(X.flatten(), Y.flatten(), s=marker_size, c='b', alpha=alpha)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("2D Grid Visualization")
            plt.grid(True)
            plt.show()

        elif self.dims == 3:
            X, Y, Z = self.total_meshes
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), s=marker_size, c='b', alpha=alpha)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=elev, azim=azim)
            plt.title("3D Grid Visualization")
            plt.show()

    def visualize_slice(self, slice_axis: str, slice_value: float, marker_size=20, alpha=0.7, c="b"):
        """
        Visualizes a slice of the grid along a specified axis at the given value.
        
        Parameters
        ----------
        slice_axis : str
            The axis along which to slice ('x', 'y', or 'z').
        slice_value : float
            The value at which to slice the grid.
        marker_size : int, optional
            Size of the markers in the plot.
        alpha : float, optional
            Transparency of the markers.
        c : str, optional
            Color of the markers.
        """
        slice_axis = slice_axis.lower()
        
        if self.dims == 1:
            raise ValueError("Slicing is not applicable for 1D grids.")
        
        elif self.dims == 2:
            if slice_axis == 'x':
                idx = np.argmin(np.abs(self.mesh_x - slice_value))
                X = np.full_like(self.mesh_y, self.mesh_x[idx])
                Y = self.mesh_y
            elif slice_axis == 'y':
                idx = np.argmin(np.abs(self.mesh_y - slice_value))
                Y = np.full_like(self.mesh_x, self.mesh_y[idx])
                X = self.mesh_x
            else:
                raise ValueError("For 2D grids, slice_axis must be 'x' or 'y'.")
            
            plt.figure(figsize=(8, 4))
            plt.scatter(X, Y, s=marker_size, c=c, alpha=alpha)
            plt.xlabel('x' if slice_axis == 'y' else 'y')
            plt.ylabel('y' if slice_axis == 'y' else 'x')
            plt.title(f"Slice at {slice_axis} = {self.mesh_x[idx]:.2f}" if slice_axis == 'x' 
                      else f"Slice at {slice_axis} = {self.mesh_y[idx]:.2f}")
            plt.grid(True)
            plt.show()
        
        elif self.dims == 3:
            if slice_axis == 'x':
                idx = np.argmin(np.abs(self.mesh_x - slice_value))
                actual_value = self.mesh_x[idx]
                # Create a meshgrid of y and z values
                Y, Z = np.meshgrid(self.mesh_y, self.mesh_z)
                # Prepare points for the slice
                points_y = Y.flatten()
                points_z = Z.flatten()
                # All x values are the same
                points_x = np.full_like(points_y, actual_value)
                # Plot y and z coordinates
                plt.figure(figsize=(8, 6))
                plt.scatter(points_y, points_z, s=marker_size, c=c, alpha=alpha)
                plt.xlabel('y')
                plt.ylabel('z')
                
            elif slice_axis == 'y':
                idx = np.argmin(np.abs(self.mesh_y - slice_value))
                actual_value = self.mesh_y[idx]
                # Create a meshgrid of x and z values
                X, Z = np.meshgrid(self.mesh_x, self.mesh_z)
                # Prepare points for the slice
                points_x = X.flatten()
                points_z = Z.flatten()
                # All y values are the same
                points_y = np.full_like(points_x, actual_value)
                # Plot x and z coordinates
                plt.figure(figsize=(8, 6))
                plt.scatter(points_x, points_z, s=marker_size, c=c, alpha=alpha)
                plt.xlabel('x')
                plt.ylabel('z')
                
            elif slice_axis == 'z':
                idx = np.argmin(np.abs(self.mesh_z - slice_value))
                actual_value = self.mesh_z[idx]
                # Create a meshgrid of x and y values
                X, Y = np.meshgrid(self.mesh_x, self.mesh_y)
                # Prepare points for the slice
                points_x = X.flatten()
                points_y = Y.flatten()
                # All z values are the same
                points_z = np.full_like(points_x, actual_value)
                # Plot x and y coordinates
                plt.figure(figsize=(8, 6))
                plt.scatter(points_x, points_y, s=marker_size, c=c, alpha=alpha)
                plt.xlabel('x')
                plt.ylabel('y')
                
            else:
                raise ValueError("For 3D grids, slice_axis must be one of 'x', 'y', or 'z'.")
            
            plt.title(f"Slice at {slice_axis} = {actual_value:.2f}")
            plt.grid(True)
            plt.show()


# ----------------------------------------------------------
# ----------------------------------------------------------
# Canonic Operators
# ----------------------------------------------------------
# ----------------------------------------------------------

class canonic_ops:
    def __init__(self, mesh, ops_to_compute, basis=None, additional_subspaces=None, limit_divisions=None, hbar=1, threshold=1e-15):
        """
        Constructs the discretized quantum mechanical operators for position and momentum.
        
        Parameters
        ----------
        mesh : object
            Must have an attribute 'dims' (1, 2, or 3) and a get_grids() method returning the grid(s).
        ops_to_compute : list of str
            List specifying which operators to compute: "p", "p2", "x", "x2".
        basis : tuple or dict or None, optional
            If provided, integration routines (with a function and basis size) are used.
        additional_subspaces : list, optional
            List such as [position, I_second, I_third, …] used to embed the operator into a tensor product space.
        limit_divisions : int, optional
            Parameter for the integration routine.
        hbar : float, optional
            Planck's constant.
        threshold : float, optional
            Elements with absolute value below this threshold are set to zero when using integration.
            This improves sparsity and numerical stability. Default is 1e-15.
        """
        self.hbar = hbar
        self.ops_to_compute = ops_to_compute
        self.limit_divisions = limit_divisions
        self.basis = basis
        self.additional_subspaces = additional_subspaces
        self.threshold = threshold
        self.dim = mesh.dims

        # Get the mesh grids (only the interior points are used)
        if self.dim == 1:
            self.mesh_x = mesh.get_grids()  # 1D array
            self.Nx_grid = len(self.mesh_x) - 2
        elif self.dim == 2:
            self.mesh_x, self.mesh_y = mesh.get_grids()
            self.Nx_grid = len(self.mesh_x) - 2
            self.Ny_grid = len(self.mesh_y) - 2
        elif self.dim == 3:
            self.mesh_x, self.mesh_y, self.mesh_z = mesh.get_grids()
            self.Nx_grid = len(self.mesh_x) - 2
            self.Ny_grid = len(self.mesh_y) - 2
            self.Nz_grid = len(self.mesh_z) - 2

        # If no basis is provided, use finite-difference approximations.
        if self.basis is None:
            # Precompute identity matrices for embedding.
            I_x = eye(self.Nx_grid, format='csr')
            if self.dim >= 2:
                I_y = eye(self.Ny_grid, format='csr')
            if self.dim == 3:
                I_z = eye(self.Nz_grid, format='csr')
            # Build additional subspace lists.
            if self.additional_subspaces is None:
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
                # Use provided additional subspaces.
                if self.dim == 1:
                    additional_x = [len(self.additional_subspaces)] + self.additional_subspaces
                elif self.dim == 2:
                    additional_x = [len(self.additional_subspaces)] + self.additional_subspaces + [I_y]
                    additional_y = [len(self.additional_subspaces)+1] + self.additional_subspaces + [I_x]
                elif self.dim == 3:
                    additional_x = [len(self.additional_subspaces)] + self.additional_subspaces + [I_y, I_z]
                    additional_y = [len(self.additional_subspaces)+1] + self.additional_subspaces + [I_x, I_z]
                    additional_z = [len(self.additional_subspaces)+2] + self.additional_subspaces + [I_x, I_y]
            # Compute using finite-difference methods.
            if self.dim == 1:
                if "p" in self.ops_to_compute:
                    self.px = self.p(self.mesh_x, additional_x)
                if "p2" in self.ops_to_compute:
                    self.px2 = self.p2(self.mesh_x, additional_x)
                if "x" in self.ops_to_compute:
                    self.x_op = self.x(self.mesh_x, additional_x)
                if "x2" in self.ops_to_compute:
                    self.x2_op = self.x2(self.mesh_x, additional_x)
            elif self.dim == 2:
                if "p" in self.ops_to_compute:
                    self.px = self.p(self.mesh_x, additional_x)
                    self.py = self.p(self.mesh_y, additional_y)
                if "p2" in self.ops_to_compute:
                    self.px2 = self.p2(self.mesh_x, additional_x)
                    self.py2 = self.p2(self.mesh_y, additional_y)
                if "x" in self.ops_to_compute:
                    self.x_op = self.x(self.mesh_x, additional_x)
                    self.y_op = self.x(self.mesh_y, additional_y)
                if "x2" in self.ops_to_compute:
                    self.x2_op = self.x2(self.mesh_x, additional_x)
                    self.y2_op = self.x2(self.mesh_y, additional_y)
            elif self.dim == 3:
                if "p" in self.ops_to_compute:
                    self.px = self.p(self.mesh_x, additional_x)
                    self.py = self.p(self.mesh_y, additional_y)
                    self.pz = self.p(self.mesh_z, additional_z)
                if "p2" in self.ops_to_compute:
                    self.px2 = self.p2(self.mesh_x, additional_x)
                    self.py2 = self.p2(self.mesh_y, additional_y)
                    self.pz2 = self.p2(self.mesh_z, additional_z)
                if "x" in self.ops_to_compute:
                    self.x_op = self.x(self.mesh_x, additional_x)
                    self.y_op = self.x(self.mesh_y, additional_y)
                    self.z_op = self.x(self.mesh_z, additional_z)
                if "x2" in self.ops_to_compute:
                    self.x2_op = self.x2(self.mesh_x, additional_x)
                    self.y2_op = self.x2(self.mesh_y, additional_y)
                    self.z2_op = self.x2(self.mesh_z, additional_z)

        # Otherwise (if a basis is provided), use integration routines.
        else:
            # Helper: return the basis tuple for a given direction.
            def get_basis(direction):
                if isinstance(self.basis, dict):
                    return self.basis.get(direction)
                else:
                    return self.basis
            if self.dim == 1:
                basis_x = get_basis("x")
                fn_x, N_basis_x = basis_x
                
                # If additional_subspaces is provided, build additional_x.
                if self.additional_subspaces is None:
                    additional_x = None
                else:
                    additional_x = [len(self.additional_subspaces)] + self.additional_subspaces

                if "p" in self.ops_to_compute:
                    self.px = self.compute_p_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                if "p2" in self.ops_to_compute:
                    self.px2 = self.compute_p2_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                if "x" in self.ops_to_compute:
                    self.x_op = self.compute_x_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                if "x2" in self.ops_to_compute:
                    self.x2_op = self.compute_x2_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
            elif self.dim == 2:
                basis_x = get_basis("x")
                basis_y = get_basis("y") or basis_x
                fn_x, N_basis_x = basis_x
                fn_y, N_basis_y = basis_y
                # Build identities for embedding.
                I_x = eye(N_basis_x, format='csr')
                I_y = eye(N_basis_y, format='csr')
                # Construct additional subspace lists if provided.
                if self.additional_subspaces is None:
                    additional_x = [0, I_y]
                    additional_y = [1, I_x]
                else:
                    additional_x = [len(self.additional_subspaces)] + self.additional_subspaces + [I_y]
                    additional_y = [len(self.additional_subspaces)+1] + self.additional_subspaces + [I_x]
                if "p" in self.ops_to_compute:
                    p_x = self.compute_p_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    p_y = self.compute_p_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    self.px = p_x
                    self.py = p_y
                if "p2" in self.ops_to_compute:
                    p2_x = self.compute_p2_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    p2_y = self.compute_p2_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    self.px2 = p2_x
                    self.py2 = p2_y
                if "x" in self.ops_to_compute:
                    x_op = self.compute_x_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    y_op = self.compute_x_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    self.x_op = x_op
                    self.y_op = y_op
                if "x2" in self.ops_to_compute:
                    x2_op = self.compute_x2_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    y2_op = self.compute_x2_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    self.x2_op = x2_op
                    self.y2_op = y2_op
            elif self.dim == 3:
                basis_x = get_basis("x")
                basis_y = get_basis("y") or basis_x
                basis_z = get_basis("z") or basis_x
                fn_x, N_basis_x = basis_x
                fn_y, N_basis_y = basis_y
                fn_z, N_basis_z = basis_z
                I_x = eye(N_basis_x, format='csr')
                I_y = eye(N_basis_y, format='csr')
                I_z = eye(N_basis_z, format='csr')
                if self.additional_subspaces is None:
                    additional_x = [0, I_y, I_z]
                    additional_y = [1, I_x, I_z]
                    additional_z = [2, I_x, I_y]
                else:
                    additional_x = [len(self.additional_subspaces)] + self.additional_subspaces + [I_y, I_z]
                    additional_y = [len(self.additional_subspaces)+1] + self.additional_subspaces + [I_x, I_z]
                    additional_z = [len(self.additional_subspaces)+2] + self.additional_subspaces + [I_x, I_y]
                if "p" in self.ops_to_compute:
                    p_x = self.compute_p_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    p_y = self.compute_p_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    p_z = self.compute_p_matrix(fn_z, self.mesh_z, N_basis_z, additional=additional_z)
                    self.px = p_x
                    self.py = p_y
                    self.pz = p_z
                if "p2" in self.ops_to_compute:
                    p2_x = self.compute_p2_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    p2_y = self.compute_p2_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    p2_z = self.compute_p2_matrix(fn_z, self.mesh_z, N_basis_z, additional=additional_z)
                    self.px2 = p2_x
                    self.py2 = p2_y
                    self.pz2 = p2_z
                if "x" in self.ops_to_compute:
                    x_op = self.compute_x_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    y_op = self.compute_x_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    z_op = self.compute_x_matrix(fn_z, self.mesh_z, N_basis_z, additional=additional_z)
                    self.x_op = x_op
                    self.y_op = y_op
                    self.z_op = z_op
                if "x2" in self.ops_to_compute:
                    x2_op = self.compute_x2_matrix(fn_x, self.mesh_x, N_basis_x, additional=additional_x)
                    y2_op = self.compute_x2_matrix(fn_y, self.mesh_y, N_basis_y, additional=additional_y)
                    z2_op = self.compute_x2_matrix(fn_z, self.mesh_z, N_basis_z, additional=additional_z)
                    self.x2_op = x2_op
                    self.y2_op = y2_op
                    self.z2_op = z2_op
            else:
                raise ValueError("Unsupported dimension: {}".format(self.dim))
    
    def get_ops(self):
        """
        Returns the constructed operators in a dictionary.
        """
        ops = {}
        if self.dim == 1:
            if hasattr(self, "px"):
                ops["p"] = self.px
            if hasattr(self, "px2"):
                ops["p2"] = self.px2
            if hasattr(self, "x_op"):
                ops["x"] = self.x_op
            if hasattr(self, "x2_op"):
                ops["x2"] = self.x2_op
        elif self.dim == 2:
            if hasattr(self, "px") and hasattr(self, "py"):
                ops["p"] = [self.px, self.py]
            if hasattr(self, "px2") and hasattr(self, "py2"):
                ops["p2"] = [self.px2, self.py2]
            if hasattr(self, "x_op") and hasattr(self, "y_op"):
                ops["x"] = [self.x_op, self.y_op]
            if hasattr(self, "x2_op") and hasattr(self, "y2_op"):
                ops["x2"] = [self.x2_op, self.y2_op]
        elif self.dim == 3:
            if hasattr(self, "px") and hasattr(self, "py") and hasattr(self, "pz"):
                ops["p"] = [self.px, self.py, self.pz]
            if hasattr(self, "px2") and hasattr(self, "py2") and hasattr(self, "pz2"):
                ops["p2"] = [self.px2, self.py2, self.pz2]
            if hasattr(self, "x_op") and hasattr(self, "y_op") and hasattr(self, "z_op"):
                ops["x"] = [self.x_op, self.y_op, self.z_op]
            if hasattr(self, "x2_op") and hasattr(self, "y2_op") and hasattr(self, "z2_op"):
                ops["x2"] = [self.x2_op, self.y2_op, self.z2_op]
        return ops

    # --- Helper: Embed operator via tensor products if additional subspaces are provided ---
    def embed_operator(self, op, additional):
        """
        If additional is provided, embed the operator op into the tensor product space.
        The additional list is expected to be of the form:
            [pos, sub1, sub2, ...]
        where pos indicates the insertion index.
        """
        
        if additional is None:
            return op
        pos = additional[0]
        kron_list = additional[1:].copy()
        kron_list.insert(pos, op)
        return iterative_kron(kron_list)

    # --- Finite-difference based operators (when basis is None) ---
    def p(self, mesh, additional=None):
        dx = np.abs(mesh[1] - mesh[0])
        N = len(mesh) - 2
        offsets = [-1, 1]
        coeffs = [-1, 1]
        prefactor = -1j * self.hbar / (2 * dx)
        diagonals = [prefactor * coeff * np.ones(N - abs(offset), dtype=complex)
                     for offset, coeff in zip(offsets, coeffs)]
        p_mat = diags(diagonals, offsets, shape=(N, N), format='csr')
        return self.embed_operator(p_mat, additional)
    
    def p2(self, mesh, additional=None):
        dx = np.abs(mesh[1] - mesh[0])
        N = len(mesh) - 2
        offsets = [-1, 0, 1]
        coeffs = [1, -2, 1]
        prefactor = -self.hbar**2 / (dx**2)
        diagonals = [prefactor * coeff * np.ones(N - abs(offset), dtype=float)
                     for offset, coeff in zip(offsets, coeffs)]
        p2_mat = diags(diagonals, offsets, shape=(N, N), format='csr')
        return self.embed_operator(p2_mat, additional)
    
    def x(self, mesh, additional=None):
        x_interior = mesh[1:-1]
        X = diags(x_interior, 0, format='csr')
        return self.embed_operator(X, additional)
    
    def x2(self, mesh, additional=None):
        x_interior = mesh[1:-1]
        X2 = diags(x_interior**2, 0, format='csr')
        return self.embed_operator(X2, additional)
    
    # --- Integration-based operators (when a basis is provided) ---
    def _compute_operator_matrix(self, fn, mesh, N, integrand_func, scale, additional=None):
        """
        Optimized function to compute an operator matrix element via numerical integration.
        Only computes (m,n) for m<=n and then fills in the Hermitian lower-triangle.
        Elements with absolute value below self.threshold are set to zero to improve sparsity.
        
        Parameters
        ----------
        fn : callable
            Basis function generator that returns (psi, dpsi, d2psi)
        mesh : ndarray
            1D grid array
        N : int
            Dimension of the basis
        integrand_func : callable
            Function taking (psi_m, psi_n, dpsi_n, d2psi_n, x) that defines the integrand
        scale : float or complex
            Multiplicative prefactor (e.g. -1j*self.hbar for p)
        additional : list, optional
            Information for embedding the operator in a larger space
        """
        x_min = mesh[0]
        x_max = mesh[-1]
        
        # Use Gaussian quadrature for faster integration
        # Number of points can be adjusted based on required accuracy
        from scipy.integrate import fixed_quad
        n_points = max(int(3 * N), 50)  # Adaptive number of points
        
        
        # Pre-compute basis functions at quadrature points
        from scipy.special import roots_legendre
        x_quad, weights = roots_legendre(n_points)
        
        # Transform from [-1, 1] to [x_min, x_max]
        x_quad = 0.5 * (x_max - x_min) * x_quad + 0.5 * (x_max + x_min)
        weights = 0.5 * (x_max - x_min) * weights
        
        # Pre-compute all basis functions and derivatives at quadrature points
        basis_values = np.zeros((N, n_points, 3), dtype=complex)  # [basis_idx, quad_point, (psi, dpsi, d2psi)]
        
        # Batch computation of basis functions
        for i in range(N):
            for j, x in enumerate(x_quad):
                psi, dpsi, d2psi = fn(i, x)
                basis_values[i, j, 0] = psi
                basis_values[i, j, 1] = dpsi
                basis_values[i, j, 2] = d2psi
        
        # Initialize sparse matrix in COO format for efficiency
        from scipy.sparse import coo_matrix
        rows = []
        cols = []
        data = []
        
        # Compute only upper triangle (m <= n)
        for m in range(N):
            for n in range(m, N):
                # Vectorized integration using pre-computed values
                integrand_vals = np.array([
                    integrand_func(
                        basis_values[m, i, 0],  # psi_m
                        basis_values[n, i, 0],  # psi_n
                        basis_values[n, i, 1],  # dpsi_n
                        basis_values[n, i, 2],  # d2psi_n
                        x_quad[i]               # x
                    ) for i in range(n_points)
                ])
                
                # Compute integral using quadrature
                integral = np.sum(integrand_vals * weights)
                scaled_val = scale * integral
                
                # Apply threshold to improve sparsity - NEW CODE
                if np.abs(scaled_val) >= self.threshold:
                    # Add to sparse matrix data
                    rows.append(m)
                    cols.append(n)
                    data.append(scaled_val)
                    
                    # Fill in lower triangle for Hermitian operators
                    if m != n and np.abs(np.conj(scaled_val)) >= self.threshold:
                        rows.append(n)
                        cols.append(m)
                        data.append(np.conj(scaled_val))
        
        # Create sparse matrix
        mat = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
        
        # Embed if needed
        return self.embed_operator(mat, additional)
    
    def compute_p_matrix(self, fn, mesh, N, additional=None):
        """
        Compute momentum operator matrix elements: <m|p|n> = -i ħ ∫ psi_m*(x) d/dx psi_n(x) dx
        """
        integrand = lambda psi_m, psi_n, dpsi_n, d2psi_n, x: np.conj(psi_m) * dpsi_n
        return self._compute_operator_matrix(fn, mesh, N, integrand, -1j * self.hbar, additional)
    
    def compute_p2_matrix(self, fn, mesh, N, additional=None):
        """
        Compute squared momentum operator matrix elements: <m|p^2|n> = -ħ² ∫ psi_m*(x) d²/dx² psi_n(x) dx
        """
        integrand = lambda psi_m, psi_n, dpsi_n, d2psi_n, x: np.conj(psi_m) * d2psi_n
        return self._compute_operator_matrix(fn, mesh, N, integrand, -self.hbar**2, additional)
    
    def compute_x_matrix(self, fn, mesh, N, additional=None):
        """
        Compute position operator matrix elements: <m|x|n> = ∫ psi_m*(x) x psi_n(x) dx
        """
        integrand = lambda psi_m, psi_n, dpsi_n, d2psi_n, x: np.conj(psi_m) * x * psi_n
        return self._compute_operator_matrix(fn, mesh, N, integrand, 1.0, additional)
    
    def compute_x2_matrix(self, fn, mesh, N, additional=None):
        """
        Compute squared position operator matrix elements: <m|x^2|n> = ∫ psi_m*(x) x^2 psi_n(x) dx
        """
        integrand = lambda psi_m, psi_n, dpsi_n, d2psi_n, x: np.conj(psi_m) * (x**2) * psi_n
        return self._compute_operator_matrix(fn, mesh, N, integrand, 1.0, additional)









# ----------------------------------------------------------
# ----------------------------------------------------------
# Hamiltonian
# ----------------------------------------------------------
# ----------------------------------------------------------

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix, lil_matrix

class Hamiltonian:
    def __init__(self, H, mesh, basis=None, other_subspaces_dims=None, verbose=False, threshold=1e-10):
        """
        Parameters
        ----------
        H : sparse matrix
            The Hamiltonian operator defined on the interior points.
        mesh : object
            A mesh object that contains grid information. It should have:
                - dims: (1, 2, or 3)
                - full 1D grid arrays (mesh.mesh_x, mesh.mesh_y, etc.)
                - number of grid points (mesh.Nx, mesh.Ny, …)
                - domain bounds (mesh.x0, mesh.x1, etc.)
            It is assumed that the operators act on the interior nodes (mesh.mesh_x[1:-1], etc.).
        basis : tuple or dict or None
            Basis information. Either a single tuple (fn, N) to be used for all directions,
            or a dict with keys "x", "y", (and "z") for higher dimensions.
            If provided, the eigenvectors from solve() are expansion coefficients in that basis.
        other_subspaces_dims : list[int], optional
            Extra tensor–product dimensions (these will be summed over later).
        threshold : float, optional
            Elements with absolute value below this threshold are set to zero.
            This improves sparsity and numerical stability. Default is 1e-10.
        """
        self.threshold = threshold
        self.H = H
        self.H = self._threshold_sparse_matrix(self.H)
        self.mesh = mesh
        self.basis = basis
        self.other_subspaces_dims = other_subspaces_dims
        self.energies = None
        self.wavefunctions = None
        self.densities = None
        self.verbose = verbose

        if self.other_subspaces_dims:
            if any(not isinstance(dim, int) for dim in self.other_subspaces_dims):
                raise ValueError("other_subspaces_dims must be a list of integers")
        
        # Flag to note whether basis matrices have been cached.
        self._cached_basis = False
        
    def _threshold_sparse_matrix(self, matrix):
        """
        Sets to zero all elements of a scipy sparse matrix whose absolute value
        is below the specified threshold. Optimized for performance.
            
        Parameters:
        -----------
        matrix : scipy.sparse.spmatrix
            The input sparse matrix to threshold
            
        Returns:
        --------
        scipy.sparse.spmatrix
            A sparse matrix with small elements removed
        """
        # For CSR and CSC formats, we can operate directly on the data
        if isspmatrix_csr(matrix) or isspmatrix_csc(matrix):
            # Make a copy to avoid modifying the original
            result = matrix.copy()
            
            # Create a mask for values to keep
            mask = np.abs(result.data) >= self.threshold
            
            # Apply the mask to the data and indices
            result.data = result.data[mask]
            result.indices = result.indices[mask]
            
            # Reconstruct the indptr array
            if isspmatrix_csr(matrix):
                # Process each row
                new_indptr = np.zeros_like(result.indptr)
                for i in range(matrix.shape[0]):
                    # Count elements in this row that pass the threshold
                    row_start, row_end = matrix.indptr[i], matrix.indptr[i+1]
                    row_mask = mask[row_start:row_end]
                    new_indptr[i+1] = new_indptr[i] + np.count_nonzero(row_mask)
                result.indptr = new_indptr
            else:  # CSC format
                # Process each column
                new_indptr = np.zeros_like(result.indptr)
                for i in range(matrix.shape[1]):
                    # Count elements in this column that pass the threshold
                    col_start, col_end = matrix.indptr[i], matrix.indptr[i+1]
                    col_mask = mask[col_start:col_end]
                    new_indptr[i+1] = new_indptr[i] + np.count_nonzero(col_mask)
                result.indptr = new_indptr
                
            return result
        else:
            # For other formats, convert to CSR first, then apply threshold
            # IMPORTANT: Don't call this function recursively!
            csr_matrix = matrix.tocsr()
            
            # Now apply the threshold directly to the CSR matrix
            result_csr = csr_matrix.copy()
            mask = np.abs(result_csr.data) >= self.threshold
            
            # Apply the mask to the data and indices
            result_csr.data = result_csr.data[mask]
            result_csr.indices = result_csr.indices[mask]
            
            # Reconstruct the indptr array
            new_indptr = np.zeros_like(result_csr.indptr)
            for i in range(csr_matrix.shape[0]):
                row_start, row_end = csr_matrix.indptr[i], csr_matrix.indptr[i+1]
                row_mask = mask[row_start:row_end]
                new_indptr[i+1] = new_indptr[i] + np.count_nonzero(row_mask)
            result_csr.indptr = new_indptr
            
            # Convert back to the original format
            return result_csr.asformat(matrix.format)

    def _cache_basis(self):
        """
        Precomputes and caches the basis matrices (for physical-space evaluation)
        based on the provided basis functions. Once computed, these matrices are reused
        in the reconstruction of eigenfunctions.
        """
        if self.basis is None:
            return

        dims = self.mesh.dims
        if dims == 1:
            # Get the 1D basis tuple (or dictionary entry).
            b_tuple = self.basis if not isinstance(self.basis, dict) else self.basis["x"]
            fn, N_basis = b_tuple
            # Evaluate basis functions on the interior 1D grid.
            self._basis_x = np.stack([fn(n, self.mesh.mesh_x[1:-1])[0] for n in range(N_basis)], axis=0)
        elif dims == 2:
            if isinstance(self.basis, dict):
                fnx, N_basis_x = self.basis["x"]
                fny, N_basis_y = self.basis["y"]
            else:
                fnx, N_basis_x = self.basis
                fny, N_basis_y = self.basis
            self._basis_x = np.stack([fnx(n, self.mesh.mesh_x[1:-1])[0] for n in range(N_basis_x)], axis=0)
            self._basis_y = np.stack([fny(n, self.mesh.mesh_y[1:-1])[0] for n in range(N_basis_y)], axis=0)
        elif dims == 3:
            if isinstance(self.basis, dict):
                fnx, N_basis_x = self.basis["x"]
                fny, N_basis_y = self.basis["y"]
                fnz, N_basis_z = self.basis["z"]
            else:
                fnx, N_basis_x = self.basis
                fny, N_basis_y = self.basis
                fnz, N_basis_z = self.basis
            self._basis_x = np.stack([fnx(n, self.mesh.mesh_x[1:-1])[0] for n in range(N_basis_x)], axis=0)
            self._basis_y = np.stack([fny(n, self.mesh.mesh_y[1:-1])[0] for n in range(N_basis_y)], axis=0)
            self._basis_z = np.stack([fnz(n, self.mesh.mesh_z[1:-1])[0] for n in range(N_basis_z)], axis=0)
        self._cached_basis = True

    def solve(self, k):
        """
        Diagonalizes the Hamiltonian and reconstructs the physical-space wavefunctions
        (if basis is provided, the eigenvectors are expansion coefficients which are transformed).
        The reconstruction is performed in a batched and optimized manner.
        
        Parameters
        ----------
        k : int
            Number of eigenpairs to compute.
        
        Returns
        -------
        energies : ndarray
            Sorted eigenvalues.
        wavefunctions : list of ndarrays
            The reconstructed and normalized physical-space wavefunctions.
        """
        energies, eigvecs = eigsh(self.H, k=k, which="SM")
        if self.verbose:
                print("Solved eigenvalue problem.")

        # Sort eigenpairs.
        order = np.argsort(energies)
        energies = energies[order]
        eigvecs = eigvecs[:, order]

        # Determine raw reshaping shape.
        dims = self.mesh.dims
        if dims == 1:
            if self.basis is None:
                dim_primary = self.mesh.Nx - 2
            else:
                b = self.basis if not isinstance(self.basis, dict) else self.basis["x"]
                _, dim_primary = b
            raw_shape = (dim_primary,) if not self.other_subspaces_dims else tuple(self.other_subspaces_dims) + (dim_primary,)
        elif dims == 2:
            if self.basis is None:
                dim_x = self.mesh.Nx - 2
                dim_y = self.mesh.Ny - 2
            else:
                if isinstance(self.basis, dict):
                    b_x = self.basis["x"]
                    b_y = self.basis["y"]
                else:
                    b_x = b_y = self.basis
                _, dim_x = b_x
                _, dim_y = b_y
            raw_shape = (dim_x, dim_y) if not self.other_subspaces_dims else tuple(self.other_subspaces_dims) + (dim_x, dim_y)
        elif dims == 3:
            if self.basis is None:
                dim_x = self.mesh.Nx - 2
                dim_y = self.mesh.Ny - 2
                dim_z = self.mesh.Nz - 2
            else:
                if isinstance(self.basis, dict):
                    b_x = self.basis["x"]
                    b_y = self.basis["y"]
                    b_z = self.basis["z"]
                else:
                    b_x = b_y = b_z = self.basis
                _, dim_x = b_x
                _, dim_y = b_y
                _, dim_z = b_z
            raw_shape = (dim_x, dim_y, dim_z) if not self.other_subspaces_dims else tuple(self.other_subspaces_dims) + (dim_x, dim_y, dim_z)
        else:
            raise ValueError("Mesh dimensionality must be 1, 2, or 3.")

        # Reshape the eigenvector coefficients into their raw form.
        raw_wavefunctions = [vec.reshape(raw_shape) for vec in eigvecs.T]

        # If extra subspaces exist, sum over them.
        if self.other_subspaces_dims:
            for _ in range(len(self.other_subspaces_dims)):
                raw_wavefunctions = [np.sum(vec, axis=0) for vec in raw_wavefunctions]

        # If a basis is provided, cache the basis matrices once.
        if self.basis is not None and not self._cached_basis:
            self._cache_basis()

        # Batch process eigenfunctions.
        if self.basis is None:
            # Already in physical space.
            reconstructed = np.array(raw_wavefunctions)
        else:
            if dims == 1:
                # In the 1D case:
                # raw_wavefunctions have shape (k, N_basis) and self._basis_x has shape (N_basis, M),
                # where M = len(mesh.mesh_x[1:-1]). We perform a single matrix multiplication.
                raw_batch = np.stack(raw_wavefunctions, axis=0)  # shape: (k, N_basis)
                # NOTE: Remove .T so the inner dimensions align.
                reconstructed = np.dot(raw_batch, self._basis_x)   # yields (k, M)
            elif dims == 2:
                # For 2D case:
                # raw_wavefunctions: (k, N_basis_x, N_basis_y)
                raw_batch = np.stack(raw_wavefunctions, axis=0)  # shape: (k, N_basis_x, N_basis_y)
                # First, contract along y direction.
                # self._basis_y has shape (N_basis_y, len(mesh.mesh_y[1:-1]))
                temp = np.matmul(raw_batch, self._basis_y)  # shape: (k, N_basis_x, len(mesh.mesh_y[1:-1]))
                # Next, contract along x direction:
                # self._basis_x has shape (N_basis_x, len(mesh.mesh_x[1:-1]))
                reconstructed = np.einsum('ij,kjl->kil', self._basis_x.T, temp)
                # Now, reconstructed has shape: (k, len(mesh.mesh_x[1:-1]), len(mesh.mesh_y[1:-1]))
            elif dims == 3:
                # For 3D case:
                # raw_wavefunctions: (k, N_basis_x, N_basis_y, N_basis_z)
                raw_batch = np.stack(raw_wavefunctions, axis=0)  # shape: (k, n_x, n_y, n_z)
                # First contract along z:
                t1 = np.tensordot(raw_batch, self._basis_z, axes=([3],[0]))  # shape: (k, n_x, n_y, M_z)
                # Then contract along y:
                t2 = np.tensordot(t1, self._basis_y, axes=([2],[0]))         # shape: (k, n_x, M_y, M_z)
                # Finally contract along x:
                psi_temp = np.tensordot(t2, self._basis_x.T, axes=([1],[1]))       # shape: (k, M_y, M_z, M_x)
                # Rearrange axes to get shape: (k, M_x, M_y, M_z)
                reconstructed = np.moveaxis(psi_temp, -1, 1)
            else:
                raise ValueError("Unsupported dimension.")

        # Compute densities and normalization factors using the mesh spacing.
        if dims == 1:
            dx = self.mesh.mesh_x[1] - self.mesh.mesh_x[0]
            densities = [np.abs(wf)**2 for wf in reconstructed]
            norm_factors = [np.sum(dens) * dx for dens in densities]
        elif dims == 2:
            dx = self.mesh.mesh_x[1] - self.mesh.mesh_x[0]
            dy = self.mesh.mesh_y[1] - self.mesh.mesh_y[0]
            densities = [np.abs(wf)**2 for wf in reconstructed]
            norm_factors = [np.sum(dens) * dx * dy for dens in densities]
        elif dims == 3:
            dx = self.mesh.mesh_x[1] - self.mesh.mesh_x[0]
            dy = self.mesh.mesh_y[1] - self.mesh.mesh_y[0]
            dz = self.mesh.mesh_z[1] - self.mesh.mesh_z[0]
            densities = [np.abs(wf)**2 for wf in reconstructed]
            norm_factors = [np.sum(dens) * dx * dy * dz for dens in densities]

        # Normalize the wavefunctions.
        normalized_wavefunctions = [wf / np.sqrt(norm) for wf, norm in zip(reconstructed, norm_factors)]
        normalized_densities = [np.abs(wf)**2 for wf in normalized_wavefunctions]

        # Save results.
        self.energies = energies
        self.wavefunctions = normalized_wavefunctions
        self.densities = normalized_densities

        return energies, normalized_wavefunctions



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
            iso_levels = np.linspace(global_min * 1.1, global_max * 0.9, n_surfaces)
            
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






