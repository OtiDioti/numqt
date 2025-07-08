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
from scipy.sparse import csr_matrix, lil_matrix, isspmatrix_csr, isspmatrix_csc
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt

class Hamiltonian:
    def __init__(self, H=None, H_time_indep=None, H_time_dep=None, time_dep_args=None, 
                 mesh=None, basis=None, other_subspaces_dims=None, verbose=False, threshold=1e-10):
        """
        Parameters
        ----------
        H : sparse matrix, optional
            The time-independent Hamiltonian operator (for backward compatibility).
        H_time_indep : sparse matrix, optional
            The time-independent part of the Hamiltonian.
        H_time_dep : callable, optional
            Function representing time-dependent part: H_time_dep(amplitude, frequency, t)
        time_dep_args : dict, optional
            Dictionary with keys:
            - "amplitudes": scalar or array of amplitude values
            - "frequencies": scalar or array of frequency values  
            - "periods": scalar or array of period values (must match frequencies dimensionality)
            Example: {"amplitudes": [0.1, 0.5, 1.0], "frequencies": [1.0, 2.0], "periods": [6.28, 3.14]}
        mesh : object, optional
            A mesh object that contains grid information.
        basis : tuple or dict or None
            Basis information.
        other_subspaces_dims : list[int], optional
            Extra tensor–product dimensions.
        threshold : float, optional
            Elements with absolute value below this threshold are set to zero.
        """
        self.threshold = threshold
        
        # Handle backward compatibility
        if H is not None and H_time_indep is None:
            self.H = H
            self.H_time_indep = H
            self.H_time_dep = None
            self.time_dep_args = None
        else:
            self.H = H_time_indep
            self.H_time_indep = H_time_indep
            self.H_time_dep = H_time_dep
            self.time_dep_args = time_dep_args
            
        # Validate time-dependent setup
        if self.H_time_dep is not None:
            if self.time_dep_args is None:
                raise ValueError("time_dep_args must be provided for time-dependent Hamiltonians")
            
            if isinstance(self.time_dep_args, (tuple, list)):
                # Backward compatibility with old tuple format
                if len(self.time_dep_args) < 3:
                    raise ValueError("tuple format time_dep_args must contain at least (param1, param2, T)")
                print("Warning: Using deprecated tuple format for time_dep_args. Consider using dictionary format.")
            elif isinstance(self.time_dep_args, dict):
                # New dictionary format
                required_keys = {"amplitudes", "frequencies", "periods"}
                if not required_keys.issubset(self.time_dep_args.keys()):
                    raise ValueError(f"time_dep_args dict must contain keys: {required_keys}")
                
                # Validate frequencies and periods have matching dimensionality
                freq_array = np.asarray(self.time_dep_args["frequencies"])
                period_array = np.asarray(self.time_dep_args["periods"])
                
                if freq_array.shape != period_array.shape:
                    raise ValueError(f"frequencies shape {freq_array.shape} must match periods shape {period_array.shape}")
            else:
                raise ValueError("time_dep_args must be a tuple (deprecated) or dictionary")
            
            if not callable(self.H_time_dep):
                raise ValueError("H_time_dep must be a callable function")
        
        if self.H is not None:
            self.H = self._threshold_sparse_matrix(self.H)
        if self.H_time_indep is not None:
            self.H_time_indep = self._threshold_sparse_matrix(self.H_time_indep)
            
        self.mesh = mesh
        self.basis = basis
        self.other_subspaces_dims = other_subspaces_dims
        self.energies = None
        self.wavefunctions = None
        self.densities = None
        self.quasi_energies = None
        self.verbose = verbose

        if self.other_subspaces_dims:
            if any(not isinstance(dim, int) for dim in self.other_subspaces_dims):
                raise ValueError("other_subspaces_dims must be a list of integers")
        
        # Flag to note whether basis matrices have been cached.
        self._cached_basis = False
        if self.basis and (not self.mesh): # if basis is given but mesh is not
            raise ValueError("A Mesh Object must be provided when using basis")
        
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

    def solve_quasi_energies(self, Nt=600, hbar=1, rescale_omega=None, brillouin_omega=None):
        """
        Compute quasi-energies using Floquet theory for time-dependent Hamiltonians.
        Supports array parameters to compute quasi-energies for multiple parameter values.
        
        Parameters
        ----------
        Nt : int, optional
            Number of time steps for time evolution (default: 600)
        hbar : float, optional
            Reduced Planck constant (default: 1)
        rescale_omega : float, optional
            If provided, rescale quasi-energies by 1/(hbar * rescale_omega)
            If None, uses brillouin_omega or inferred omega
        brillouin_omega : float, optional
            Frequency to use for Brillouin zone mapping [-ħω/2, +ħω/2]
            If None, uses the frequency from each calculation
            
        Returns
        -------
        quasi_energies : ndarray
            Computed quasi-energies. Shape depends on input parameters:
            - If all parameters are scalars: shape (N_states,)
            - If amplitudes is array of length M, frequencies scalar: shape (M, N_states)
            - If amplitudes scalar, frequencies array of length N: shape (N, N_states)  
            - If amplitudes array (M) and frequencies array (N): shape (M, N, N_states)
        """
        if self.H_time_dep is None:
            raise ValueError("Time-dependent Hamiltonian not specified. Use solve() for time-independent problems.")
        
        # Handle both old tuple format and new dictionary format
        if isinstance(self.time_dep_args, dict):
            return self._solve_quasi_energies_dict_format(Nt, hbar, rescale_omega, brillouin_omega)
        else:
            # Backward compatibility with old tuple format
            return self._solve_quasi_energies_tuple_format(Nt, hbar, rescale_omega, brillouin_omega)
    
    def _solve_quasi_energies_dict_format(self, Nt=600, hbar=1, rescale_omega=None, brillouin_omega=None):
        """Handle the new dictionary format for time_dep_args."""
        
        # Extract parameters
        amplitudes_raw = self.time_dep_args["amplitudes"]
        frequencies_raw = self.time_dep_args["frequencies"] 
        periods_raw = self.time_dep_args["periods"]
        
        # Convert to arrays and check dimensions
        amplitudes = np.asarray(amplitudes_raw)
        frequencies = np.asarray(frequencies_raw)
        periods = np.asarray(periods_raw)
        
        # Ensure frequencies and periods have same shape
        if frequencies.shape != periods.shape:
            raise ValueError(f"frequencies shape {frequencies.shape} must match periods shape {periods.shape}")
        
        # Determine output shape
        amp_is_array = amplitudes.ndim > 0
        freq_is_array = frequencies.ndim > 0
        
        if not amp_is_array and not freq_is_array:
            # Both scalars - single calculation
            return self._solve_quasi_energies_single_dict(
                amplitudes.item(), frequencies.item(), periods.item(), 
                Nt, hbar, rescale_omega, brillouin_omega
            )
        
        # Get matrix dimensions
        N = self.H_time_indep.shape[0]
        
        # Determine output shape and create parameter combinations
        if amp_is_array and not freq_is_array:
            # Amplitudes array, frequencies scalar
            output_shape = (len(amplitudes), N)
            if self.verbose:
                print(f"Computing quasi-energies: {len(amplitudes)} amplitudes × 1 frequency")
                print(f"Output shape: {output_shape}")
            
            quasi_energies_result = np.zeros(output_shape)
            
            freq_scalar = frequencies.item()
            period_scalar = periods.item()
            
            for i, amp in enumerate(amplitudes):
                if self.verbose and (i + 1) % max(1, len(amplitudes) // 10) == 0:
                    print(f"Progress: {i + 1}/{len(amplitudes)} ({100*(i+1)/len(amplitudes):.1f}%)")
                
                qe = self._solve_quasi_energies_single_dict(
                    amp, freq_scalar, period_scalar, Nt, hbar, rescale_omega, brillouin_omega
                )
                quasi_energies_result[i] = qe
                
        elif not amp_is_array and freq_is_array:
            # Amplitudes scalar, frequencies array
            output_shape = (len(frequencies), N)
            if self.verbose:
                print(f"Computing quasi-energies: 1 amplitude × {len(frequencies)} frequencies")
                print(f"Output shape: {output_shape}")
            
            quasi_energies_result = np.zeros(output_shape)
            
            amp_scalar = amplitudes.item()
            
            for j, (freq, period) in enumerate(zip(frequencies, periods)):
                if self.verbose and (j + 1) % max(1, len(frequencies) // 10) == 0:
                    print(f"Progress: {j + 1}/{len(frequencies)} ({100*(j+1)/len(frequencies):.1f}%)")
                
                qe = self._solve_quasi_energies_single_dict(
                    amp_scalar, freq, period, Nt, hbar, rescale_omega, brillouin_omega
                )
                quasi_energies_result[j] = qe
                
        else:
            # Both arrays
            output_shape = (len(amplitudes), len(frequencies), N)
            if self.verbose:
                print(f"Computing quasi-energies: {len(amplitudes)} amplitudes × {len(frequencies)} frequencies")
                print(f"Output shape: {output_shape}")
            
            quasi_energies_result = np.zeros(output_shape)
            
            total_combinations = len(amplitudes) * len(frequencies)
            combination_count = 0
            
            for i, amp in enumerate(amplitudes):
                for j, (freq, period) in enumerate(zip(frequencies, periods)):
                    combination_count += 1
                    if self.verbose and combination_count % max(1, total_combinations // 10) == 0:
                        print(f"Progress: {combination_count}/{total_combinations} ({100*combination_count/total_combinations:.1f}%)")
                    
                    qe = self._solve_quasi_energies_single_dict(
                        amp, freq, period, Nt, hbar, rescale_omega, brillouin_omega
                    )
                    quasi_energies_result[i, j] = qe
        
        # Store results
        self.quasi_energies = quasi_energies_result
        
        if self.verbose:
            print(f"Computed quasi-energies with shape: {quasi_energies_result.shape}")
        
        return quasi_energies_result
    
    def _solve_quasi_energies_single_dict(self, amplitude, frequency, period, Nt=600, hbar=1, 
                                         rescale_omega=None, brillouin_omega=None):
        """Compute quasi-energies for a single set of parameters using dict format."""
        
        # Set up time array
        dt = period / (Nt - 1)
        t_array = dt * np.arange(Nt)
        
        # Get matrix dimensions
        N = self.H_time_indep.shape[0]
        
        # Convert to dense arrays for efficient matrix operations
        H0_dense = self.H_time_indep.toarray()
        identity_N = np.eye(N, dtype=complex)
        
        # Pre-allocate temporary array
        H_temp = np.zeros((N, N), dtype=complex)
        
        # Time evolution
        U = identity_N.copy()
        dt_over_hbar = dt / hbar
        
        for k in range(Nt):
            t = t_array[k]
            # Compute time-dependent Hamiltonian: H_time_dep(amplitude, frequency, t)
            H_td = self.H_time_dep(amplitude, frequency, t)
            
            # Total Hamiltonian = H0 + H_time_dep(t)
            if hasattr(H_td, 'toarray'):
                H_td_dense = H_td.toarray()
            else:
                H_td_dense = H_td
                
            H_temp[:] = H0_dense + H_td_dense
            U = expm(-1j * dt_over_hbar * H_temp) @ U
        
        # Compute Floquet Hamiltonian: HF = (i * hbar / period) * log[U]
        HF = 1j * hbar / period * logm(U)
        
        # Diagonalize Floquet Hamiltonian
        quasi_energies_raw, _ = np.linalg.eigh(HF)
        quasi_energies_raw = quasi_energies_raw.real
        
        # Determine omega for Brillouin zone mapping
        if brillouin_omega is not None:
            omega = brillouin_omega
        else:
            omega = frequency  # Use the frequency from this calculation
        
        # Map to first Brillouin zone: [-ħω/2, +ħω/2]
        brillouin_half_width = hbar * omega / 2
        quasi_energies_BZ = ((quasi_energies_raw + brillouin_half_width) % (hbar * omega)) - brillouin_half_width
        
        # Apply rescaling if requested
        if rescale_omega is not None:
            rescale_factor = 1 / (hbar * rescale_omega)
            quasi_energies_final = quasi_energies_BZ * rescale_factor
        else:
            # Use the same omega as for Brillouin zone mapping
            rescale_factor = 1 / (hbar * omega)
            quasi_energies_final = quasi_energies_BZ * rescale_factor
        
        # Sort quasi-energies
        quasi_energies_final = np.sort(quasi_energies_final)
        
        return quasi_energies_final
        
    def _solve_quasi_energies_tuple_format(self, Nt=600, hbar=1, rescale_omega=None, brillouin_omega=None):
        """Handle the old tuple format for backward compatibility."""
        
        # Extract parameters from time_dep_args
        # Expected format: (param1, param2, ..., T) where T is the period
        *params, T = self.time_dep_args
        
        # Identify which parameters are arrays and their shapes
        param_arrays = []
        param_shapes = []
        param_is_array = []
        
        for i, param in enumerate(params):
            param_array = np.asarray(param)
            if param_array.ndim == 0:
                # Scalar parameter
                param_arrays.append(param_array.item())
                param_shapes.append(1)
                param_is_array.append(False)
            else:
                # Array parameter
                if param_array.ndim > 1:
                    raise ValueError(f"Parameter {i} has {param_array.ndim} dimensions. Only 1D arrays are supported.")
                param_arrays.append(param_array)
                param_shapes.append(len(param_array))
                param_is_array.append(True)
        
        # Determine output shape
        array_shapes = [shape for shape, is_array in zip(param_shapes, param_is_array) if is_array]
        if len(array_shapes) == 0:
            # All parameters are scalars - original behavior
            return self._solve_quasi_energies_single(params, T, Nt, hbar, rescale_omega, brillouin_omega)
        
        # Get matrix dimensions
        N = self.H_time_indep.shape[0]
        output_shape = tuple(array_shapes) + (N,)
        
        if self.verbose:
            print(f"Computing quasi-energies for parameter grid with shape: {array_shapes}")
            print(f"Output quasi-energies shape will be: {output_shape}")
        
        # Create parameter grids using meshgrid
        array_params = [param_arrays[i] for i in range(len(params)) if param_is_array[i]]
        if len(array_params) == 1:
            # Single array parameter - no meshgrid needed
            param_grid = array_params[0]
            param_combinations = [(param_grid[i],) for i in range(len(param_grid))]
        else:
            # Multiple array parameters - use meshgrid
            param_grids = np.meshgrid(*array_params, indexing='ij')
            param_combinations = list(zip(*[grid.ravel() for grid in param_grids]))
        
        # Initialize result array
        quasi_energies_result = np.zeros(output_shape)
        
        # Iterate over all parameter combinations
        total_combinations = np.prod(array_shapes)
        
        if self.verbose:
            print(f"Computing {total_combinations} parameter combinations...")
        
        for idx, param_combination in enumerate(param_combinations):
            if self.verbose and (idx + 1) % max(1, total_combinations // 10) == 0:
                print(f"Progress: {idx + 1}/{total_combinations} ({100*(idx+1)/total_combinations:.1f}%)")
            
            # Reconstruct full parameter list for this combination
            full_params = []
            array_idx = 0
            for i, is_array in enumerate(param_is_array):
                if is_array:
                    full_params.append(param_combination[array_idx])
                    array_idx += 1
                else:
                    full_params.append(param_arrays[i])
            
            # Compute quasi-energies for this parameter combination
            quasi_energies_single = self._solve_quasi_energies_single(
                full_params, T, Nt, hbar, rescale_omega, brillouin_omega, verbose=False
            )
            
            # Store result in appropriate location
            if len(array_shapes) == 1:
                quasi_energies_result[idx] = quasi_energies_single
            else:
                # Convert flat index to multi-dimensional index
                multi_idx = np.unravel_index(idx, array_shapes)
                quasi_energies_result[multi_idx] = quasi_energies_single
        
        # Store results
        self.quasi_energies = quasi_energies_result
        
        if self.verbose:
            print(f"Computed quasi-energies with shape: {quasi_energies_result.shape}")
        
        return quasi_energies_result
    
    def _solve_quasi_energies_single(self, params, T, Nt=600, hbar=1, rescale_omega=None, brillouin_omega=None, verbose=None):
        """
        Internal method to compute quasi-energies for a single set of parameters.
        """
        if verbose is None:
            verbose = self.verbose
        
        # Set up time array
        dt = T / (Nt - 1)
        t_array = dt * np.arange(Nt)
        
        # Get matrix dimensions
        N = self.H_time_indep.shape[0]
        
        # Convert to dense arrays for efficient matrix operations
        H0_dense = self.H_time_indep.toarray()
        identity_N = np.eye(N, dtype=complex)
        
        # Pre-allocate temporary array
        H_temp = np.zeros((N, N), dtype=complex)
        
        # Time evolution
        U = identity_N.copy()
        dt_over_hbar = dt / hbar
        
        for k in range(Nt):
            t = t_array[k]
            # Compute time-dependent Hamiltonian at time t
            H_td = self.H_time_dep(*params, t)
            
            # Total Hamiltonian = H0 + H_time_dep(t)
            if hasattr(H_td, 'toarray'):
                H_td_dense = H_td.toarray()
            else:
                H_td_dense = H_td
                
            H_temp[:] = H0_dense + H_td_dense
            U = expm(-1j * dt_over_hbar * H_temp) @ U
        
        # Compute Floquet Hamiltonian: HF = (i * hbar / T) * log[U]
        HF = 1j * hbar / T * logm(U)
        
        # Diagonalize Floquet Hamiltonian
        quasi_energies_raw, _ = np.linalg.eigh(HF)
        quasi_energies_raw = quasi_energies_raw.real
        
        # Determine omega for Brillouin zone mapping
        if brillouin_omega is not None:
            omega = brillouin_omega
        elif len(params) >= 2:
            # Assume second parameter is omega (for backward compatibility)
            omega = params[1]
        else:
            # Compute omega from period
            omega = 2 * np.pi / T
        
        # Map to first Brillouin zone: [-ħω/2, +ħω/2]
        brillouin_half_width = hbar * omega / 2
        quasi_energies_BZ = ((quasi_energies_raw + brillouin_half_width) % (hbar * omega)) - brillouin_half_width
        
        # Apply rescaling if requested
        if rescale_omega is not None:
            rescale_factor = 1 / (hbar * rescale_omega)
            quasi_energies_final = quasi_energies_BZ * rescale_factor
        else:
            # Use the same omega as for Brillouin zone mapping
            rescale_factor = 1 / (hbar * omega)
            quasi_energies_final = quasi_energies_BZ * rescale_factor
        
        # Sort quasi-energies
        quasi_energies_final = np.sort(quasi_energies_final)
        
        return quasi_energies_final

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
        if self.H_time_dep is not None:
            raise ValueError("For time-dependent Hamiltonians, use solve_quasi_energies() instead.")
            
        energies, eigvecs = eigsh(self.H, k=k, which="SM")
        if self.verbose:
                print("Solved eigenvalue problem.")

        # Sort eigenpairs.
        order = np.argsort(energies)
        energies = energies[order]
        eigvecs = eigvecs[:, order]

        if self.mesh: # if a mesh is provided
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
        self.wavefunctions = normalized_wavefunctions if self.mesh else eigvecs
        self.densities = normalized_densities if self.mesh else None

        return self.energies, self.wavefunctions

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
            if not self.mesh: # if mesh is not provided
                raise ValueError(""".plot() does not support plotting when mesh is not provided. WARNING: if you are using bosonic operator 
                implementation, you will need to also provide a basis.""")
            else:
                raise RuntimeError("No densities found. Call .solve(k) first.")
            
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