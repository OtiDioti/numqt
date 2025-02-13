import numpy as np
from scipy import sparse
from scipy.sparse import kron
from .utils import iterative_kron





# ----------------------------------------------------------
# Spin-3/2 Operators
# ----------------------------------------------------------
s32 = np.sqrt(3) * 0.5
s12 = 0.5
is32 = 1j * s32

def spin32(dimx: int = None, dimy: int = None, dimz: int = None, hbar=1):
    """
    Returns the spin-3/2 operators (J0, Jx, Jy, Jz and their squares) acting on a 4-dimensional
    spin space. Optionally, if additional subspaces are provided (via non-None dimx, dimy, and/or dimz),
    the spin operator is embedded into the tensor product with identity operators for each extra subspace.
    
    Parameters
    ----------
    dimx, dimy, dimz : int, optional
        Dimensions of extra Hilbert spaces. For each one that is not None, an identity operator 
        of that size will be tensor-multiplied on the right.
    hbar : float, optional
        Planck's constant (default 1).
    
    Returns
    -------
    J0, Jx, Jy, Jz, Jx2, Jy2, Jz2 : sparse matrices
        The spin-3/2 operators (and their squares) acting on the full tensor product space.
    """
    # Define the basic 4x4 spin-3/2 matrices.
    J0 = np.eye(4, dtype=complex)
    Jx = np.array([[0, 0, s32 * hbar, 0],
                   [0, 0, 0, s32 * hbar],
                   [s32 * hbar, 0, 0, hbar],
                   [0, s32 * hbar, hbar, 0]], dtype=complex)
    Jy = np.array([[0, 0, -is32 * hbar, 0],
                   [0, 0, 0, is32 * hbar],
                   [is32 * hbar, 0, 0, -1j * hbar],
                   [0, -is32 * hbar, 1j * hbar, 0]], dtype=complex)
    Jz = np.array([[3 * s12, 0, 0, 0],
                   [0, -3 * s12, 0, 0],
                   [0, 0, s12, 0],
                   [0, 0, 0, -s12]], dtype=complex) * hbar

    Jx2 = np.linalg.matrix_power(Jx, 2)
    Jy2 = np.linalg.matrix_power(Jy, 2)
    Jz2 = np.linalg.matrix_power(Jz, 2)

    # Build the list of extra subspaces.
    extra_ops = []
    for d in [dimx, dimy, dimz]:
        if d is not None:
            extra_ops.append(sparse.eye(d, format='csr'))
    
    # Define a helper to embed an operator op.
    def embed(op):
        op_sparse = sparse.csr_array(op)
        if extra_ops:
            return iterative_kron([op_sparse] + extra_ops)
        else:
            return op_sparse

    return (embed(J0), embed(Jx), embed(Jy), embed(Jz),
            embed(Jx2), embed(Jy2), embed(Jz2))

# ----------------------------------------------------------
# Spin-1/2 Operators
# ----------------------------------------------------------
def spin12(dimx: int = None, dimy: int = None, dimz: int = None, hbar=1):
    """
    Returns the spin-1/2 operators (S0, Sx, Sy, Sz and their squares) acting on a 2-dimensional
    spin space. Optionally, if extra subspace dimensions are provided (via dimx, dimy, dimz),
    the operators are embedded via the tensor product with identities on those spaces.
    
    Parameters
    ----------
    dimx, dimy, dimz : int, optional
        Dimensions of extra Hilbert spaces. For each that is not None, an identity operator 
        of that size is included in the tensor product.
    hbar : float, optional
        Planck's constant (default 1).
    
    Returns
    -------
    S0, Sx, Sy, Sz, Sx2, Sy2, Sz2 : sparse matrices
        The spin-1/2 operators (and their squares) acting on the full tensor product space.
    """
    # Pauli matrices (times hbar/2)
    Sx = (hbar/2) * np.array([[0, 1],
                              [1, 0]], dtype=complex)
    Sy = (hbar/2) * np.array([[0, -1j],
                              [1j, 0]], dtype=complex)
    Sz = (hbar/2) * np.array([[1, 0],
                              [0, -1]], dtype=complex)
    S0 = np.eye(2, dtype=complex)

    # For spin-1/2, note that Sx^2 = Sy^2 = Sz^2 = (hbar/2)^2 I.
    Sx2 = (hbar/2)**2 * np.eye(2, dtype=complex)
    Sy2 = (hbar/2)**2 * np.eye(2, dtype=complex)
    Sz2 = (hbar/2)**2 * np.eye(2, dtype=complex)

    extra_ops = []
    for d in [dimx, dimy, dimz]:
        if d is not None:
            extra_ops.append(sparse.eye(d, format='csr'))

    def embed(op):
        op_sparse = sparse.csr_array(op)
        if extra_ops:
            return iterative_kron([op_sparse] + extra_ops)
        else:
            return op_sparse

    return (embed(S0), embed(Sx), embed(Sy), embed(Sz),
            embed(Sx2), embed(Sy2), embed(Sz2))