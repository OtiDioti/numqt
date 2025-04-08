from scipy.sparse import kron
from scipy.special import hermite, eval_hermite
import numpy as np
import math

def iterative_kron(listt:list):
     """
    Returns kronecker product of the matrices contained in the provided list,
    whilst maininting the correct order.
    
    Parameters
    ----------
    listt : list
        The list of matrices to be tensor product.
    
    Returns
    -------
    ndarray
        The matrix resulting from the tensor product of all the matrices in the 
        listt
    """
    # Kronecker product
     op = listt[0]
     for f in listt[1:]:
         op = kron(op, f, format='csr')
     return op



def iw_state(n: int, mesh: np.ndarray, center: float, L :float) -> tuple:
    r"""
    Compute the discretized eigenstate for an infinite square well and its derivatives.
    
    The infinite square well (or particle in a box) is defined on the interval
    \([center - L/2, \; center + L/2]\) where 
        L = mesh[-1] - mesh[0]
    is the full width of the well.
    
    The normalized eigenfunctions are given by
    
    .. math::
        \psi_n(x) = \sqrt{\frac{2}{L}} \sin\!\left( \frac{(n+1)\pi \, [x - (center - L/2)]}{L} \right)
    
    for \(n = 0, 1, 2, \dots\) (with \(n=0\) corresponding to the first eigenstate).
    
    Parameters
    ----------
    n : int
        Quantum number (starting at 0 for the first eigenstate).
    mesh : np.ndarray
        Array of x values (covering the entire domain of the well).
    center : float
        The center of the infinite square well.
    L : float
        Well length
        
    Returns
    -------
    tuple
        tuple containing the wavefunction and its derivatives:
        - 'psi': The wavefunction
        - 'dpsi': First derivative 
        - 'd2psi': Second derivative 
    """

    
    # For a well centered at 'center', the left edge is at (center - L/2).
    x_shifted = mesh - (center - L/2)
    k = (n + 1) * np.pi / L  # Wave number
    
    # Calculate the wavefunction
    amplitude = np.sqrt(2 / L)
    psi = amplitude * np.sin(k * x_shifted)
    
    result = (psi,)
    
    # First derivative: d/dx[sin(kx)] = k*cos(kx)
    dpsi = amplitude * k * np.cos(k * x_shifted)
    result += (dpsi,)
    
    # Second derivative: d²/dx²[sin(kx)] = -k²*sin(kx)
    d2psi = -amplitude * k**2 * np.sin(k * x_shifted)
    result += (d2psi,)
    
    return result

def ho_state(n: int, mesh: np.ndarray, omega: float, mass: float = 1, hbar: float = 1) -> tuple:
    r"""
    Compute the discretized eigenstate for the one-dimensional harmonic oscillator
    and its derivatives for a given mass, frequency, and reduced Planck constant.

    The normalized eigenfunctions for the harmonic oscillator are given by:
    
    .. math::
        \psi_n(x) = \left(\frac{m\omega}{\pi\hbar}\right)^{1/4} \frac{1}{\sqrt{2^n n!}}
                     \, e^{-\frac{m\omega}{2\hbar}x^2} H_n\left(\sqrt{\frac{m\omega}{\hbar}}\,x\right),
    
    where \(H_n\) is the Hermite polynomial of order \(n\).

    By defining:
    
    .. math::
        y = \sqrt{\frac{m\omega}{\hbar}}\,x,
    
    the eigenfunctions can be written as:
    
    .. math::
        \psi_n(x) = N \, e^{-y^2/2} \, H_n(y),
    
    with the normalization constant

    .. math::
        N = \frac{(\frac{m\omega}{\hbar})^{1/4}}{\sqrt{2^n n! \, \pi^{1/4}}}.
    
    Note that when \(m=\omega=\hbar=1\) the expressions reduce to the natural units version.

    Parameters
    ----------
    n : int
        Quantum number (starting at 0).
    mesh : ndarray
        Array of x values at which to evaluate the eigenstate.
    mass : float
        Mass of the oscillator.
    omega : float
        Angular frequency of the oscillator.
    hbar : float
        Reduced Planck constant.

    Returns
    -------
    tuple
        A tuple containing:
         - psi: The wavefunction \(\psi(x)\)
         - dpsi: The first derivative \(\psi'(x)\)
         - d2psi: The second derivative \(\psi''(x)\)
    """
    # Compute the scaling factor and change of variable:
    alpha = mass * omega / hbar
    # y is the dimensionless variable:
    y = np.sqrt(alpha) * mesh

    # Normalization constant:
    norm = (alpha / np.pi)**0.25 / np.sqrt(2.0**n * math.factorial(n))
    
    # Evaluate the Hermite polynomial:
    Hn = eval_hermite(n, y)
    
    # Gaussian factor:
    gaussian = np.exp(-0.5 * y**2)
    
    # Wavefunction:
    psi = norm * gaussian * Hn
    
    # --- First Derivative ---
    # The derivative d/dx works via the chain rule d/dx = sqrt(alpha) d/dy.
    # And note that: dHn/dy = 2n * H_{n-1}(y) for n>0 (and 0 for n = 0).
    if n > 0:
        dHn = 2 * n * eval_hermite(n - 1, y)
    else:
        dHn = np.zeros_like(y)
    # dpsi/dy (derivative with respect to y) using the product rule:
    # d/dy [e^{-y^2/2} H_n(y)] = e^{-y^2/2} [H'_n(y) - y H_n(y)]
    dpsi_dy = gaussian * (dHn - y * Hn)
    # Convert to derivative with respect to x
    dpsi = norm * np.sqrt(alpha) * dpsi_dy
    
    # --- Second Derivative ---
    # The second derivative: d^2/dx^2 = alpha * d^2/dy^2.
    # First, compute the second derivative with respect to y.
    # We have:
    # d^2/dy^2 [e^{-y^2/2} H_n(y)] = e^{-y^2/2} [H''_n(y) - 2y H'_n(y) + (y^2 - 1) H_n(y)].
    if n > 1:
        # H''_n(y) can be obtained via: H''_n(y) = 4n(n-1) H_{n-2}(y)
        d2Hn = 4 * n * (n - 1) * eval_hermite(n - 2, y)
    else:
        d2Hn = np.zeros_like(y)
    # Compute second derivative with respect to y:
    d2psi_dy2 = gaussian * (d2Hn - 2 * y * dHn + (y**2 - 1) * Hn)
    # Multiply by alpha to get derivative with respect to x:
    d2psi = norm * alpha * d2psi_dy2
    
    return (psi, dpsi, d2psi)



def a_comm(a,b):
    """
    Returns anti commutator between a and b.
    
    Parameters
    ----------
    a,b : ndarray
        the operators in matrix form to take the anticommutator of
    
    Returns
    -------
    ndarray
        The anticommutator of a and b
    """
    return 0.5 * (a@b + b@a)


def within(x0,x1,listt):
    """
    checks if a list of values is all contained within the interval [x0,x1]
    
    Parameters
    ----------
    x0,x1 : float
        bounds of the interval to check
    listt : list
        list of numbers
    
    Returns
    -------
    bool
        whether all elements of list fall within the bounds
    """
    return (x0 <=np.array(listt)).all() and (np.array(listt) <= x1).all()
