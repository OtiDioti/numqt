from scipy.sparse import eye as sp_eye
from numpy import sqrt
from .bos_ops import bos_ops

def landau_ops(dim, other_dims=None):
    """Returns the Fock-Darwin bosonic operators a+, a+†, a-, a-†"""
    bosonic_ops = bos_ops(dimx=dim, other_dims=other_dims) # Obtaining bosonic operators
    a, ad = bosonic_ops["x"] # Extracting bosonic operators

    return a, ad

def landau_pi2(a, ad, Bz, e, hbar):
    LB = sqrt(hbar / (e * Bz))
    const = 2 * hbar**2 / (LB**2)
    return const * (ad@a + 0.5 * sp_eye(a.shape[0], format='csr'))

def landau_pi_plus(a, ad, Bz, e, hbar):
    LB = sqrt(hbar / (e * Bz))
    const = sqrt(2) * hbar / LB
    return const * ad

def landau_pi_minus(a, ad, Bz, e, hbar):
    LB = sqrt(hbar / (e * Bz))
    const = sqrt(2) * hbar / LB
    return const * a
