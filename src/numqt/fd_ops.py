from scipy.sparse import eye as sp_eye, diags, kron, csr_matrix
from numpy import sqrt, arange, arctan, cos, sin, complex128, pi

def fock_darwin_ops(dim_plus, dim_minus, other_dims=None):
    """Returns the Fock-Darwin bosonic operators a+, a+†, a-, a-†"""
    diag_plus = sqrt(arange(1, dim_plus))
    a_plus = diags(diag_plus, 1, shape=(dim_plus, dim_plus), format='csr')
    ad_plus = diags(diag_plus, -1, shape=(dim_plus, dim_plus), format='csr')
    
    diag_minus = sqrt(arange(1, dim_minus))
    a_minus = diags(diag_minus, 1, shape=(dim_minus, dim_minus), format='csr')
    ad_minus = diags(diag_minus, -1, shape=(dim_minus, dim_minus), format='csr')
    
    id_plus = sp_eye(dim_plus, format='csr')
    id_minus = sp_eye(dim_minus, format='csr')
    
    a_plus_full = kron(a_plus, id_minus)
    ad_plus_full = kron(ad_plus, id_minus)
    a_minus_full = kron(id_plus, a_minus)
    ad_minus_full = kron(id_plus, ad_minus)
    
    if other_dims is not None:
        for dim in reversed(other_dims):
            id_op = sp_eye(dim, format='csr')
            a_plus_full = kron(id_op, a_plus_full)
            ad_plus_full = kron(id_op, ad_plus_full)
            a_minus_full = kron(id_op, a_minus_full)
            ad_minus_full = kron(id_op, ad_minus_full)
    
    return {
        'plus': [a_plus_full, ad_plus_full],
        'minus': [a_minus_full, ad_minus_full]
    }


def calculate_anisotropic_frequencies(omega_x, omega_y, omega_c):
    """Calculate frequencies for anisotropic Fock-Darwin system"""
    omega_sum = (omega_x**2 + omega_y**2)
    omega_dif = (omega_x**2 - omega_y**2)
    
    discriminant = sqrt(omega_dif**2 + 2*omega_c**2 * omega_sum + omega_c**4)
    
    omega_plus = sqrt(0.5*(omega_sum + omega_c**2 + discriminant))
    omega_minus = sqrt(0.5*(omega_sum + omega_c**2 - discriminant))
    
    if abs(omega_dif) < 1e-15:
        theta = pi / 4
    else:
        theta = 0.5 * arctan(omega_c * sqrt(omega_c**2 + 2*omega_sum)) / (omega_dif)
    
    return omega_plus, omega_minus, theta


def fock_darwin_x(a_plus, ad_plus, a_minus, ad_minus, omega_x, omega_plus, omega_minus,
                  theta, mass, hbar):
    """x position operator"""
    l_x = sqrt(hbar / (2 * mass * omega_x))
    k_p = omega_plus  / omega_x
    k_m = omega_minus / omega_x

    constant_plus  =    l_x * sqrt(k_p / (k_p**2 + 1)) * cos(theta)
    constant_minus = 1j*l_x * sqrt(k_m / (k_m**2 + 1)) * sin(theta)
    x = (constant_plus  * (ad_plus  + a_plus) + 
         constant_minus * (ad_minus - a_minus)).astype(complex128)
    
    return x


def fock_darwin_y(a_plus, ad_plus, a_minus, ad_minus, omega_x, omega_plus, omega_minus,
                  theta, mass, hbar):
    """y position operator"""
    l_x = sqrt(hbar / (2 * mass * omega_x))
    k_p = omega_plus  / omega_x
    k_m = omega_minus / omega_x

    constant_minus  =    l_x * sqrt(1 / (k_m*(k_p**2 + 1))) * cos(theta)
    constant_plus   = 1j*l_x * sqrt(1 / (k_p*(k_m**2 + 1))) * sin(theta)
    y = (constant_minus * (ad_minus + a_minus) + 
         constant_plus  * (ad_plus  - a_plus)).astype(complex128)

    return y


def fock_darwin_px(a_plus, ad_plus, a_minus, ad_minus, omega_x, omega_plus, omega_minus,
                  theta, mass, hbar):
    """px momentum operator"""
    l_x = sqrt(hbar / (2 * mass * omega_x))
    k_p = omega_plus  / omega_x
    k_m = omega_minus / omega_x

    constant_minus  =   -hbar/(2*l_x) * sqrt((k_m**2 + 1)/(k_m)) * sin(theta)
    constant_plus   = 1j*hbar/(2*l_x) * sqrt((k_p**2 + 1)/(k_p)) * cos(theta)
    px = (constant_minus * (ad_minus + a_minus) + 
          constant_plus  * (ad_plus  - a_plus)).astype(complex128)
    
    return px


def fock_darwin_py(a_plus, ad_plus, a_minus, ad_minus, omega_x, omega_plus, omega_minus,
                  theta, mass, hbar):
    """px momentum operator"""
    l_x = sqrt(hbar / (2 * mass * omega_x))
    k_p = omega_plus  / omega_x
    k_m = omega_minus / omega_x

    constant_plus   =   -hbar/(2*l_x) * sqrt(k_p * (k_m**2 + 1)) * sin(theta)
    constant_minus  = 1j*hbar/(2*l_x) * sqrt(k_m * (k_p**2 + 1)) * cos(theta)
    py = (constant_plus  * (ad_plus  + a_plus) + 
          constant_minus * (ad_minus - a_minus)).astype(complex128)
    
    return py
