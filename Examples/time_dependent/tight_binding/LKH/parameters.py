from numpy import pi, sqrt, array, vdot
from numpy import sum as np_sum
from perturbative_well_solver import TriangularWell

def get_params(Lx, Ly, Lz, LB, LE, m, e, hbar, Nz = 5, center_z = 0):
    """Calculate derived parameters from the configuration values."""

    # ===================================
    # Material-related parameters
    # ===================================
    g1 = 13.35 # Luttinger parameter Ge
    g2 = 4.25 # Luttinger parameter Ge
    g3 = 5.69 # Luttinger parameter Ge
    gs = g2 * sqrt(1 - 3/8 * (1 - (g3 / g2)**2)) # spherical Luttinger parameter
    kappa = 3.41 # Isotropic magnetic Luttinger parameter
    Gk = g1 + 2.5 * gs
    Gs = -2 * gs

    mzH = m / (Gk + 9 * hbar**2 / 4 * Gs) # out_of-plane "HH" effective mass 
    mzL = m / (Gk + 1 * hbar**2 / 4 * Gs) # out_of-plane "HH" effective mass

    mpH = m / (Gk + 3 * hbar**2 / 4 * Gs) # in-plane "HH" effective mass
    mpL = m / (Gk + 7 * hbar**2 / 4 * Gs) # in-plane "LH" effective mass

    # ===================================
    # Lengths derived parameters
    # ===================================
    mu_b = e * hbar / (2 * m) # Bohr magneton
    Ez = hbar**2 / (2 * mzH * e * LE**3) # electric field along z
    Bz = hbar / (e * LB**2) # Magnetic field
    wx = hbar / (mpH * Lx**2) # harmonic frequency along x 
    wy = hbar / (mpH * Ly**2) # harmonic frequency along y

    # ===================================
    # Energies and overlaps along z
    # ===================================
    solver_heavy = TriangularWell(L=Lz, E=Ez, N=Nz+2, center=center_z, mass=mzH, hbar=hbar)
    EH_s, vecs_heavy = solver_heavy.solve(Nz)
    
    solver_light = TriangularWell(L=Lz, E=Ez, N=Nz+2, center=center_z, mass=mzL, hbar=hbar)
    EL_s, vecs_light = solver_light.solve(Nz)
    
    EH_0 = EH_s[0]
    pz_HL_s = array([solver_light.p_mn(vecs_heavy[:,0], vecs_light[:,j]) for j in range(Nz)], dtype=complex) # which solver is used to compute p_mn is irrelevant
    overlaps = array([vdot(vecs_heavy[:,0], vecs_light[:,j]) for j in range(Nz)], dtype=complex) # which solver is used to compute <m|n> is irrelevant

    # ===================================
    # Effective Hamiltonian parameters
    # ===================================
    lambda_1 = 3 * hbar**4 * Gs**2 / (4 * m**2) * np_sum(overlaps**2 / (EH_0 - EL_s) ) # lambda_1 correction coefficient 
    lambda_2 = 3 * hbar**4 * Gs**2 / (4 * m**2) * np_sum(pz_HL_s**2 /  (EH_0 - EL_s) ) # lambda_2 correction coefficient  

    a3 = 3 * hbar**4 * Gs**2 / (4 * m**2) * np_sum(pz_HL_s * overlaps /(EH_0 - EL_s) ) # cubic SOI strength

    Delta = 6 * hbar * Bz * kappa * mu_b # Zeeman splitting

    # ===================================
    # Formatting output dictionary
    # ===================================
    params = {
        "g1" : g1,
        "g2" : g2,
        "g3" : g3,
        "gs" : gs,
        "wx" : wx,
        "wy" : wy,
        "EH" : EH_0,
        "EL" : EL_s,
        "overlap" : overlaps,
        "a3" : a3,
        "Delta" : Delta,
        "Ez" : Ez,
        "Bz" : Bz,
        "mu_b" : mu_b,
        "Gk" : Gk,
        "Gs" : Gs,
        "mpH" : mpH,
        "mpL" : mpL,
        "mzH" : mzH,
        "mzL" : mzL,
        "lambda_1" : lambda_1,
        "lambda_2" : lambda_2,
        "pz_HL" : pz_HL_s,
        "kappa" : kappa,
    }

    return params