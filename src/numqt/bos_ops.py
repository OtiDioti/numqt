from scipy.sparse import eye as sp_eye, diags, kron, csr_matrix
import numpy as np

def bos_ops(dimx=None, dimy=None, dimz=None, other_dims=None):
    """
    Returns the bosonic operators a, ad for as many directions as the dimensionalities provided.
    Properly embeds operators in the tensor product space for multiple dimensions.
    
    Parameters
    ----------
    dimx, dimy, dimz : int, optional
        Dimensions of Hilbert spaces for x, y, z directions. 
    other_dims: list[int], optional
        Dimensions of additional subspaces such as spin.
        
    Returns
    -------
    dict : dictionary of sparse matrices
        dictionary of the form {'x': [a_x, ad_x], 'y':[a_y, ad_y], 'z': [a_z, ad_z] }
    """
    if dimx is None and dimy is None and dimz is None:
        raise ValueError("At least one dimensionality must be provided")
    
    # Collect dimensions and direction names in order
    dims = []
    dir_names = []
    
    if dimx is not None:
        dims.append(dimx)
        dir_names.append('x')
    if dimy is not None:
        dims.append(dimy)
        dir_names.append('y')
    if dimz is not None:
        dims.append(dimz)
        dir_names.append('z')
    
    # Create basic operators for each dimension separately
    basic_ops = {}
    for dir_name, dim in zip(dir_names, dims):
        # Annihilation operator
        diag_elements = np.sqrt(np.arange(1, dim))
        a = diags(diag_elements, 1, shape=(dim, dim), format='csr')
        
        # Creation operator
        ad = diags(diag_elements, -1, shape=(dim, dim), format='csr')
        
        basic_ops[dir_name] = [a, ad]
    
    # Identity operators for each dimension
    identities = {}
    for dir_name, dim in zip(dir_names, dims):
        identities[dir_name] = sp_eye(dim, format='csr')
    
    # Other dimension identities
    other_identities = []
    if other_dims is not None:
        for dim in other_dims:
            other_identities.append(sp_eye(dim, format='csr'))
    
    # Create properly embedded operators for each direction
    result = {}
    
    # Different construction based on number of dimensions
    if len(dims) == 1:
        # Only one dimension - no need for tensor products between main dimensions
        dir_name = dir_names[0]
        a, ad = basic_ops[dir_name]
        
        # Add tensor products with other dimensions if needed
        if other_dims is not None:
            for id_op in reversed(other_identities):
                a = kron(id_op, a)
                ad = kron(id_op, ad)
        
        result[dir_name] = [a, ad]
    
    elif len(dims) == 2:
        # Two dimensions (e.g., x and y)
        # For x operator: I_other ⊗ a_x ⊗ I_y
        # For y operator: I_other ⊗ I_x ⊗ a_y
        
        # X operators
        a_x, ad_x = basic_ops['x']
        a_x_full = kron(a_x, identities['y'])
        ad_x_full = kron(ad_x, identities['y'])
        
        # Y operators
        a_y, ad_y = basic_ops['y']
        a_y_full = kron(identities['x'], a_y)
        ad_y_full = kron(identities['x'], ad_y)
        
        # Add tensor products with other dimensions if needed
        if other_dims is not None:
            for id_op in reversed(other_identities):
                a_x_full = kron(id_op, a_x_full)
                ad_x_full = kron(id_op, ad_x_full)
                a_y_full = kron(id_op, a_y_full)
                ad_y_full = kron(id_op, ad_y_full)
        
        result['x'] = [a_x_full, ad_x_full]
        result['y'] = [a_y_full, ad_y_full]
    
    elif len(dims) == 3:
        # Three dimensions (x, y, and z)
        # For x operator: I_other ⊗ a_x ⊗ I_y ⊗ I_z
        # For y operator: I_other ⊗ I_x ⊗ a_y ⊗ I_z
        # For z operator: I_other ⊗ I_x ⊗ I_y ⊗ a_z
        
        # X operators
        a_x, ad_x = basic_ops['x']
        a_x_full = kron(kron(a_x, identities['y']), identities['z'])
        ad_x_full = kron(kron(ad_x, identities['y']), identities['z'])
        
        # Y operators
        a_y, ad_y = basic_ops['y']
        a_y_full = kron(kron(identities['x'], a_y), identities['z'])
        ad_y_full = kron(kron(identities['x'], ad_y), identities['z'])
        
        # Z operators
        a_z, ad_z = basic_ops['z']
        a_z_full = kron(kron(identities['x'], identities['y']), a_z)
        ad_z_full = kron(kron(identities['x'], identities['y']), ad_z)
        
        # Add tensor products with other dimensions if needed
        if other_dims is not None:
            for id_op in reversed(other_identities):
                a_x_full = kron(id_op, a_x_full)
                ad_x_full = kron(id_op, ad_x_full)
                a_y_full = kron(id_op, a_y_full)
                ad_y_full = kron(id_op, ad_y_full)
                a_z_full = kron(id_op, a_z_full)
                ad_z_full = kron(id_op, ad_z_full)
        
        result['x'] = [a_x_full, ad_x_full]
        result['y'] = [a_y_full, ad_y_full]
        result['z'] = [a_z_full, ad_z_full]
    
    return result


def bos_p(a, ad, omega, mass, hbar=1):
    """
    Constructs the momentum operator using bosonic operators.
    
    Parameters
    ----------
    a : scipy.sparse.csr_matrix
        Annihilation operator
    ad : scipy.sparse.csr_matrix
        Creation operator
    mass : float
        Particle mass
    omega : float
        Angular frequency
    hbar : float
        Reduced Planck constant
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Momentum operator in the given basis
    """
    # Calculate the coefficient
    coeff = np.sqrt(hbar * mass * omega / 2)
    
    # Calculate p = i * sqrt(hbar * m * omega / 2) * (ad - a)
    p_operator = 1j * coeff * (ad - a).astype(np.complex128)
    
    return p_operator


def bos_x(a, ad, omega, mass, hbar=1):
    """
    Constructs the position operator using bosonic operators.
    
    Parameters
    ----------
    a : scipy.sparse.csr_matrix
        Annihilation operator
    ad : scipy.sparse.csr_matrix
        Creation operator
    mass : float
        Particle mass
    omega : float
        Angular frequency
    hbar : float
        Reduced Planck constant
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Position operator in the given basis
    """
    # Calculate the coefficient
    coeff = np.sqrt(hbar / (2 * mass * omega))
    
    # Calculate x = sqrt(hbar / (2 * m * omega)) * (a + ad)
    x_operator = coeff * (a + ad).astype(np.complex128)
    
    return x_operator