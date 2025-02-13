from scipy.sparse import kron
import numpy as np

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
         op = kron(op, f)
     return op

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