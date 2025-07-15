import numpy as np
from scipy import sparse
from numqt import *

class TriangularWell:
    """
    Triangular potential well solver using a basis of infinite well states.
    """

    def __init__(self, L: float, E: float, N: int, center: float, mass: float, hbar: float):
        """
        Initialize the triangular well problem.

        :param L: Width of the infinite potential well (length units).
        :type L: float
        :param E: Slope of the triangular potential (energy per length).
        :type E: float
        :param N: Number of basis states to include.
        :type N: int
        :param center: Center position of the well.
        :type center: float
        :param mass: Particle mass.
        :type mass: float
        :param hbar: Reduced Planck constant hbar.
        :type hbar: float
        """
        self.L = L
        self.E = E
        self.N = N
        self.center = center
        self.mass = mass

        # Set up spatial mesh and basis functions
        self._setup_mesh_and_basis(dx=0.001)
        # Compute canonical operators
        self._get_operators(threshold=1e-15, hbar=hbar)
        # Construct Hamiltonian
        self._get_H()

    def _setup_mesh_and_basis(self, dx: float):
        """
        Create the spatial mesh and define the basis functions.

        :param dx: Spatial step size for the mesh.
        :type dx: float
        """
        # Compute number of grid points
        nx = int(np.abs((self.center + self.L/2) - (self.center - self.L/2)) / dx)
        xbounds = (self.center - self.L/2, self.center + self.L/2)
        # Initialize mesh
        self.mesh = Mesh(dims=1, xbounds=xbounds, nx=nx)
        # Define basis constructor
        self.basis = lambda n, mesh: iw_state(n, mesh, center=self.center, L=self.L)

    def _get_operators(self, threshold: float, hbar: float):
        """
        Compute canonical position and momentum operators in the chosen basis.

        :param threshold: Threshold for discarding small matrix elements.
        :type threshold: float
        :param hbar: Reduced Planck constant hbar for operator scaling.
        :type hbar: float
        """
        # Compute operators
        operators = canonic_ops(
            self.mesh,
            basis=(self.basis, self.N),
            ops_to_compute=["p2", "p", "x2", "x"],
            hbar=hbar,
            threshold=threshold
        )
        ops = operators.get_ops()

        # Store operators
        self.p = ops["p"]
        self.p2 = ops["p2"]
        self.x = ops["x"]
        self.x2 = ops["x2"]
        # Identity matrix for convenience
        self.id = sparse.eye(self.x.shape[0])

    def _get_H(self):
        """
        Construct the total Hamiltonian matrix for the triangular well.

        The Hamiltonian is given by:
         H = p^2 / 2m + E x ,
        where p^2/2m is the kinetic term and E * x is the linear potential.
        """
        # Kinetic energy term
        self.H0 = self.p2 / (2 * self.mass)
        # Potential energy term
        self.V = self.E * self.x
        # Total Hamiltonian
        self.H = self.H0 + self.V

    def solve(self, k: int, verbose: bool = False):
        """
        Solve the time-independent Schrödinger equation for lowest eigenstates.

        :param k: Number of eigenvalues and eigenfunctions to compute.
        :type k: int
        :param verbose: If True, print solver progress.
        :type verbose: bool
        :return: Tuple of eigenvalues array and wavefunctions matrix.
        :rtype: (np.ndarray, np.ndarray)
        """
        # Diagonalize Hamiltonian
        eigvals, eigvecs = Hamiltonian(
            H_time_indep=self.H,
            mesh=self.mesh,
            basis=(self.basis, self.N),
            verbose=verbose
        ).solve(k)
        # Store results
        self.eigenvalues = eigvals
        self.wavefunctions = eigvecs
        return eigvals, eigvecs

    def p_mn(self, m: np.ndarray, n: np.ndarray) -> complex:
        """
        Compute the matrix element <m|p|n> for two quantum states and a momentum operator.
    
        <m|p|n> = m^† p n = conjugate(m) dot (p dot n)
    
        :param m: Bra state |m> represented as a 1D complex-valued numpy array.
        :type m: np.ndarray
        :param n: Ket state |n> represented as a 1D complex-valued numpy array.
        :type n: np.ndarray
        :return: Complex expectation value <m|p|n>.
        :rtype: complex
        """
        # Ensure shapes agree
        if m.ndim != 1 or n.ndim != 1:
            raise ValueError("States m and n must be one-dimensional arrays.")
        if self.p.shape[1] != n.shape[0] or m.shape[0] != self.p.shape[0]:
            raise ValueError(
                "Dimensions of m, p, and n do not align: "
                f"m: {m.shape}, p: {self.p.shape}, n: {n.shape}"
            )
    
        # Compute p acting on |n>
        pn = self.p.dot(n)
        # Compute bra m times pn
        return np.vdot(m, pn)
