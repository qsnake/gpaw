"""This module implements the linear operator in the Sternheimer equation."""

import numpy as np

from gpaw.fd_operators import Laplace

class SternheimerOperator:
    """Class implementing the linear operator in the Sternheimer equation.

    Sternheimer equation::
    
         (H - eps_nk) P_c |\delta\Psi_nk> = - P_c * \delta V_q * |\Psi_nk>

    where P_c is the projection operator onto the unoccupied states. For a
    perturbation with a specific q-vector, only projections onto states at k+q
    will be different from zero.
    
    The main purpose of this class is to implement the multiplication with a
    vector (``apply`` member function). An additional ``matvec`` member
    function has been defined so that instances of this class can be passed as
    a linear operator to scipy's iterative Krylov solvers in
    ``scipy.sparse.linalg``. 
    
    """
    
    def __init__(self, hamiltonian, wfs, gd, dtype=float):
        """Init method."""

        self.hamiltonian = hamiltonian
        self.kin = Laplace(gd, scale=-0.5, n=3, dtype=dtype, allocate=True)        
        self.wfs = wfs
        self.gd = gd
        
        # Variables for k-point and band index
        self.k = None
        self.n = None

        # For scipy's linear solver
        N = np.prod(gd.N_c)
        self.shape = (N,N)
        self.dtype = dtype
        
    def set_blochstate(self, n=None, k=None):
        """Set k-vector and band index for the Bloch-state in consideration.

        Parameters
        ----------
        n: int
           Band index
        k: int
           k-point index

        """

        self.n = n
        # k+q vector
        self.k = k

    def apply(self, x_nG, y_nG):
        """Apply the Sternheimer operator to a vector.

        Parameters
        ----------
        x_nG: ndarray
            GPAW grid vector(s) to which the Sternheimer operator is applied.
        y_nG: ndarray
            Resulting vector(s).
            
        """

        assert len(x_nG.shape) in (3, 4)
        assert x_nG.shape == y_nG.shape
        assert self.k is not None

        kpt = self.wfs.kpt_u[self.k]

        # Kintetic energy
        self.kin.apply(x_nG, y_nG, kpt.phase_cd)
        # Local part of effective potential 
        self.hamiltonian.apply_local_potential(x_nG, y_nG, kpt.s)
        # Non-local part from projectors
        shape = x_nG.shape[:-3]
        P_ani = self.wfs.pt.dict(shape)
        self.wfs.pt.integrate(x_nG, P_ani, kpt.q)
            
        for a, P_ni in P_ani.items():
            dH_ii = unpack(self.hamiltonian.dH_asp[a][kpt.s])
            P_ani[a] = np.dot(P_ni, dH_ii)

        self.wfs.pt.add(y_nG, P_ani, kpt.q)

        # Eigenvalue term
        if len(x_nG.shape == 3):
            assert self.n is not None
            y_nG -= kpt.eps_n[self.n] * x_nG
        else:
            for n, a_G in enumerate(x_nG):
                y_nG[n,:] -= kpt.eps_n[n] * a_G

        # Project out undesired (numerical) components
        self.project(y_nG)

    def project(self, x_nG):
        """Project the vector onto the unoccupied states at k+q.

        Implementation for q != 0 to be done !

        ::

               --                    --             
          P  = >  |Psi ><Psi | = 1 - >  |Psi ><Psi |
           c   --     c     c        --     v     v 
                c                     v
        
        """
        
        assert self.k is not None

        # This should be the k+q vector !!!!
        kpt = self.wfs.kpt_u[self.k]
        
        # Occupied wave function
        nbands = max(1, self.wfs.nvalence/2)
        psit_nG = kpt.psit_nG[:nbands]
        
        # Project out one orbital at a time
        for n in range(nbands):
            proj_n = self.gd.integrate(psit_nG[n].conjugate() * x_nG)
            x_nG -= proj_n * psit_nG[n]

        # Do the projection in one go - figure out how to use np.dot correctly
        # a_G -= np.dot(proj_n, psit_nG)

    def matvec(self, x):
        """Matrix-vector multiplication for scipy's Krylov solvers.

        This is a wrapper around the ``apply`` member function above. It allows
        to multiply the sternheimer operator onto the 1-dimensional scipy
        representation of a gpaw grid vector(s).
        
        Parameters
        ----------
        a: ndarray
            1-dimensional array holding the representation of a gpaw grid
            vector (possibly a set of vectors).

        """

        # Check that a is 1-dimensional
        assert len(x.shape) == 1
        
        # Find the number of states in x
        grid_shape = tuple(self.gd.N_c)
        assert ((x.size % np.prod(grid_shape)) == 0), ("Incompatible array ",
                                                       "shapes")
        # Number of states
        N = x.size / np.prod(grid_shape)
        
        # Output array
        y_nG = self.gd.zeros(n=N, dtype=self.dtype)
        shape = y_nG.shape

        assert x.size == y_nG.shape
        
        x_nG = x.reshape(shape)

        self.apply(x_nG, y_nG)
        
        y = y_nG.ravel()
        
        return y
