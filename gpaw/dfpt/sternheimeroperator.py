"""This module implements the linear operator in the Sternheimer equation."""

import numpy as np


class SternheimerOperator:
    """Class implementing the linear operator in the Sternheimer equation.

    Sternheimer equation::
    
         (H - eps_n) P_c |\delta\Psi_n> = - P_c * \deltaV_KS * |\Psi_n>

    where P_c is the projection operator onto the unoccupied states.
    
    The main purpose of this class is to provide a method ``matvec`` that can
    evaluate the multiplication with a vector. This is the only information
    about the linear operator that is required by iterative Krylov solvers.
    
    """
    
    def __init__(self, hamiltonian, wfs, gd, dtype=float):
        """Init method."""

        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.gd = gd
        
        # Variables for k-point and band index
        self.k = None
        self.n = None

        # For scipy's linear solver
        N = np.prod(gd.n_c)
        self.shape = (N,N)
        self.dtype = dtype
        
    def set_blochstate(self, n, k):
        """Set k-vector and band index for the Bloch-state in consideration.

        Parameters
        ----------
        n: int
           Band index
        k: int
           k-point index

        """

        self.n = n        
        self.k = k
        
    def matvec(self, x):
        """Matrix-vector multiplication for scipy.sparse.linalg solvers.

        Parameters
        ----------
        x: ndarry
            1-dimensional array holding the representation of a gpaw grid
            vector.

        """

        assert self.n is not None
        assert self.k is not None
        print "We segfault here. 34"
        
        kpt = self.wfs.kpt_u[self.k]
        
        # Output array
        y_G = self.gd.zeros(dtype=self.dtype)
        shape = y_G.shape

        size = x.size
        assert size == np.prod(shape)
        
        x_G = x.reshape(shape)
      
        self.hamiltonian.apply(x_G, y_G, self.wfs, kpt,
                               calculate_P_ani=True)
        y_G -= kpt.eps_n[self.n] * x_G

        # Project out undesired (numerical) components
        self.project(y_G)
        
        y = y_G.ravel()
        
        return y
    
    def project(self, x_G):
        """Project the vector onto the unoccupied states at k+q.

        A new vector is not created!
        
        Implementation for q != 0 to be done !

        ::

               --                    --             
          P  = >  |Psi ><Psi | = 1 - >  |Psi ><Psi |
           c   --     c     c        --     v     v 
                c                     v
        
        """

        assert self.k is not None

        nbands = max(1, self.wfs.nvalence/2)
        # k+q vector
        kpt = self.wfs.kpt_u[self.k]
        psit_nG = kpt.psit_nG[:nbands]

        proj_n = self.gd.integrate(psit_nG.conjugate() * x_G)

        # Project out one orbital at a time
        for n in range(nbands):

            x_G -= proj_n[n] * psit_nG[n]

        # Do the projection in one go - figure out how to use np.dot correctly
        # a_G -= np.dot(proj_n, psit_nG)
