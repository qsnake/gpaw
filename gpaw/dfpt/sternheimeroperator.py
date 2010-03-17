"""This module implements the linear operator in the Sternheimer equation."""

__author__ = "Kristen Kaasbjerg (kkaa@fysik.dtu.dk)"
__date__ = "2010-01-01 -- 20xx-xx-xx"

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
    
    def __init__(self, hamiltonian, wfs, gd):
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
        self.dtype = float
        
    def set_blochstate(self, k, n):
        """Set k-vector and band index for the Bloch-state in consideration.

        Parameters
        ----------
        k: int
           k-point index
        n: int
           Band index

        """
        
        self.k = k
        self.n = n
        
    def matvec(self, x):
        """Matrix vector multiplication for scipy.sparse.linalg cgs solver.

        Parameters
        ----------
        x: ndarry
            1-dimensional array holding the grid representation of the vector

        """

        assert self.k is not None
        assert self.n is not None

        # Output array
        y_G = self.gd.zeros()
        shape = y_G.shape

        size = x.size
        assert size == np.prod(shape)
        
        x_G = x.reshape(shape)
        
        kpt = self.wfs.kpt_u[self.k]
        
        self.hamiltonian.apply(x_G, y_G, self.wfs, kpt,
                               calculate_P_ani=True)

        y_G -= kpt.eps_n[self.n] * x_G
        # Project out undesired (numerical) components
        self.project(y_G)
        
        y = y_G.ravel()
        
        return y
    
    def project(self, a_G):
        """Project the vector onto the unoccupied states at k+q.

        A new vector is not created!
        
        Implementation for q != 0 to be done !

        ::

               --                    --             
          P  = >  |Psi ><Psi | = 1 - >  |Psi ><Psi |
           c   --     c     c        --     v     v 
                c                     v
        
        """

        # assert self.k is not None
        
        nbands = self.wfs.nvalence/2
        # k+q-vector
        kpt = self.wfs.kpt_u[0]
        psit_nG = kpt.psit_nG[:nbands]

        proj_n = self.gd.integrate(a_G * psit_nG)

        # Project out one orbital at a time
        for n in range(nbands):

            a_G -= proj_n[n] * psit_nG[n]
            
        # Do the projection in one go - figure out how to use np.dot correctly
        # a_G -= np.dot(proj_n, psit_nG)
