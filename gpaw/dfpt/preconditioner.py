"""This module wraps the gpaw preconditioner for use with scipy solvers."""

import numpy as np

from gpaw.preconditioner import Preconditioner


class ScipyPreconditioner:

    def __init__(self, gd, kin, project, dtype=float):
        """Init the gpaw preconditioner.

        Parameters
        ----------
        gd: GridDescriptor
            Coarse grid
        kin: Something
            Kinetic energy operator
            
        """

        self.project = project
        self.pc = Preconditioner(gd, kin, dtype=dtype)
        self.pc.allocate()
        self.gd = gd
        
        # For scipy's linear solver
        N = np.prod(gd.n_c)
        self.shape = (N,N)
        self.dtype = float
        
    def matvec(self, x):
        """Matrix vector multiplication for scipy.sparse.linalg solvers.

        Parameters
        ----------
        x: ndarry
            1-dimensional array holding the grid representation of the vector

        """

        # Output array
        y_G = self.gd.zeros()
        shape = y_G.shape

        size = x.size
        assert size == np.prod(shape)
        
        x_G = x.reshape(shape)

        # Call gpaw preconditioner
        y_G = self.pc(x_G)

        # Project out undesired (numerical) components
        self.project(y_G)
        
        y = y_G.ravel()
        
        return y
