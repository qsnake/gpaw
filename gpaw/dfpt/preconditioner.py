"""This module wraps the gpaw preconditioner for use with scipy solvers."""

import numpy as np

from gpaw.fd_operators import Laplace
from gpaw.preconditioner import Preconditioner


class ScipyPreconditioner:

    def __init__(self, gd, project, dtype=float):
        """Init the gpaw preconditioner.

        Parameters
        ----------
        gd: GridDescriptor
            Coarse grid
            
        """

        self.project = project
        self.gd = gd
        self.phase_cd = None
        
        kin = Laplace(gd, scale=-0.5, n=3, dtype=dtype, allocate=True)
        self.pc = Preconditioner(gd, kin, dtype=dtype)
        self.pc.allocate()
        
        # For scipy's linear solver
        N = np.prod(gd.n_c)
        self.shape = (N,N)
        self.dtype = dtype

    def set_phases(self, phase_cd):
        """Set phases for ``Transformer`` objects inside the preconditioner."""

        self.phase_cd = phase_cd
        
    def matvec(self, x):
        """Matrix vector multiplication for ``scipy.sparse.linalg`` solvers.

        Parameters
        ----------
        x: ndarry
            1-dimensional array holding the grid representation of the vector

        """

        # Output array
        y_G = self.gd.zeros(dtype=self.dtype)
        shape = y_G.shape

        size = x.size
        assert size == np.prod(shape)
        
        x_G = x.reshape(shape)

        # Call gpaw preconditioner
        y_G = self.pc(x_G, phases=self.phase_cd)

        # Project out undesired (numerical) components
        self.project(y_G)
        
        y = y_G.ravel()
        
        return y
