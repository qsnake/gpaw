# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class."""

import numpy as np

from gpaw.operators import Gradient
from gpaw.utilities.blas import axpy


class KPoint:
    """Class for a single k-point.

    The KPoint class takes care of all wave functions for a
    certain k-point and a certain spin.

    Attributes:

    phase_cd: complex ndarray
        Bloch phase-factors for translations - axis c=0,1,2
        and direction d=0,1.
    eps_n: float ndarray
        Eigenvalues.
    f_n: float ndarray
        Occupation numbers.
    psit_nG: ndarray
        Wave functions.
    nbands: int
        Number of bands.

    Parallel stuff:

    comm: Communicator object
        MPI-communicator for domain.
    root: int
        Rank of the CPU that does the matrix diagonalization of
        H_nn and the Cholesky decomposition of S_nn.
    """
    
    def __init__(self, weight, s, k, q, phase_cd):
        """Construct k-point object.

        Parameters:

        gd: GridDescriptor object
            Descriptor for wave-function grid.
        weight: float
            Weight of this k-point.
        s: int
            Spin index: up or down (0 or 1).
        k: int
            k-point index.
        u: int
            Combined spin and k-point index.
        k_c: float-ndarray of shape (3,)
            scaled **k**-point vector (coordinates scaled to
            [-0.5:0.5] interval).
        dtype: type object
            Data type of wave functions (float or complex).
        timer: Timer object
            Optional.

        Note that s and k are global spin/k-point indices,
        whereas u is a local spin/k-point pair index for this
        processor.  So if we have `S` spins and `K` k-points, and
        the spins/k-points are parallelized over `P` processors
        (kpt_comm), then we have this equation relating s,
        k and u::

           rSK
           --- + u = sK + k,
            P

        where `r` is the processor rank within kpt_comm.  The
        total number of spin/k-point pairs, `SK`, is always a
        multiple of the number of processors, `P`.
        """

        self.weight = weight
        self.s = s  # spin index
        self.k = k  # k-point index
        self.q = q  # local k-point index
        self.phase_cd = phase_cd
        
        self.eps_n = None
        self.f_n = None
        self.P_ani = None

        # Only one of these two will be used:
        self.psit_nG = None  # wave functions on 3D grid
        self.C_nM = None     # LCAO coefficients for wave functions XXX

        self.rho_MM = None
        
        self.P_aMi = None
        self.S_MM = None
        self.T_MM = None

    def add_to_density(self, nt_G, use_lcao, basis_functions):
        raise DeprecationWarning
        """Add contribution to pseudo electron-density."""
        self.add_to_density_with_occupation(nt_G, use_lcao, self.f_n,
                                            basis_functions)
        
    def add_to_density_with_occupation(self, nt_G, use_lcao, f_n,
                                       basis_functions):
        raise DeprecationWarning
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        print 'nnnnnnnnnnnnnnoooooooooooo'
        if use_lcao:
            C_nM = self.C_nM
            rho_MM = np.dot(C_nM.conj().T * f_n, C_nM)
            basis_functions.construct_density(rho_MM, nt_G)
        else:
            if self.dtype == float:
                for f, psit_G in zip(f_n, self.psit_nG):
                    axpy(f, psit_G**2, nt_G)  # nt_G += f * psit_G**2
            else:
                for f, psit_G in zip(f_n, self.psit_nG):
                    nt_G += f * (psit_G * np.conjugate(psit_G)).real
        
    def add_to_kinetic_density(self, taut_G):
        """Add contribution to pseudo kinetic energy density."""
        ddr = [Gradient(self.gd, c, dtype=self.dtype).apply for c in range(3)]
        d_G = self.gd.empty(dtype=self.dtype)
        for f, psit_G in zip(self.f_n, self.psit_nG):
            for c in range(3):
                if self.dtype == float:
                    ddr[c](psit_G,d_G)
                    axpy(0.5*f, d_G**2, taut_G) #taut_G += 0.5*f * d_G**2
                else:
                    ddr[c](psit_G,d_G,self.phase_cd)
                    taut_G += 0.5* f * (d_G * np.conjugate(d_G)).real

    def create_atomic_orbitals(self, nao, nuclei):
        """Initialize the wave functions from atomic orbitals.

        Create nao atomic orbitals."""
        raise DeprecationWarning
        # Allocate space for wave functions, occupation numbers,
        # eigenvalues and projections:
        self.allocate(nao)
        self.psit_nG = self.gd.zeros(nao, self.dtype)

        # fill in the atomic orbitals:
        nao0 = 0
        for nucleus in nuclei:
            nao1 = nao0 + nucleus.get_number_of_atomic_orbitals()
            nucleus.create_atomic_orbitals(self.psit_nG[nao0:nao1], self.k)
            nao0 = nao1
        assert nao0 == nao

    def create_random_orbitals(self, nbands):
        """Initialize all the wave functions from random numbers"""
        xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.allocate(nbands)
        self.psit_nG = self.gd.zeros(nbands, self.dtype)
        self.random_wave_functions(self.psit_nG)                   

