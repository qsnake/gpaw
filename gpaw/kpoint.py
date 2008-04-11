# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class."""

from math import pi, sqrt
from cmath import exp

import numpy as npy
from numpy.random import random, seed

from gpaw import mpi
from gpaw.operators import Gradient
from gpaw.transformers import Transformer
from gpaw.utilities.blas import axpy, rk, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.lapack import diagonalize

from gpaw.polynomial import Polynomial

class KPoint:
    """Class for a singel k-point.

    The KPoint class takes care of all wave functions for a
    certain k-point and a certain spin.

    Attributes
    ==========
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

    Parallel stuff
    ==============
    comm: Communicator object
        MPI-communicator for domain.
    root: int
        Rank of the CPU that does the matrix diagonalization of
        H_nn and the Cholesky decomposition of S_nn.
    """
    
    def __init__(self, nuclei, gd, weight, s, k, u, k_c, dtype, timer=None):
        """Construct k-point object.

        Parameters
        ==========
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

        self.nuclei = nuclei
        self.weight = weight
        self.dtype = dtype
        self.timer = timer
        
        self.phase_cd = npy.ones((3, 2), complex)
        if dtype == float:
            # Gamma-point calculation:
            self.k_c = None
        else:
            sdisp_cd = gd.domain.sdisp_cd
            for c in range(3):
                for d in range(2):
                    self.phase_cd[c, d] = exp(2j * pi *
                                              sdisp_cd[c, d] * k_c[c])
            self.k_c = k_c

        self.s = s  # spin index
        self.k = k  # k-point index
        self.u = u  # combined spin and k-point index

        self.set_grid_descriptor(gd)

        # Only one of these two will be used:
        self.psit_nG = None  # wave functions on 3D grid
        self.C_nm = None     # LCAO coefficients for wave functions
        
    def set_grid_descriptor(self, gd):
        self.gd = gd
        # Which CPU does overlap-matrix Cholesky-decomposition and
        # Hamiltonian-matrix diagonalization?
        self.comm = self.gd.comm
        self.root = self.u % self.comm.size

    def allocate(self, nbands):
        """Allocate arrays."""
        self.nbands = nbands
        self.eps_n = npy.empty(nbands)
        self.f_n = npy.empty(nbands)
        
    def add_extra_bands(self, nbands, nao):
        """Add extra states.

        If the number of atomic orbitals is less than the desired
        number of bands, then extra random wave functions are added.
        """

        eps_n = self.eps_n
        f_n = self.f_n

        self.allocate(nbands)

        self.eps_n[:nao] = eps_n
        self.f_n[:nao] = f_n

        # Generate random wave functions:
        self.eps_n[nao:] = self.eps_n[nao - 1] + 0.5
        self.f_n[nao:] = 0.0
        self.random_wave_functions(self.psit_nG[nao:])
                    
    def random_wave_functions(self, psit_nG):
        """Generate random wave functions"""

        gd1 = self.gd.coarsen()
        gd2 = gd1.coarsen()

        psit_G1 = gd1.empty(dtype=self.dtype)
        psit_G2 = gd2.empty(dtype=self.dtype)

        interpolate2 = Transformer(gd2, gd1, 1, self.dtype).apply
        interpolate1 = Transformer(gd1, self.gd, 1, self.dtype).apply

        shape = tuple(gd2.n_c)

        scale = sqrt(12 / npy.product(gd2.domain.cell_c))

        seed(4 + mpi.rank)

        for psit_G in psit_nG:
            if self.dtype == float:
                psit_G2[:] = (random(shape) - 0.5) * scale
            else:
                psit_G2.real = (random(shape) - 0.5) * scale
                psit_G2.imag = (random(shape) - 0.5) * scale

            interpolate2(psit_G2, psit_G1, self.phase_cd)
            interpolate1(psit_G1, psit_G, self.phase_cd)

    def add_to_density(self, nt_G, use_lcao):
        """Add contribution to pseudo electron-density."""
        self.add_to_density_with_occupation(nt_G, use_lcao, self.f_n)
        
    def add_to_density_with_occupation(self, nt_G, use_lcao, f_n):
        """Add contribution to pseudo electron-density. Do not use the standard
        occupation numbers, but ones given with argument f_n."""
        if use_lcao:
            psit_G = self.gd.empty(dtype=self.dtype)
            nuclei = [nucleus
                      for nucleus in self.nuclei if nucleus.phit_i is not None]
            for f, C_m in zip(f_n, self.C_nm):
                psit_G[:] = 0.0
                m1 = 0
                for nucleus in nuclei:
                    niao = nucleus.get_number_of_atomic_orbitals()
                    m2 = m1 + niao
                    nucleus.phit_i.add(psit_G, C_m[m1:m2], self.k)
                    m1 = m2
                nt_G += f * (psit_G * psit_G.conj()).real
        else:
            if self.dtype == float:
                for psit_G, f in zip(self.psit_nG, f_n):
                    axpy(f, psit_G**2, nt_G)  # nt_G += f * psit_G**2
            else:
                for psit_G, f in zip(self.psit_nG, f_n):
                    nt_G += f * (psit_G * npy.conjugate(psit_G)).real

        # Hack used in delta-scf calculations:
        if hasattr(self, 'ft_omn'):
            for ft_mn in self.ft_omn:
                for ft_n, psi_m in zip(ft_mn, self.psit_nG):
                    for ft, psi_n in zip(ft_n, self.psit_nG):
                        if abs(ft) > 1.e-12:
                            nt_G += (npy.conjugate(psi_m) *
                                     ft * psi_n).real

    def add_to_kinetic_density(self, taut_G):
        """Add contribution to pseudo kinetic energy density."""

        ddr = [Gradient(self.gd, c).apply for c in range(3)]
        d_G = self.gd.empty()
        for f,psit_G in zip(self.f_n,self.psit_nG):
            for c in range(3):
                ddr[c](psit_G,d_G)
                if self.dtype == float:
                    axpy(f, d_G[c]**2, taut_G) #taut_G += f * d_G[c]**2
                else:
                    taut_G += f * (d_G * npy.conjugate(d_G)).real

    def calculate_wave_functions_from_lcao_coefficients(self, nbands):
        self.nbands = nbands
        self.psit_nG = self.gd.zeros(nbands, dtype=self.dtype)
        psit_nG = self.psit_nG[:len(self.C_nm)]
        m1 = 0
        for nucleus in self.nuclei:
            niao = nucleus.get_number_of_atomic_orbitals()
            m2 = m1 + niao
            if nucleus.phit_i is not None:
                nucleus.phit_i.add(psit_nG, self.C_nm[:, m1:m2].copy(), self.k)
            m1 = m2

    def create_atomic_orbitals(self, nao, nuclei):
        """Initialize the wave functions from atomic orbitals.

        Create nao atomic orbitals."""

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

        self.allocate(nbands)
        self.psit_nG = self.gd.zeros(nbands, self.dtype)
        self.random_wave_functions(self.psit_nG)                   


