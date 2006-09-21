# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a ``KPoint`` class."""

from math import pi
from cmath import exp

import Numeric as num
import LinearAlgebra as linalg
from multiarray import innerproduct as inner # avoid the dotblas version!

from gpaw.utilities.blas import axpy, rk, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import unpack
from gpaw.utilities.timing import Timer
from gpaw.operators import Gradient


class KPoint:
    """Class for a singel **k**-point.

    The ``KPoint`` class takes care of all wave functions for a
    certain **k**-point and a certain spin."""
    
    def __init__(self, gd, weight, s, k, u, k_c, typecode):
        """Construct **k**-point object.

        Parameters:
         ============ =======================================================
         ``gd``       Descriptor for wave-function grid.
         ``weight``   Weight of this **k**-point.
         ``s``        Spin index: up or down (0 or 1).
         ``k``        **k**-point index.
         ``u``        Combined spin and **k**-point index.
         ``k_c``      scaled **k**-point vector (coordinates scaled to
                      [-0.5:0.5] interval).
         ``typecode`` Data type of wave functions (``Float`` or ``Complex``).
         ============ =======================================================

        Attributes:
         ============= =======================================================
         ``phase_cd``  Bloch phase-factors for translations - axis ``c=0,1,2``
                       and direction ``d=0,1``.
         ``eps_n``     Eigenvalues.
         ``f_n``       Occupation numbers.
         ``H_nn``      Hamiltonian matrix.
         ``S_nn``      Overlap matrix.
         ``psit_nG``   Wave functions.
         ``Htpsit_nG`` Pseudo-part of the Hamiltonian applied to the wave
                       functions.
         ``timer``     ``Timer`` object.
         ``nbands``    Number of bands.
         ============= =======================================================

        Parallel stuff:
         ======== =======================================================
         ``comm`` MPI-communicator for domain.
         ``root`` Rank of the CPU that does the matrix diagonalization of
                  ``H_nn`` and the Cholesky decomposition of ``S_nn``.
         ======== =======================================================
        """

        self.gd = gd
        self.weight = weight
        self.typecode = typecode
        
        self.phase_cd = num.ones((3, 2), num.Complex)
        if typecode == num.Float:
            # Gamma-point calculation:
            self.k_c = None
        else:
            sdisp_cd = self.gd.domain.sdisp_cd
            for c in range(3):
                for d in range(2):
                    self.phase_cd[c, d] = exp(2j * pi *
                                              sdisp_cd[c, d] * k_c[c])
            self.k_c = k_c

        self.s = s  # spin index
        self.k = k  # k-point index
        self.u = u  # combined spin and k-point index

        # Which CPU does overlap-matrix Cholesky-decomposition and
        # Hamiltonian-matrix diagonalization?
        self.comm = self.gd.comm
        self.root = u % self.comm.size
        
        self.psit_nG = None
        self.Htpsit_nG = None

        self.timer = Timer()
        
    def allocate(self, nbands):
        """Allocate arrays."""
        self.nbands = nbands
        self.eps_n = num.zeros(nbands, num.Float)
        self.f_n = num.ones(nbands, num.Float) * self.weight
        self.H_nn = num.zeros((nbands, nbands), self.typecode)
        self.S_nn = num.zeros((nbands, nbands), self.typecode)

    def diagonalize(self, kin, vt_sG, my_nuclei, exx):
        """Subspace diagonalization of wave functions.

        First, the Hamiltonian (defined by ``kin``, ``vt_sG``, and
        ``my_nuclei``) is applied to the wave functions, then the
        ``H_nn`` matrix is calculated and diagonalized, and finally,
        the wave functions are rotated.  Also the projections
        ``P_uni`` (an attribute of the nuclei) are rotated.
        """

        kin.apply(self.psit_nG, self.Htpsit_nG, self.phase_cd)
        self.Htpsit_nG += self.psit_nG * vt_sG[self.s]
        if exx is not None:
            exx.adjust_hamiltonian(self.psit_nG, self.Htpsit_nG, self.nbands,
                                   self.f_n, self.u, self.s)
        r2k(0.5 * self.gd.dv, self.psit_nG, self.Htpsit_nG, 0.0, self.H_nn)
        # XXX Do EXX here XXX
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            self.H_nn += num.dot(P_ni, num.dot(unpack(nucleus.H_sp[self.s]),
                                               cc(num.transpose(P_ni))))
            if exx is not None:
                exx.adjust_hamitonian_matrix(self.H_nn, P_ni, nucleus, self.s)

        self.comm.sum(self.H_nn, self.root)

        if self.comm.rank == self.root:
            info = diagonalize(self.H_nn, self.eps_n)
            if info != 0:
                raise RuntimeError, 'Very Bad!!'

        self.comm.broadcast(self.H_nn, self.root)
        self.comm.broadcast(self.eps_n, self.root)

        # Rotate psit_nG:
        # We should block this so that we can use a smaller temp !!!!!
        temp = num.array(self.psit_nG)
        gemm(1.0, temp, self.H_nn, 0.0, self.psit_nG)
        
        # Rotate Htpsit_nG:
        temp[:] = self.Htpsit_nG
        gemm(1.0, temp, self.H_nn, 0.0, self.Htpsit_nG)
        
        # Rotate P_ani:
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            temp_ni = P_ni.copy()
            gemm(1.0, temp_ni, self.H_nn, 0.0, P_ni)
        
    def adjust_number_of_bands(self, nbands, random_wave_function_generator):
        """Adjust the number of states.

        If we are starting from atomic orbitals, then the desired
        number of bands (``nbands``) will most likely differ from the
        number of current atomic orbitals (``self.nbands``).  If this
        is the case, then new arrays are allocated:

        * Too many bands: The bands with the lowest eigenvalues are
          used.
        * Too few bands: Extra random wave functions are added.
        """
        
        if nbands == self.nbands:
            return

        nao = self.nbands  # number of atomic orbitals
        nmin = min(nao, nbands)

        tmp_nG = self.psit_nG
        self.psit_nG = self.gd.new_array(nbands, self.typecode)
        self.psit_nG[:nmin] = tmp_nG[:nmin]

        tmp_nG = self.Htpsit_nG
        self.Htpsit_nG = self.gd.new_array(nbands, self.typecode)
        self.Htpsit_nG[:nmin] = tmp_nG[:nmin]
        del tmp_nG

        tmp_n = self.eps_n
        self.allocate(nbands)
        self.eps_n[:nmin] = tmp_n[:nmin]

        extra = nbands - nao
        if extra > 0:
            # Generate random wave functions:
            self.eps_n[nao:] = self.eps_n[nao - 1] + 0.5
            for psit_G in self.psit_nG[nao:]:
                random_wave_function_generator.generate(psit_G, self.phase_cd)
        
    def calculate_residuals(self, pt_nuclei, converge_all=False):
        """Calculate wave function residuals.

        On entry, ``Htpsit_nG`` contains the soft part of the
        Hamiltonian applied to the wave functions.  After this call,
        ``Htpsit_nG`` holds the residuals::

          ^  ~        ^  ~   
          H psi - eps S psi =
                                _ 
              ~  ~         ~   \   ~a    a           a     ~a   ~
              H psi - eps psi + )  p  (dH    - eps dS    )<p  |psi>
                               /_   i1   i1i2        i1i2   i2
                              ai1i2

                                
        The size of the residuals is returned.
        
        Parameters:
         ================ ================================================
         ``pt_nuclei``    List of nuclei that have part of their projector
                          functions in this domain.
         ``converge_all`` Converge all wave functions or just occupied.
         ================ ================================================
        """
        
        R_nG = self.Htpsit_nG
        # optimize XXX 
        for R_G, eps, psit_G in zip(R_nG, self.eps_n, self.psit_nG):
            R_G -= eps * psit_G

        for nucleus in pt_nuclei:
            nucleus.adjust_residual(R_nG, self.eps_n, self.s, self.u, self.k)

        error = 0.0
        for R_G, f in zip(R_nG, self.f_n):
            weight = f
            if converge_all: weight = 1.
            error += weight * real(num.vdot(R_G, R_G))

        return error
        
    def orthonormalize(self, my_nuclei):
        """Orthogonalize wave functions."""
        S_nn = self.S_nn

        # Fill in the lower triangle:
        rk(self.gd.dv, self.psit_nG, 0.0, S_nn)
        
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]

            S_nn += num.dot(P_ni, cc(inner(nucleus.setup.O_ii, P_ni)))
        
        self.comm.sum(S_nn, self.root)

        if self.comm.rank == self.root:
            # inverse returns a non-contigous matrix - grrrr!  That is
            # why there is a copy.  Should be optimized with a
            # different lapack call to invert a triangular matrix XXXXX
            S_nn[:] = linalg.inverse(
                linalg.cholesky_decomposition(S_nn)).copy()

        self.comm.broadcast(S_nn, self.root)
        
        # This step will overwrite the Htpsit_nG array!
        gemm(1.0, self.psit_nG, S_nn, 0.0, self.Htpsit_nG)
        self.psit_nG, self.Htpsit_nG = self.Htpsit_nG, self.psit_nG  # swap

        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            gemm(1.0, P_ni.copy(), S_nn, 0.0, P_ni)

    def add_to_density(self, nt_G):
        """Add contribution to pseudo electron-density."""
        if self.typecode is num.Float:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                nt_G += f * psit_G**2
        else:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                nt_G += f * (psit_G * num.conjugate(psit_G)).real
                
    def add_to_kinetic_electron_density(self, taut_G, ddr):
        """Add contribution to pseudo kinetic electron density."""
        for psit_G, f in zip(self.psit_nG, self.f_n):
            d_G = num.zeros(psit_G.shape, num.Float)
            for c in range(3):
                ddr[c](psit_G,d_G)
                if self.typecode is num.Float:
                    taut_G += f * d_G[c]**2
                else:
                    taut_G += f * (d_G * num.conjugate(d_G)).real
                
    def rmm_diis(self, pt_nuclei, preconditioner, kin, vt_sG):
        """Improve the wave functions.

        Take two steps along the preconditioned residuals.  Step
        lengths are optimized for the first step and reused for the
        seconf."""
        
        vt_G = vt_sG[self.s]
        for n in range(self.nbands):
            R_G = self.Htpsit_nG[n]

            dR_G = num.zeros(R_G.shape, self.typecode)

            pR_G = preconditioner(R_G, self.phase_cd, self.psit_nG[n],
                                  self.k_c)
            
            kin.apply(pR_G, dR_G, self.phase_cd)

            dR_G += vt_G * pR_G

            dR_G -= self.eps_n[n] * pR_G

            for nucleus in pt_nuclei:
                nucleus.adjust_residual2(pR_G, dR_G, self.eps_n[n],
                                         self.s, self.k)
            
            RdR = self.comm.sum(real(num.vdot(R_G, dR_G)))
            dRdR = self.comm.sum(real(num.vdot(dR_G, dR_G)))
            lam = -RdR / dRdR

            R_G *= 2.0 * lam
            axpy(lam**2, dR_G, R_G)  # R_G += lam**2 * dR_G
            self.psit_nG[n] += preconditioner(R_G, self.phase_cd,
                                              self.psit_nG[n], self.k_c)

    def create_atomic_orbitals(self, nao, nuclei):
        """Initialize the wave functions from atomic orbitals.

        Create ``nao`` atomic orbitals."""
        
        # Allocate space for wave functions, occupation numbers,
        # eigenvalues and projections:
        self.allocate(nao)
        self.psit_nG = self.gd.new_array(nao, self.typecode)
        self.Htpsit_nG = self.gd.new_array(nao, self.typecode)
        
        # fill in the atomic orbitals:
        nao0 = 0
        for nucleus in nuclei:
            nao1 = nao0 + nucleus.get_number_of_atomic_orbitals()
            nucleus.create_atomic_orbitals(self.psit_nG[nao0:nao1], self.k)
            nao0 = nao1
        assert nao0 == nao

    def apply_h(self, pt_nuclei, kin, vt_sG, psit, Htpsit):
        """Applies the Hamiltonian to the wave function psit"""

        Htpsit[:] = 0.0
        kin.apply(psit, Htpsit, self.phase_cd)
        Htpsit += psit * vt_sG[self.s]
        
        for nucleus in pt_nuclei:
            #apply the non-local part
            nucleus.apply_hamiltonian(psit, Htpsit, self.s, self.k)

    def apply_s(self, pt_nuclei, psit, Spsit):
        """Applies the overlap operator to the wave function psit"""

        Spsit[:] = psit[:]
        for nucleus in pt_nuclei:
            #apply the non-local part
            nucleus.apply_overlap(psit, Spsit, self.k)
            
