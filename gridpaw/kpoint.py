# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import generators
from math import pi

import Numeric as num
import LinearAlgebra as linalg

from gridpaw.utilities.blas import rk, r2k, gemm
from gridpaw.utilities.complex import cc, real
from gridpaw.utilities.lapack import diagonalize
from gridpaw.utilities import scale_add_to, square_scale_add_to, unpack
from gridpaw.utilities.timing import Timer
import gridpaw.utilities.mpi as mpi
from gridpaw.operators import Gradient


class KPoint:
    def __init__(self, gd, weight, s, k, u, k_i, typecode):
        self.gd = gd
        self.weight = weight
        self.Htpsit_nG = None
        self.typecode = typecode
        
        if typecode == num.Float:
            # Gamma-point calculation:
            self.phases = num.ones(6, num.Float) # XXX or None?
            self.k_i = None
        else:
            displacements = self.gd.domain.get_displacements()
            self.phases = num.exp(2j * pi * num.dot(displacements, k_i))
            self.k_i = k_i

        self.s = s
        self.k = k
        self.u = u

        # Which cpu does overlap-matrix Cholesky-decomposition and
        # Hamiltonian-matrix diagonalization?
        self.comm = self.gd.domain.comm
        self.root = u % self.comm.size
        
        self.psit_nG = None

        self.timer = Timer()
        
    def allocate(self, nbands, wavefunctions=True):
        self.nbands = nbands
        if wavefunctions:
            self.allocate_wavefunctions(nbands)
        self.eps_n = num.zeros(nbands, num.Float)
        self.f_n = num.ones(nbands, num.Float) * self.weight
        self.H_n1n2 = num.zeros((nbands, nbands), self.typecode)
        self.S_n1n2 = num.zeros((nbands, nbands), self.typecode)

    def allocate_wavefunctions(self, nbands):
        shape = (nbands,) + tuple(self.gd.myng)
        self.psit_nG = num.zeros(shape, self.typecode)
        self.Htpsit_nG = num.zeros(shape, self.typecode)
        
    def diagonalize(self, kin, vt_sG, my_nuclei, nbands):
        """Diagonalize wave functions.

        """

        # Put in some yields ???? XXXX
        vt_G = vt_sG[self.s]
        self.timer.start('apply')
        kin.apply(self.psit_nG, self.Htpsit_nG, self.phases)
        self.timer.stop('apply')
        self.timer.start('pot')
        self.Htpsit_nG += self.psit_nG * vt_G
        self.timer.stop('pot')
        self.timer.start('H')
        self.H_n1n2[:] = 0.0  # is that necessary? XXXX
        r2k(0.5 * self.gd.dv, self.psit_nG, self.Htpsit_nG, 0.0, self.H_n1n2)
        self.timer.stop('H')

        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            self.H_n1n2 += num.dot(P_ni, num.dot(unpack(nucleus.H_sp[self.s]),
                                               cc(num.transpose(P_ni))))

        self.comm.sum(self.H_n1n2, self.root)
        
        self.timer.start('diag')
        if self.comm.rank == self.root:
            info = diagonalize(self.H_n1n2, self.eps_n)
            if info != 0:
                raise RuntimeError, 'Very Bad!!'
        self.timer.stop('diag')
        
        if mpi.parallel:
            self.comm.broadcast(self.H_n1n2, self.root)
            self.comm.broadcast(self.eps_n, self.root)

        # Rotate psit_nG:
        # We should block this so that we can use a smaller temp !!!!!
        self.timer.start('rot1')
        temp = num.array(self.psit_nG)
        gemm(1.0, temp, self.H_n1n2, 0.0, self.psit_nG)
        self.timer.stop('rot1')
        
        # Rotate Htpsit_nG:
        self.timer.start('rot2')
        temp[:] = self.Htpsit_nG
        gemm(1.0, temp, self.H_n1n2, 0.0, self.Htpsit_nG)
        self.timer.stop('rot2')
        
        # Rotate P_ani:
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            temp_ni = P_ni.copy()
            gemm(1.0, temp_ni, self.H_n1n2, 0.0, P_ni)
        
        if nbands != self.nbands:
            self.timer.start('extra')
            nao = self.nbands
            
            # Hold on to atomic stuff before reallocating:
            psitao_nG = self.psit_nG
            Htpsitao_nG = self.Htpsit_nG
            epsao_n = self.eps_n
            Hao_n1n2 = self.H_n1n2
            
            self.allocate(nbands)

            nmin = min(nao, nbands)
            self.psit_nG[:nmin] = psitao_nG[:nmin]
            self.Htpsit_nG[:nmin] = Htpsitao_nG[:nmin]
            self.eps_n[:nmin] = epsao_n[:nmin]
            self.H_n1n2[:nmin, :nmin] = Hao_n1n2[:nmin, :nmin]
            del psitao_nG, Htpsitao_nG, epsao_n, Hao_n1n2

            extra = nbands - nao
            if extra > 0:
                self.eps_n[nao:] = self.eps_n[nao - 1] + 0.5
                self.H_n1n2.flat[nao * (nbands + 1)::nbands + 1] = 1.0
                slice = self.psit_nG[nao:]
                grad = Gradient(self.gd, 0, typecode=self.typecode).apply
                grad(self.psit_nG[:extra], slice, self.phases)
            self.timer.stop('extra')
        
    def calculate_residuals(self, p_nuclei):
        R_n = self.Htpsit_nG
        # optimize XXX 
        for R, eps, psit_G in zip(R_n, self.eps_n, self.psit_nG):
            R -= eps * psit_G
        
        for nucleus in p_nuclei:
            nucleus.adjust_residual(R_n, self.eps_n, self.s, self.u, self.k_i)

        error = 0.0
        for R, f in zip(R_n, self.f_n):
            error += f * real(num.dot(cc(R).flat, R.flat))

        return error
        
    def orthonormalize(self, my_nuclei):
        S = self.S_n1n2

        # Fill in the lower triangle:
        rk(self.gd.dv, self.psit_nG, 0.0, S)
        
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            S += num.dot(P_ni, cc(num.innerproduct(nucleus.setup.O_i1i2, P_ni)))
        
        self.comm.sum(S, self.root)

        yield None

        if self.comm.rank == self.root:
            # inverse returns a non-contigous matrix - grrrr!  That is
            # why there is a copy.  Should be optimized with a
            # different lapack call to invert a triangular matrix XXXXX
            S = linalg.inverse(
                linalg.cholesky_decomposition(S)).copy()

        yield None

        if mpi.parallel:
            self.comm.broadcast(S, self.root)

        # This step will overwrite the Htpsit_nG array!
        gemm(1.0, self.psit_nG, S, 0.0, self.Htpsit_nG)
        self.psit_nG, self.Htpsit_nG = self.Htpsit_nG, self.psit_nG

        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[self.u]
            gemm(1.0, P_ni.copy(), S, 0.0, P_ni)

        yield None

    def add_to_density(self, nt_G):
        if self.typecode is num.Float:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                square_scale_add_to(psit_G, f, nt_G)
        else:
            for psit_G, f in zip(self.psit_nG, self.f_n):
                nt_G.flat[:] += f * real(psit_G.flat * cc(psit_G.flat))
                
    def rmm_diis(self, p_nuclei, preconditioner, kin, vt_sG):
        vt_G = vt_sG[self.s]
        for n in range(self.nbands):
            R_G = self.Htpsit_nG[n]

            dR_G = num.zeros(R_G.shape, self.typecode)

            pR_G = preconditioner(R_G, self.phases, self.psit_nG[n], self.k_i)
            
            kin.apply(pR_G, dR_G, self.phases)

            dR_G += vt_G * pR_G

            dR_G -= self.eps_n[n] * pR_G

            for nucleus in p_nuclei:
                nucleus.adjust_residual2(pR_G, dR_G, self.eps_n[n],
                                         self.s, self.k_i)
            
            RdR = self.comm.sum(real(num.dot(cc(R_G).flat, dR_G.flat)))
            dRdR = self.comm.sum(real(num.dot(cc(dR_G.flat), dR_G.flat)))
            lam = -RdR / dRdR

            R_G *= 2.0 * lam
            scale_add_to(dR_G, lam**2, R_G)
            self.psit_nG[n] += preconditioner(R_G, self.phases,
                                              self.psit_nG[n], self.k_i)

    def create_atomic_orbitals(self, nao, nuclei):
        # Allocate space for wave functions, occupation numbers,
        # eigenvalues and projections:
        self.allocate(nao)  # nao: number of atomic orbitals
        
        # fill in the atomic orbitals:
        nao0 = 0
        for nucleus in nuclei:
            nao1 = nao0 + nucleus.get_number_of_atomic_orbitals()
            nucleus.create_atomic_orbitals(self.psit_nG[nao0:nao1], self.k_i)
            nao0 = nao1
        assert nao0 == nao
