# Copyright (C) 2010  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange with k-point sampling.
"""
from math import pi, sqrt, ceil

import numpy as np

from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.xc.functional import XCFunctional
from gpaw.poisson import PoissonSolver
from gpaw.utilities import hartree, pack, pack2, unpack, unpack2, packed_index
from gpaw.utilities.tools import symmetrize
from gpaw.atom.configurations import core_states
from gpaw.lfc import LFC
from gpaw.gaunt import make_gaunt
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.fd_operators import Laplace
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.eigensolvers.rmm_diis import RMM_DIIS


class HybridXC(XCFunctional):
    orbital_dependent = True
    def __init__(self, name, hybrid=None, xc=None, finegrid=False,
                 alpha=None):
        """Mix standard functionals with exact exchange.

        name: str
            Name of hybrid functional.
        hybrid: float
            Fraction of exact exchange.
        xc: str or XCFunctional object
            Standard DFT functional with scaled down exchange.
        finegrid: boolean
            Use fine grid for energy functional evaluations?
        """

        if name == 'EXX':
            assert hybrid is None and xc is None
            hybrid = 1.0
            xc = XC(XCNull())
        elif name == 'PBE0':
            assert hybrid is None and xc is None
            hybrid = 0.25
            xc = XC('HYB_GGA_XC_PBEH')
        elif name == 'B3LYP':
            assert hybrid is None and xc is None
            hybrid = 0.2
            xc = XC('HYB_GGA_XC_B3LYP')
            
        if isinstance(xc, str):
            xc = XC(xc)

        self.hybrid = hybrid
        self.xc = xc
        self.type = xc.type
        self.alpha = alpha
        self.exx = 0.0
        
        XCFunctional.__init__(self, name)

    def get_setup_name(self):
        return 'PBE'

    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg=None, rnablaY_Lv=None,
                         tau_sg=None, dedtau_sg=None):
        return self.xc.calculate_radial(rgd, n_sLg, Y_L, v_sg,
                                        dndr_sLg, rnablaY_Lv)
    
    def initialize(self, density, hamiltonian, wfs, occupations):
        self.xc.initialize(density, hamiltonian, wfs, occupations)
        self.kpt_comm = wfs.kpt_comm
        self.nspins = wfs.nspins
        self.setups = wfs.setups
        self.density = density
        self.kpt_u = wfs.kpt_u
        
        self.ghat = LFC(density.gd,
                        [setup.ghat_l for setup in density.setups],
                        integral=np.sqrt(4 * pi), forces=True)
        self.gd = density.gd
        self.finegd = self.ghat.gd
        self.kd = wfs.kd

        N_c = self.gd.N_c
        N = self.gd.N_c.prod()
        vol = self.gd.dv * N
        
        if self.alpha is None:
            self.alpha = 6 * vol**(2 / 3.0) / pi**2
            
        self.gamma = (vol / (2 * pi)**2 * sqrt(pi / self.alpha) *
                      self.kd.nbzkpts)
        ecut = 0.5 * pi**2 / (self.gd.h_cv**2).sum(1).max()

        if self.kd.N_c is None:
            self.bzk_kc = np.zeros((1, 3))
        else:
            bzk_kc = np.indices(self.kd.N_c).transpose((1, 2, 3, 0))
            bzk_kc.shape = (-1, 3)
            bzk_kc -= self.kd.N_c // 2
            self.bzk_kc = bzk_kc.astype(float) / self.kd.N_c
        
        self.pwd = PWDescriptor(ecut, self.gd, self.bzk_kc)
        for k_c, Gpk2_G in zip(self.bzk_kc, self.pwd.G2_qG):
            if k_c.any():
                self.gamma -= np.dot(np.exp(-self.alpha * Gpk2_G), Gpk2_G**-1)
            else:
                self.gamma -= np.dot(np.exp(-self.alpha * Gpk2_G[1:]),
                                     Gpk2_G[1:]**-1)
                
    def set_positions(self, spos_ac):
        if 0:#not self.finegrid:
            self.ghat.set_positions(spos_ac)
    
    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)
        return exc + self.exx

    def calculate_exx(self):
        for kpt in self.kpt_u:
            self.apply_orbital_dependent_hamiltonian(kpt, kpt.psit_nG)

    def apply(self, kpt1, kpt2):        
        k12_c = ((self.kd.ibzk_kc[kpt1.k] -
                  self.kd.ibzk_kc[kpt2.k]) + 0.4999) % 1 - 0.4999
        N_c = self.gd.N_c
        expikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k12_c / N_c).T)

        for q, k_c in enumerate(self.bzk_kc):
            if abs(k_c + k12_c).max() < 1e-9:
                q0 = q
                break
            
        Gpk2_G = self.pwd.G2_qG[q0]
        if kpt1 is kpt2:
            Gpk2_G = Gpk2_G.copy()
            Gpk2_G[0] = 1.0 / self.gamma
        N = N_c.prod()
        vol = self.gd.dv * N
        nspins = self.nspins
        
        deg = 2 // self.nspins
        
        for n1, psit1_R in enumerate(kpt1.psit_nG):
            f1 = kpt1.f_n[n1]
            for n2, psit2_R in enumerate(kpt2.psit_nG):
                f2 = kpt2.f_n[n2]

                nt_R = self.calculate_pair_density(n1, n2,
                                                   psit1_R, psit2_R,
                                                   )
                nt_G = self.pwd.fft(nt_R * expikr_R / N)
                vt_G = nt_G.copy()
                vt_G *= -pi * vol / Gpk2_G
                e = f1 * f2 * np.vdot(nt_G, vt_G).real * nspins
                self.exx += e
                self.ekin -= 2 * e
                
                vt_R = self.pwd.ifft(vt_G).conj() * expikr_R * N / vol
                if kpt1 is kpt2 and n1 == n2:
                    kpt1.vt_nG[n1] = f1 * vt_R

                kpt1.Htpsit_nG[n1] += f2 * nspins * psit2_R * vt_R
                kpt2.Htpsit_nG[n2] += f1 * nspins * psit1_R * vt_R.conj()

    def calculate_pair_density(self, n1, n2, psit1_G, psit2_G):#, P_ani):
        Q_aL = {}
        if 0:#for a, P_ni in P_ani.items():
            P1_i = P_ni[n1]
            P2_i = P_ni[n2]
            D_ii = np.outer(P1_i, P2_i.conj()).real
            D_p = pack(D_ii, tolerance=1e30)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)
            
        nt_G = psit1_G.conj() * psit2_G

        #rhot_g = nt_g.copy()
        #self.ghat.add(rhot_g, Q_aL)

        return nt_G#, rhot_g

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        if not hasattr(kpt, 'vxx_ani'):
            return

        if self.gd.comm.rank > 0:
            H_nn[:] = 0.0
            
        nocc = 1
        if 0:#for a, P_ni in kpt.P_ani.items():
            H_nn[:nocc, :nocc] += symmetrize(np.inner(P_ni[:nocc],
                                                      kpt.vxx_ani[a]))
        self.gd.comm.sum(H_nn)
        
        H_nn[:nocc, nocc:] = 0.0
        H_nn[nocc:, :nocc] = 0.0

    def add_correction(self, kpt, psit_xG, Htpsit_xG, P_axi, c_axi, n_x,
                       calculate_change=False):
        if kpt.f_n is None:
            return

        if calculate_change:
            for x, n in enumerate(n_x):
                Htpsit_xG[x] += kpt.vt_nG[n] * psit_xG[x]
                #for a, P_xi in P_axi.items():
                #    c_axi[a][x] += np.dot(kpt.vxx_anii[a][n], P_xi[x])
        else:
            if 0:#for a, c_xi in c_axi.items():
                c_xi[:nocc] += kpt.vxx_ani[a]
        
    def rotate(self, kpt, U_nn):
        if kpt.f_n is None:
            return

        gemm(1.0, kpt.vt_nG.copy(), U_nn, 0.0, kpt.vt_nG)
        #for v_ni in kpt.vxx_ani.values():
        #    gemm(1.0, v_ni.copy(), U_nn, 0.0, v_ni)
        #for v_nii in kpt.vxx_anii.values():
        #    gemm(1.0, v_nii.copy(), U_nn, 0.0, v_nii)





class HybridRMMDIIS(RMM_DIIS):
    def update(self, wfs, xc, ham):
        kd = wfs.kd
        assert kd.mynks * kd.comm.size == kd.nks
        #assert kd.nspins == 1
        assert kd.symmetry is None

        B = kd.mynks
        rank = kd.comm.rank
        P = kd.comm.size

        for u1, kpt1 in enumerate(wfs.kpt_u):
            wfs.apply_pseudo_hamiltonian(kpt1, ham, kpt1.psit_nG,
                                         kpt1.Htpsit_nG)
        if wfs.kpt_u[0].f_n is None:
            return

        xc.exx = 0.0
        xc.ekin = 0.0
        for u1, kpt1 in enumerate(wfs.kpt_u):
            for u2, kpt2 in enumerate(wfs.kpt_u):
                if kpt1.s == kpt2.s:
                    xc.apply(kpt1, kpt2)
                
    def subspace_diagonalize(self, ham, wfs, kpt, rotate=True):
        if self.band_comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        if not hasattr(kpt, 'vt_nG'):
            for kpt1 in wfs.kpt_u:
                kpt1.vt_nG = wfs.gd.empty(wfs.bd.mynbands, wfs.dtype)
                kpt1.Htpsit_nG = wfs.gd.empty(wfs.bd.mynbands, wfs.dtype)

        if kpt is wfs.kpt_u[0]:
            self.update(wfs, ham.xc, ham)

        psit_nG = kpt.psit_nG
        self.Htpsit_nG = kpt.Htpsit_nG
        P_ani = kpt.P_ani

        def H(psit_xG):
            return kpt.Htpsit_nG

        def dH(a, P_ni):
            return np.dot(P_ni, unpack(ham.dH_asp[a][kpt.s]))

        self.timer.start('calc_matrix')
        H_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
                                                       H, dH)
        ham.xc.correct_hamiltonian_matrix(kpt, H_nn)
        self.timer.stop('calc_matrix')

        self.timer.start('Subspace diag')
        diagonalization_string = repr(self.ksl)
        wfs.timer.start(diagonalization_string)
        self.ksl.diagonalize(H_nn, kpt.eps_n)
        # H_nn now contains the result of the diagonalization.
        wfs.timer.stop(diagonalization_string)

        if not rotate:
            self.timer.stop('Subspace diag')
            return

        self.timer.start('rotate_psi')
        kpt.psit_nG = self.operator.matrix_multiply(H_nn, psit_nG, P_ani)
        if self.keep_htpsit:
            kpt.Htpsit_nG = self.operator.matrix_multiply(H_nn, kpt.Htpsit_nG)

        # Rotate orbital dependent XC stuff:
        ham.xc.rotate(kpt, H_nn)

        self.timer.stop('rotate_psi')

        self.timer.stop('Subspace diag')

    def estimate_memory(self, mem, gd, dtype, mynbands, nbands):
        gridmem = gd.bytecount(dtype)

        keep_htpsit = self.keep_htpsit and (mynbands == nbands)

        if keep_htpsit:
            mem.subnode('Htpsit', nbands * gridmem)
        else:
            mem.subnode('No Htpsit', 0)

        # mem.subnode('U_nn', nbands*nbands*mem.floatsize)
        mem.subnode('eps_n', nbands*mem.floatsize)
        mem.subnode('Preconditioner', 4 * gridmem)
        mem.subnode('Work', gridmem)

