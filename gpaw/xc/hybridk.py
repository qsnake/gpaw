# Copyright (C) 2010  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange with k-point sampling."""

from math import pi, sqrt

import numpy as np

from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.xc.functional import XCFunctional
from gpaw.utilities import hartree, pack, unpack2, packed_index
from gpaw.lfc import LFC
from gpaw.wavefunctions.pw import PWDescriptor


class KPoint:
    def __init__(self, kd, kpt=None):
        """Helper class for parallelizing over k-points.

        Placeholder for wave functions, occupation numbers,
        projections, and global k-point index."""
        
        self.kd = kd
        
        if kpt is not None:
            self.psit_nG = kpt.psit_nG
            self.f_n = kpt.f_n
            self.P_ani = kpt.P_ani
            self.k = kpt.k
            self.s = kpt.s
            
        self.requests = []
        
    def next(self):
        """Create empty object.

        Data will be received from other processor."""
        
        kpt = KPoint(self.kd)

        # intialize array for receiving:
        kpt.psit_nG = np.empty_like(self.psit_nG)
        kpt.f_n = np.empty_like(self.f_n)

        # Total number of projector functions:
        I = sum([P_ni.shape[1] for P_ni in self.P_ani.values()])
        
        kpt.P_In = np.empty((I, len(kpt.f_n)), complex)

        kpt.P_ani = {}
        I1 = 0
        for a, P_ni in self.P_ani.items():
            I2 = I1 + P_ni.shape[1]
            kpt.P_ani[a] = kpt.P_In[I1:I2].T
            I1 = I2

        kpt.k = (self.k + 1) % self.kd.nibzkpts
        kpt.s = self.s
        
        return kpt
        
    def start_sending(self, rank):
        P_In = np.concatenate([P_ni.T for P_ni in self.P_ani.values()])
        self.requests += [
            self.kd.comm.send(self.psit_nG, rank, block=False, tag=1),
            self.kd.comm.send(self.f_n, rank, block=False, tag=2),
            self.kd.comm.send(P_In, rank, block=False, tag=3)]
        
    def start_receiving(self, rank):
        self.requests += [
            self.kd.comm.receive(self.psit_nG, rank, block=False, tag=1),
            self.kd.comm.receive(self.f_n, rank, block=False, tag=2),
            self.kd.comm.receive(self.P_In, rank, block=False, tag=3)]
        
    def wait(self):
        self.kd.comm.waitall(self.requests)
        self.requests = []
        

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
        self.nspins = wfs.nspins
        self.setups = wfs.setups
        self.density = density
        self.kpt_u = wfs.kpt_u
        
        self.ghat = LFC(density.gd,
                        [setup.ghat_l for setup in density.setups],
                        integral=np.sqrt(4 * pi), forces=True)

        self.gd = density.gd
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
            n = self.kd.N_c * 2 - 1
            bzk_kc = np.indices(n).transpose((1, 2, 3, 0))
            bzk_kc.shape = (-1, 3)
            bzk_kc -= self.kd.N_c - 1
            self.bzk_kc = bzk_kc.astype(float) / self.kd.N_c
        
        self.pwd = PWDescriptor(ecut, self.gd, self.bzk_kc)

        n = 0
        for k_c, Gpk2_G in zip(self.bzk_kc[:], self.pwd.G2_qG):
            if (k_c > -0.5).all() and (k_c <= 0.5).all(): #XXX???
                if k_c.any():
                    self.gamma -= np.dot(np.exp(-self.alpha * Gpk2_G),
                                         Gpk2_G**-1)
                else:
                    self.gamma -= np.dot(np.exp(-self.alpha * Gpk2_G[1:]),
                                         Gpk2_G[1:]**-1)
                n += 1

        assert n == self.kd.N_c.prod()
        
    def set_positions(self, spos_ac):
        self.ghat.set_positions(spos_ac)
    
    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

        # Add EXX contribution:
        return exc + self.exx

    def calculate_exx(self):
        """Non-selfconsistent calculation."""

        self.exx = 0.0
        K = self.kd.nibzkpts
        parallel = (self.kd.comm.size > self.nspins)
        
        for s in range(self.nspins):
            kpt1_q = [KPoint(self.kd, kpt) for kpt in self.kpt_u if kpt.s == s]
            kpt2_q = kpt1_q[:]

            k = (kpt1_q[-1].k + 1) % K
            rrank = self.kd.get_rank_and_index(s, k)[0]
            k = (kpt1_q[0].k - 1) % K
            srank = self.kd.get_rank_and_index(s, k)[0]
            
            for i in range(K // 2 + 1):
                if i < K // 2:
                    if parallel:
                        kpt = kpt2_q[-1].next()
                        kpt.start_receiving(rrank)
                        kpt2_q[0].start_sending(srank)
                    else:
                        kpt = kpt2_q[0]

                for kpt1, kpt2 in zip(kpt1_q, kpt2_q):
                    if 2 * i == K:
                        self.apply(kpt1, kpt2, invert=(kpt1.k > kpt2.k))
                    else:
                        self.apply(kpt1, kpt2)
                        self.apply(kpt1, kpt2, invert=True)

                if i < K // 2:
                    if parallel:
                        kpt.wait()
                        kpt2_q[0].wait()
                    kpt2_q.pop(0)
                    kpt2_q.append(kpt)

        self.exx = self.kd.comm.sum(self.exx)

        self.exx += self.calculate_paw_correction()
        
    def apply(self, kpt1, kpt2, invert=False):        
        k1_c = self.kd.ibzk_kc[kpt1.k]
        k2_c = self.kd.ibzk_kc[kpt2.k]
        if invert:
            k2_c = -k2_c
        k12_c = k1_c - k2_c

        N_c = self.gd.N_c
        expikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k12_c / N_c).T)

        for q, k_c in enumerate(self.bzk_kc):
            if abs(k_c + k12_c).max() < 1e-9:
                q0 = q
                break
            
        Gpk2_G = self.pwd.G2_qG[q0]
        if Gpk2_G[0] == 0:
            Gpk2_G = Gpk2_G.copy()
            Gpk2_G[0] = 1.0 / self.gamma

        P1_ani = kpt1.P_ani
        P2_ani = kpt2.P_ani
        if invert:
            P2_ani = dict([(a, P_ni.conj()) for a, P_ni in P2_ani.items()])

        N = N_c.prod()
        vol = self.gd.dv * N
        nspins = self.nspins

        same = (kpt1.k == kpt2.k)
        
        for n1, psit1_R in enumerate(kpt1.psit_nG):
            f1 = kpt1.f_n[n1]
            for n2, psit2_R in enumerate(kpt2.psit_nG):
                if same and n2 > n1:
                    continue
                
                f2 = kpt2.f_n[n2]

                if invert:
                    psit2_R = psit2_R.conj()

                nt_R = self.calculate_pair_density(n1, n2, psit1_R, psit2_R,
                                                   P1_ani, P2_ani)
                nt_G = self.pwd.fft(nt_R * expikr_R / N)
                vt_G = nt_G.copy()
                vt_G *= -pi * vol / Gpk2_G
                e = f1 * f2 * np.vdot(nt_G, vt_G).real * nspins * self.hybrid
                if same and n1 == n2:
                    e /= 2
                self.exx += e
                self.ekin -= 2 * e

                calculate_potential = not not not not not not not not not True
                if calculate_potential:
                    vt_R = self.pwd.ifft(vt_G).conj() * expikr_R * N / vol
                    if kpt1 is kpt2 and not invert and n1 == n2:
                        kpt1.vt_nG[n1] = 0.5 * f1 * vt_R
                    kpt1.Htpsit_nG[n1] += f2 * nspins * psit2_R * vt_R
                    if kpt1 is not kpt2:
                        if invert:
                            kpt2.Htpsit_nG[n2] += (f1 * nspins *
                                                   psit1_R.conj() * vt_R)
                        else:
                            kpt2.Htpsit_nG[n2] += (f1 * nspins *
                                                   psit1_R * vt_R.conj())

    def calculate_paw_correction(self):
        exx = 0
        deg = 2 // self.nspins  # spin degeneracy
        for a, D_sp in self.density.D_asp.items():
            setup = self.setups[a]
            for D_p in D_sp:
                D_ii = unpack2(D_p)
                ni = len(D_ii)

                for i1 in range(ni):
                    for i2 in range(ni):
                        A = 0.0
                        for i3 in range(ni):
                            p13 = packed_index(i1, i3, ni)
                            for i4 in range(ni):
                                p24 = packed_index(i2, i4, ni)
                                A += setup.M_pp[p13, p24] * D_ii[i3, i4]
                        p12 = packed_index(i1, i2, ni)
                        exx -= self.hybrid / deg * D_ii[i1, i2] * A

                if setup.X_p is not None:
                    exx -= self.hybrid * np.dot(D_p, setup.X_p)
            exx += self.hybrid * setup.ExxC
        return exx
    
    def calculate_pair_density(self, n1, n2, psit1_G, psit2_G, P1_ani, P2_ani):
        Q_aL = {}
        for a, P1_ni in P1_ani.items():
            P1_i = P1_ni[n1]
            P2_i = P2_ani[a][n2]
            D_ii = np.outer(P1_i, P2_i.conj()).real
            D_p = pack(D_ii, tolerance=1e30)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)
            
        nt_G = psit1_G.conj() * psit2_G

        self.ghat.add(nt_G, Q_aL)

        return nt_G
