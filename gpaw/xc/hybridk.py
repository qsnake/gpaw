# Copyright (C) 2010  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module provides all the classes and functions associated with the
evaluation of exact exchange with k-point sampling."""

from math import pi, sqrt

import numpy as np
from ase import Atoms

from gpaw.xc import XC
from gpaw.xc.kernel import XCNull
from gpaw.xc.functional import XCFunctional
from gpaw.utilities import hartree, pack, unpack2, packed_index
from gpaw.lfc import LFC
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.kpoint import KPoint as KPoint0
from gpaw.mpi import world


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
        
        self.gd = density.gd
        self.kd = wfs.kd
        self.bd = wfs.bd

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
            dfghdfgh
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
        
        self.ghat = LFC(self.gd,
                        [setup.ghat_l for setup in density.setups],
                        dtype=complex
                        )
        self.ghat.set_k_points(self.bzk_kc)
        
        self.fullkd = KPointDescriptor(self.kd.bzk_kc, nspins=1)
        class S:
            id_a = []
            def set_symmetry(self, s): pass
            
        self.fullkd.set_symmetry(Atoms(pbc=True), S(), False)
        self.fullkd.set_communicator(world)
        self.pt = LFC(self.gd, [setup.pt_j for setup in density.setups],
                      dtype=complex)
        self.pt.set_k_points(self.fullkd.ibzk_kc)

        self.interpolator = density.interpolator

    def set_positions(self, spos_ac):
        self.ghat.set_positions(spos_ac)
        self.pt.set_positions(spos_ac)
    
    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        # Normal XC contribution:
        exc = self.xc.calculate(gd, n_sg, v_sg, e_g)

        # Add EXX contribution:
        return exc + self.exx

    def calculate_exx(self):
        """Non-selfconsistent calculation."""

        kd = self.kd
        K = self.fullkd.nibzkpts
        assert self.nspins == 1
        Q = K // world.size
        assert Q * world.size == K
        parallel = (world.size > self.nspins)
        
        self.exx = 0.0
        self.exx_skn = np.zeros((self.nspins, K, self.bd.nbands))

        kpt_u = []
        for k in range(world.rank * Q, (world.rank + 1) * Q):
            k_c = self.fullkd.ibzk_kc[k]
            for k1, k1_c in enumerate(kd.bzk_kc):
                if abs(k1_c - k_c).max() < 1e-10:
                    break
                
            # Index of symmetry related point in the irreducible BZ
            ik = kd.kibz_k[k1]
            kpt = self.kpt_u[ik]

            # KPoint from ground-state calculation
            phase_cd = np.exp(2j * pi * self.gd.sdisp_cd * k_c[:, np.newaxis])
            kpt2 = KPoint0(kpt.weight, kpt.s, k, None, phase_cd)
            kpt2.psit_nG = np.empty_like(kpt.psit_nG)
            kpt2.f_n = kpt.f_n / kpt.weight / K * 2
            for n, psit_G in enumerate(kpt2.psit_nG):
                psit_G[:] = kd.transform_wave_function(kpt.psit_nG[n], k1)

            kpt2.P_ani = self.pt.dict(len(kpt.psit_nG))
            self.pt.integrate(kpt2.psit_nG, kpt2.P_ani, k)
            kpt_u.append(kpt2)

        for s in range(self.nspins):
            kpt1_q = [KPoint(self.fullkd, kpt) for kpt in kpt_u if kpt.s == s]
            kpt2_q = kpt1_q[:]

            if len(kpt1_q) == 0:
                # No s-spins on this CPU:
                continue

            # Send rank:
            srank = self.fullkd.get_rank_and_index(s, (kpt1_q[0].k - 1) % K)[0]

            # Receive rank:
            rrank = self.fullkd.get_rank_and_index(s, (kpt1_q[-1].k + 1) % K)[0]

            # Shift k-points K // 2 times:
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
            
        self.exx = world.sum(self.exx)
        world.sum(self.exx_skn)
        self.exx += self.calculate_paw_correction()
        
    def apply(self, kpt1, kpt2, invert=False):
        #print world.rank,kpt1.k,kpt2.k,invert
        k1_c = self.fullkd.ibzk_kc[kpt1.k]
        k2_c = self.fullkd.ibzk_kc[kpt2.k]
        if invert:
            k2_c = -k2_c
        k12_c = k1_c - k2_c
        N_c = self.gd.N_c
        eikr_R = np.exp(2j * pi * np.dot(np.indices(N_c).T, k12_c / N_c).T)

        for q, k_c in enumerate(self.bzk_kc):
            if abs(k_c + k12_c).max() < 1e-9:
                q0 = q
                break

        for q, k_c in enumerate(self.bzk_kc):
            if abs(k_c - k12_c).max() < 1e-9:
                q00 = q
                break

        Gpk2_G = self.pwd.G2_qG[q0]
        if Gpk2_G[0] == 0:
            Gpk2_G = Gpk2_G.copy()
            Gpk2_G[0] = 1.0 / self.gamma

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

                nt_R = self.calculate_pair_density(n1, n2, kpt1, kpt2, q0,
                                                   invert)
                                                   
                nt_G = self.pwd.fft(nt_R * eikr_R) / N
                vt_G = nt_G.copy()
                vt_G *= -pi * vol / Gpk2_G
                e = np.vdot(nt_G, vt_G).real * nspins * self.hybrid
                if same and n1 == n2:
                    e /= 2
                    
                self.exx += e * f1 * f2
                self.ekin -= 2 * e * f1 * f2
                self.exx_skn[kpt1.s, kpt1.k, n1] += f2 * e
                self.exx_skn[kpt2.s, kpt2.k, n2] += f1 * e

                calculate_potential = not True
                if calculate_potential:
                    vt_R = self.pwd.ifft(vt_G).conj() * eikr_R * N / vol
                    if kpt1 is kpt2 and not invert and n1 == n2:
                        kpt1.vt_nG[n1] = 0.5 * f1 * vt_R

                    if invert:
                        kpt1.Htpsit_nG[n1] += \
                                           f2 * nspins * psit2_R.conj() * vt_R
                    else:
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
    
    def calculate_pair_density(self, n1, n2, kpt1, kpt2, q, invert):
        if invert:
            nt_G = kpt1.psit_nG[n1].conj() * kpt2.psit_nG[n2].conj()
        else:
            nt_G = kpt1.psit_nG[n1].conj() * kpt2.psit_nG[n2]

        Q_aL = {}
        for a, P1_ni in kpt1.P_ani.items():
            P1_i = P1_ni[n1]
            P2_i = kpt2.P_ani[a][n2]
            if invert:
                D_ii = np.outer(P1_i.conj(), P2_i.conj())
            else:
                D_ii = np.outer(P1_i.conj(), P2_i)
            D_p = pack(D_ii)
            Q_aL[a] = np.dot(D_p, self.setups[a].Delta_pL)

        self.ghat.add(nt_G, Q_aL, q)
        return nt_G


if __name__ == '__main__':
    import sys
    from gpaw import GPAW
    from gpaw.mpi import serial_comm
    calc = GPAW(sys.argv[1], txt=None, communicator=serial_comm)

    alpha = 5.0
    e = calc.get_potential_energy()
    exx = HybridXC('EXX', alpha=alpha)
    e2 = calc.get_xc_difference(exx)
    print e, e + e2, exx.exx
