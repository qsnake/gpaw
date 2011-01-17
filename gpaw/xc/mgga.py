import weakref

import numpy as np

from gpaw.xc.gga import GGA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient
from gpaw.lfc import LFC


class MGGA(GGA):
    orbital_dependent = True

    def __init__(self, kernel, nn=1):
        """Meta GGA functional.

        nn: int
            Number of neighbor grid points to use for FD stencil for
            wave function gradient.
        """
        self.nn = nn
        GGA.__init__(self, kernel)
        
    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)
        
    def get_setup_name(self):
        return 'PBE'

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.wfs = wfs
        self.tauct = LFC(wfs.gd,
                         [[setup.tauct] for setup in wfs.setups],
                         forces=True, cut=True)
        self.tauct_G = None
        self.dedtaut_sG = None
        self.restrict = hamiltonian.restrictor.apply
        self.interpolate = density.interpolator.apply
        self.taugrad_v = [Gradient(wfs.gd, v, n=self.nn, dtype=wfs.dtype,
                                   allocate=True).apply
                          for v in range(3)]

    def set_positions(self, spos_ac):
        self.tauct.set_positions(spos_ac)
        if self.tauct_G is None:
            self.tauct_G = self.wfs.gd.empty()
        self.tauct_G[:] = 0.0
        self.tauct.add(self.tauct_G)

    def calculate_gga(self, e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg):
        taut_sG = self.wfs.calculate_kinetic_energy_density(self.tauct,
                                                            self.taugrad_v)
        taut_sg = np.empty_like(nt_sg)
        for taut_G, taut_g in zip(taut_sG, taut_sg):
            taut_G += 1.0 / self.wfs.nspins * self.tauct_G
            self.interpolate(taut_G, taut_g)
        dedtaut_sg = np.empty_like(nt_sg)
        self.kernel.calculate(e_g, nt_sg, v_sg, sigma_xg, dedsigma_xg,
                                taut_sg, dedtaut_sg)
        self.dedtaut_sG = self.wfs.gd.empty(self.wfs.nspins)
        self.ekin = 0.0
        for s in range(self.wfs.nspins):
            self.restrict(dedtaut_sg[s], self.dedtaut_sG[s])
            self.ekin -= self.wfs.gd.integrate(
                self.dedtaut_sG[s] * (taut_sG[s] -
                                      self.tauct_G / self.wfs.nspins))
                                               
    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp):
        a_G = self.wfs.gd.empty(dtype=psit_xG.dtype)
        for psit_G, Htpsit_G in zip(psit_xG, Htpsit_xG):
            for v in range(3):
                self.taugrad_v[v](psit_G, a_G, kpt.phase_cd)
                self.taugrad_v[v](self.dedtaut_sG[kpt.s] * a_G, a_G,
                                  kpt.phase_cd)
                axpy(-0.5, a_G, Htpsit_G)

    def add_forces(self, F_av):
        dF_av = self.tauct.dict(derivative=True)
        self.tauct.derivative(self.dedtaut_sG.sum(0), dF_av)
        for a, dF_v in dF_av.items():
            F_av[a] += dF_v[0]

    def estimate_memory(self, mem):
        bytecount = self.wfs.gd.bytecount()
        mem.subnode('MGGA arrays', (1 + self.wfs.nspins) * bytecount)
        
    def initialize_kinetic(self, xccorr):
        nii = xccorr.nii
        nn = len(xccorr.rnablaY_nLv)
        ng = len(xccorr.phi_jg[0])

        tau_npg = np.zeros((nn, nii, ng))
        taut_npg = np.zeros((nn, nii, ng))
        self.create_kinetic(xccorr, nn, xccorr.phi_jg, tau_npg)
        self.create_kinetic(xccorr, nn, xccorr.phit_jg, taut_npg)
        return tau_npg, taut_npg

    def create_kinetic(self, x, ny, phi_jg, tau_ypg):
        """Short title here.
        
        kinetic expression is::

                                             __         __
          tau_s = 1/2 Sum_{i1,i2} D(s,i1,i2) \/phi_i1 . \/phi_i2 +tauc_s

        here the orbital dependent part is calculated::

          __         __
          \/phi_i1 . \/phi_i2 =
                      __    __
                      \/YL1.\/YL2 phi_j1 phi_j2 +YL1 YL2 dphi_j1 dphi_j2
                                                         ------  ------
                                                           dr     dr
          __    __
          \/YL1.\/YL2 [y] = Sum_c A[L1,c,y] A[L2,c,y] / r**2
          
        """
        nj = len(phi_jg)
        ni = len(x.jlL)
        nii = ni * (ni + 1) // 2
        dphidr_jg = np.zeros(np.shape(phi_jg))
        for j in range(nj):
            phi_g = phi_jg[j]
            x.rgd.derivative(phi_g, dphidr_jg[j])

        # Second term:
        for y in range(ny):
            i1 = 0
            p = 0
            Y_L = x.Y_nL[y]
            for j1, l1, L1 in x.jlL:
                for j2, l2, L2 in x.jlL[i1:]:
                    c = Y_L[L1]*Y_L[L2]
                    temp = c * dphidr_jg[j1] *  dphidr_jg[j2]
                    tau_ypg[y,p,:] += temp
                    p += 1
                i1 +=1
        ##first term
        for y in range(ny):
            i1 = 0
            p = 0
            rnablaY_Lv = x.rnablaY_nLv[y, :x.Lmax]
            Ax_L = rnablaY_Lv[:, 0]
            Ay_L = rnablaY_Lv[:, 1]
            Az_L = rnablaY_Lv[:, 2]
            for j1, l1, L1 in x.jlL:
                for j2, l2, L2 in x.jlL[i1:]:
                    temp = (Ax_L[L1] * Ax_L[L2] + Ay_L[L1] * Ay_L[L2]
                            + Az_L[L1] * Az_L[L2])
                    temp *=  phi_jg[j1] * phi_jg[j2]
                    temp[1:] /= x.rgd.r_g[1:]**2
                    temp[0] = temp[1]
                    tau_ypg[y, p, :] += temp
                    p += 1
                i1 +=1
        tau_ypg *= 0.5
                    
        return
        
