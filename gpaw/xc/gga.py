import numpy as np

from gpaw.xc.lda import LDA
from gpaw.utilities.blas import axpy
from gpaw.fd_operators import Gradient


class GGA(LDA):
    def set_grid_descriptor(self, gd):
        LDA.set_grid_descriptor(self, gd)
        self.grad_v = [Gradient(gd, v, allocate=True).apply for v in range(3)]

    def calculate_lda(self, e_g, n_sg, v_sg):
        nspins = len(n_sg)
        gradn_svg = self.gd.empty((nspins, 3))
        sigma_xg = self.gd.zeros(nspins * 2 - 1)
        dedsigma_xg = self.gd.empty(nspins * 2 - 1)
        for v in range(3):
            for s in range(nspins):
                self.grad_v[v](n_sg[s], gradn_svg[s, v])
                axpy(1.0, gradn_svg[s, v]**2, sigma_xg[2 * s])
            if nspins == 2:
                axpy(1.0, gradn_svg[0, v] * gradn_svg[1, v], sigma_xg[1])
        self.calculate_gga(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        vv_g = sigma_xg[0]
        for v in range(3):
            for s in range(nspins):
                self.grad_v[v](dedsigma_xg[2 * s] * gradn_svg[s, v], vv_g)
                axpy(-2.0, vv_g, v_sg[s])
                if nspins == 2:
                    self.grad_v[v](dedsigma_xg[1] * gradn_svg[s, v], vv_g)
                    axpy(-1.0, vv_g, v_sg[1 - s])
                    # TODO: can the number of gradient evaluations be reduced?

    def calculate_gga(self, e_g, n_sg, v_sg, sigma_xg, dedsigma_xg):
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg)
        
    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg, rnablaY_Lv,
                         tau_sg=None, dedtau_sg=None,  # used by MGGA subclass
                         e_g=None):
        nspins = len(n_sLg)
        if e_g is None:
            e_g = rgd.empty()
        n_sg = np.dot(Y_L, n_sLg)
        rd_vsg = np.dot(rnablaY_Lv.T, n_sLg)
        sigma_xg = rgd.empty(2 * nspins - 1)
        sigma_xg[::2] = (rd_vsg**2).sum(0)
        if nspins == 2:
            sigma_xg[1] = (rd_vsg[:, 0] * rd_vsg[:, 1]).sum(0)
        sigma_xg[:, 1:] /= rgd.r_g[1:]**2
        sigma_xg[:, 0] = sigma_xg[:, 1]
        d_sg = np.dot(Y_L, dndr_sLg)
        sigma_xg[::2] += d_sg**2
        if nspins == 2:
            sigma_xg[1] += d_sg[0] * d_sg[1]
        dedsigma_xg = rgd.zeros(2 * nspins - 1)
        self.kernel.calculate(e_g, n_sg, v_sg, sigma_xg, dedsigma_xg,
                                tau_sg, dedtau_sg)
        vv_sg = sigma_xg[:nspins]  # reuse array
        for s in range(nspins):
            rgd.derivative2(-2 * rgd.dv_g * dedsigma_xg[2 * s] * d_sg[s],
                            vv_sg[s])
        if nspins == 2:
            v_g = sigma_xg[2]
            rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * d_sg[1], v_g)
            vv_sg[0] -= v_g
            rgd.derivative2(rgd.dv_g * dedsigma_xg[1] * d_sg[0], v_g)
            vv_sg[1] -= v_g
        vv_sg[:, 1:] /= rgd.dv_g[1:]
        vv_sg[:, 0] = vv_sg[:, 1]
        v_sg += vv_sg
        return rgd.integrate(e_g), rd_vsg, dedsigma_xg

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        dndr_sg = np.empty_like(n_sg)
        for n_g, dndr_g in zip(n_sg, dndr_sg):
            rgd.derivative(n_g, dndr_g)
        return self.calculate_radial(rgd, n_sg[:, np.newaxis], [1.0], v_sg,
                                     dndr_sg[:, np.newaxis],
                                     np.zeros((1, 3)), e_g=e_g)[0]
