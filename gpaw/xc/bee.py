import numpy as np

import _gpaw
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc import LibXC
from gpaw.xc.vdw import FFTVDWFunctional
from gpaw import debug


class BEE1(XCKernel):
    def __init__(self, parameters=None):
        if parameters is None:
            self.name = 'BEE1'
            parameters = [0.0, 1.0]
        else:
            self.name = 'BEE1?'
        parameters = np.array(parameters, dtype=float).ravel()
        self.xc = _gpaw.XCFunctional(18, parameters)
        self.type = 'GGA'


class BEEVDWKernel(XCKernel):
    def __init__(self, xcoefs, ldac, pbec, lyp):
        self.BEE1 = BEE1(xcoefs)
        self.LDAc = LibXC('LDA_C_PW')
        self.PBEc = LibXC('GGA_C_PBE')
        self.LYP = LibXC('GGA_C_LYP')
        self.ldac = ldac
        self.pbec = pbec
        self.lyp = lyp

        self.type = 'GGA'
        self.name = 'BEEVDW'
        
    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)

        self.BEE1.calculate(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg)
        
        e0_g = np.empty_like(e_g)
        dedn0_sg = np.empty_like(dedn_sg)
        dedsigma0_xg = np.empty_like(dedsigma_xg)
        for coef, kernel in [
            (self.ldac, self.LDAc),
            (self.pbec - 1.0, self.PBEc),
            (self.lyp, self.LYP)]:
            dedn0_sg[:] = 0.0
            kernel.calculate(e0_g, n_sg, dedn0_sg, sigma_xg, dedsigma0_xg)
            e_g += coef * e0_g
            dedn_sg += coef * dedn0_sg
            if kernel.type == 'GGA':
                dedsigma_xg += coef * dedsigma0_xg

            
class BEEVDWFunctional(FFTVDWFunctional):
    def __init__(self, xcoefs=(0.0, 1.0), ccoefs=(0.0, 1.0, 0.0, 0.0),
                 **kwargs):
        ldac, pbec, lyp, vdw = ccoefs
        kernel = BEEVDWKernel(xcoefs, ldac, pbec, lyp)
        FFTVDWFunctional.__init__(self, name='BEEVDW',
                                  kernel=kernel, Zab=-0.8491, vdwcoef=vdw,
                                  **kwargs)
        
    def get_setup_name(self):
        return 'PBE'
