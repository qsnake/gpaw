import numpy as np

from gpaw.xc_functional import XCRadialGrid
from gpaw.xc_correction import OldXCCorrection


class SICXCCorrection(OldXCCorrection):
    def __init__(self, xccorr, xcfunc):
        self.nc_g = np.zeros(xccorr.ng)
        self.nct_g = np.zeros(xccorr.ng)
        self.xc = XCRadialGrid(xcfunc, xccorr.rgd, 2)
        self.Exc0 = 0.0
        self.Lmax = xccorr.Lmax
        self.rgd = xccorr.rgd
        self.dv_g = xccorr.dv_g
        self.nspins = 2
        self.Y_nL = xccorr.Y_nL
        self.ng = xccorr.ng
        self.ni = xccorr.ni
        self.nj = xccorr.nj
        self.nii = xccorr.nii
        self.B_pqL = xccorr.B_pqL
        self.B_Lqp = xccorr.B_Lqp
        self.n_qg = xccorr.n_qg
        self.nt_qg = xccorr.nt_qg
        self.ncorehole_g = None
        self.nca_g = self.ncb_g = 0.5 * self.nc_g
