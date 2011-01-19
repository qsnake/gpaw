import _gpaw
from gpaw.xc.kernel import XCKernel
from gpaw.xc.libxc_functionals import libxc_functionals
from gpaw import debug

short_names = {
    'LDA':     'LDA_X+LDA_C_PW',
    'PW91':    'GGA_X_PW91+GGA_C_PW91',
    'PBE':     'GGA_X_PBE+GGA_C_PBE',
    'revPBE':  'GGA_X_PBE_R+GGA_C_PBE',
    'RPBE':    'GGA_X_RPBE+GGA_C_PBE',
    'BLYP':    'GGA_X_B88+GGA_C_LYP',
    'HCTH407': 'GGA_XC_HCTH_407',
    'TPSS':    'MGGA_X_TPSS+MGGA_C_TPSS',
    'M06L':    'MGGA_X_M06L+MGGA_C_M06L',
    'revTPSS': 'MGGA_X_REVTPSS+MGGA_C_REVTPSS'}


class LibXC(XCKernel):
    def __init__(self, name):
        self.name = name
        self.initialize(nspins=1)

    def initialize(self, nspins):
        self.nspins = nspins
        name = short_names.get(self.name, self.name)
        if name in libxc_functionals:
            f = libxc_functionals[name]
            xc = -1
            x = -1
            c = -1
            if '_XC_' in name:
                xc = f
            elif '_C_' in name:
                c = f
            else:
                x = f
        else:
            try:
                x, c = name.split('+')
            except ValueError:
                raise NameError('Unknown functional: "%s".' % name)
            xc = -1
            x = libxc_functionals[x]
            c = libxc_functionals[c]

        if xc != -1:
            # The C code can't handle this case!
            c = xc
            xc = -1

        self.xc = _gpaw.lxcXCFunctional(xc, x, c, nspins)

        if self.xc.is_mgga():
            self.type = 'MGGA'
        elif self.xc.is_gga() or self.xc.is_hyb_gga():
            self.type = 'GGA'
        else:
            self.type = 'LDA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)
        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)

        if nspins == 1:
            self.xc.calculate_spinpaired(e_g.ravel(), n_sg, dedn_sg,
                                         sigma_xg, dedsigma_xg,
                                         tau_sg, dedtau_sg)
        else:
            if self.type == 'LDA':
                self.xc.calculate_spinpolarized(
                    e_g.ravel(),
                    n_sg[0], dedn_sg[0],
                    n_sg[1], dedn_sg[1])
            elif self.type == 'GGA':
                self.xc.calculate_spinpolarized(
                    e_g.ravel(),
                    n_sg[0], dedn_sg[0],
                    n_sg[1], dedn_sg[1],
                    sigma_xg[0], sigma_xg[1], sigma_xg[2],
                    dedsigma_xg[0], dedsigma_xg[1], dedsigma_xg[2])
            else:
                self.xc.calculate_spinpolarized(
                    e_g.ravel(),
                    n_sg[0], dedn_sg[0],
                    n_sg[1], dedn_sg[1],
                    sigma_xg[0], sigma_xg[1], sigma_xg[2],
                    dedsigma_xg[0], dedsigma_xg[1], dedsigma_xg[2],
                    tau_sg[0], tau_sg[1],
                    dedtau_sg[0], dedtau_sg[1])

