import numpy as np

from gpaw.xc.functional import XCFunctional


class LDA(XCFunctional):
    def __init__(self, kernel):
        self.kernel = kernel
        XCFunctional.__init__(self, kernel.name)
        self.type = kernel.type

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        if gd is not self.gd:
            self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        self.calculate_lda(e_g, n_sg, v_sg)
        return gd.integrate(e_g)

    def calculate_lda(self, e_g, n_sg, v_sg):
        self.kernel.calculate(e_g, n_sg, v_sg)

    def calculate_radial(self, rgd, n_sLg, Y_L, v_sg,
                         dndr_sLg=None, rnablaY_Lv=None,
                         e_g=None):
        if e_g is None:
            e_g = rgd.empty()
        n_sg = np.dot(Y_L, n_sLg)
        self.kernel.calculate(e_g, n_sg, v_sg)
        return rgd.integrate(e_g)

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        return self.calculate_radial(rgd, n_sg[:, np.newaxis], [1.0], v_sg,
                                     e_g=e_g)

    def calculate_fxc(self, gd, n_sg, f_sg):
        if gd is not self.gd:
            self.set_grid_descriptor(gd)

        assert len(n_sg) == 1
        assert n_sg.shape == f_sg.shape
        assert n_sg.flags.contiguous and n_sg.dtype == float
        assert f_sg.flags.contiguous and f_sg.dtype == float
        self.kernel.xc.calculate_fxc_spinpaired(n_sg.ravel(), f_sg)

