# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw import _gridpaw
from gridpaw.grid_descriptor import RadialGridDescriptor
from gridpaw.operators import Gradient
from gridpaw.utilities import is_contiguous


class XCFunctional:
    def __init__(self, xcname, scalarrel=True):
        self.set_xc_functional(xcname, scalarrel)
        
    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g=None, deda2_g=None):
        if self.gga:
            # e_g.flat !!!!! XXX
            self.xc.calculate_spinpaired(e_g.flat, n_g, v_g, a2_g, deda2_g)
        else:
            self.xc.calculate_spinpaired(e_g.flat, n_g, v_g)
         
    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                               a2_g=None, aa2_g=None, ab2_g=None,
                               deda2_g=None, dedaa2_g=None, dedab2_g=None):
        if self.gga:
            self.xc.calculate_spinpolarized(e_g.flat, na_g, va_g, nb_g, vb_g,
                                           a2_g, aa2_g, ab2_g,
                                           deda2_g, dedaa2_g, dedab2_g)
        else:
            self.xc.calculate_spinpolarized(e_g.flat, na_g, va_g, nb_g, vb_g)
            
    def get_xc_name(self):
        return self.xcname

    def set_xc_functional(self, xcname, scalarrel=True):
        self.xcname = xcname
        
        if xcname == 'LDA':
            self.gga = False
            type = 117 # not used!
        else:
            self.gga = True
            if xcname == 'PBE':
                type = 0
            elif xcname == 'revPBE':
                type = 1
            elif xcname == 'RPBE':
                type = 2
            elif xcname.startswith('XC'):
                type = 3
            else:
                raise TypeError('Unknown exchange-correlation functional')

        if type == 3:
            i = int(xcname[3])
            s0 = float(xcname[5:])
            self.xc = _gridpaw.XCFunctional(type, self.gga, scalarrel, s0, i)
        else:
            self.xc = _gridpaw.XCFunctional(type, self.gga, scalarrel)

    def exchange(self, rs, a2=0):
        return self.xc.exchange(rs, a2)

    def correlation(self, rs, zeta=0, a2=0):
        return self.xc.correlation(rs, zeta, a2)


class XCOperator:
    def __init__(self, xcfunc, gd, nspins=1):
        """XCFunctional(name) -> XCFunctional object

        The string name must be one of LDA, PBE or revPBE."""

        if type(xcfunc) is str:
            xcfunc = XCFunctional(xcfunc)
        self.xc = xcfunc
        
        if isinstance(gd, RadialGridDescriptor):
            self.radial = True
            self.shape = (len(gd.r_g),)
            assert self.shape[0] >= 4
            self.dv_g = gd.dv_g
            if xcfunc.gga:
                self.rgd = gd
                self.dndr_g = num.zeros(self.shape, num.Float)
                self.a2_g = num.zeros(self.shape, num.Float)
                self.deda2_g = num.zeros(self.shape, num.Float)
                if nspins == 2:
                    self.dnadr_g = num.zeros(self.shape, num.Float)
                    self.dnbdr_g = num.zeros(self.shape, num.Float)
                    self.aa2_g = num.zeros(self.shape, num.Float)
                    self.ab2_g = num.zeros(self.shape, num.Float)
                    self.dedaa2_g = num.zeros(self.shape, num.Float)
                    self.dedab2_g = num.zeros(self.shape, num.Float)
        else:
            self.radial = False
            self.shape = tuple(gd.myng)
            self.dv = gd.dv
            if xcfunc.gga:
                self.ddr = [Gradient(gd, axis).apply for axis in range(3)]
                self.dndr_ig = num.zeros((3,) + self.shape, num.Float)
                self.a2_g = num.zeros(self.shape, num.Float)
                self.deda2_g = num.zeros(self.shape, num.Float)
                if nspins == 2:
                    self.dnadr_ig = num.zeros((3,) + self.shape, num.Float)
                    self.dnbdr_ig = num.zeros((3,) + self.shape, num.Float)
                    self.aa2_g = num.zeros(self.shape, num.Float)
                    self.ab2_g = num.zeros(self.shape, num.Float)
                    self.dedaa2_g = num.zeros(self.shape, num.Float)
                    self.dedab2_g = num.zeros(self.shape, num.Float)
        self.e_g = num.zeros(self.shape, num.Float) 

    def get_xc_functional(self):
        return self.xc
    
    def get_energy_and_potential(self, na_g, va_g, nb_g=None, vb_g=None):
        if nb_g is None:
            return self.get_energy_and_potential_spinpaired(na_g, va_g)
        else:
            return self.get_energy_and_potential_spinpolarized(na_g, va_g,
                                                               nb_g, vb_g)

    def get_energy_and_potential_spinpaired(self, n_g, v_g):
        assert is_contiguous(n_g, num.Float)
        assert is_contiguous(v_g, num.Float)
        assert n_g.shape == v_g.shape == self.shape
        if self.xc.gga:
            if self.radial:
                self.rgd.derivative(n_g, self.dndr_g)
                self.a2_g[:] = self.dndr_g**2
            else:
                for i in range(3):
                    self.ddr[i](n_g, self.dndr_ig[i])
                self.a2_g[:] = num.sum(self.dndr_ig**2)

            self.xc.calculate_spinpaired(self.e_g,
                                         n_g, v_g,
                                         self.a2_g,
                                         self.deda2_g)
            if self.radial:
                tmp_g = self.dndr_g
                self.rgd.derivative2(self.dv_g * self.deda2_g *
                                     self.dndr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                v_g -= 2.0 * tmp_g
            else:
                tmp_g = self.dndr_ig[0]
                for i in range(3):
                    self.ddr[i](self.deda2_g * self.dndr_ig[i], tmp_g)
                    v_g -= 2.0 * tmp_g
        else:
            self.xc.calculate_spinpaired(self.e_g, n_g, v_g)
        if self.radial:
            return num.dot(self.e_g, self.dv_g)
        else:
            return num.sum(self.e_g.flat) * self.dv

    def get_energy_and_potential_spinpolarized(self, na_g, va_g, nb_g, vb_g):
        assert is_contiguous(na_g, num.Float)
        assert is_contiguous(va_g, num.Float)
        assert is_contiguous(nb_g, num.Float)
        assert is_contiguous(vb_g, num.Float)
        assert na_g.shape == va_g.shape == self.shape
        assert nb_g.shape == vb_g.shape == self.shape
        if self.xc.gga:
            if self.radial:
                self.rgd.derivative(na_g, self.dnadr_g)
                self.rgd.derivative(nb_g, self.dnbdr_g)
                self.dndr_g[:] = self.dnadr_g + self.dnbdr_g
                self.a2_g[:] = self.dndr_g**2
                self.aa2_g[:] = self.dnadr_g**2
                self.ab2_g[:] = self.dnbdr_g**2
            else:
                for i in range(3):
                    self.ddr[i](na_g, self.dnadr_ig[i])
                    self.ddr[i](nb_g, self.dnbdr_ig[i])
                self.dndr_ig[:] = self.dnadr_ig + self.dnbdr_ig
                self.a2_g[:] = num.sum(self.dndr_ig**2)
                self.aa2_g[:] = num.sum(self.dnadr_ig**2)
                self.ab2_g[:] = num.sum(self.dnbdr_ig**2)

            self.xc.calculate_spinpolarized(self.e_g,
                                           na_g, va_g,
                                           nb_g, vb_g,
                                           self.a2_g,
                                           self.aa2_g, self.ab2_g,
                                           self.deda2_g,
                                           self.dedaa2_g, self.dedab2_g)
            tmp_g = self.a2_g
            if self.radial:
                self.rgd.derivative2(self.dv_g * self.deda2_g *
                                     self.dndr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                va_g -= 2.0 * tmp_g
                vb_g -= 2.0 * tmp_g
                self.rgd.derivative2(self.dv_g * self.dedaa2_g *
                                     self.dnadr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                va_g -= 4.0 * tmp_g
                self.rgd.derivative2(self.dv_g * self.dedab2_g *
                                     self.dnbdr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                vb_g -= 4.0 * tmp_g
            else:
                for i in range(3):
                    self.ddr[i](self.deda2_g * self.dndr_ig[i], tmp_g)
                    va_g -= 2.0 * tmp_g
                    vb_g -= 2.0 * tmp_g
                    self.ddr[i](self.dedaa2_g * self.dnadr_ig[i], tmp_g)
                    va_g -= 4.0 * tmp_g
                    self.ddr[i](self.dedab2_g * self.dnbdr_ig[i], tmp_g)
                    vb_g -= 4.0 * tmp_g
        else:
            self.xc.calculate_spinpolarized(self.e_g, na_g, va_g, nb_g, vb_g)
        if self.radial:
            return num.dot(self.e_g, self.dv_g)
        else:
            return num.sum(self.e_g.flat) * self.dv
