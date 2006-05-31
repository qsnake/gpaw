# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw.grid_descriptor import RadialGridDescriptor
from gridpaw.operators import Gradient
from gridpaw.utilities import is_contiguous
from gridpaw.exx import XXFunctional
import _gridpaw

class XCFunctional:
    def __init__(self, xcname, scalarrel=True, parameters=None):
        self.xcname = xcname
        self.parameters = parameters
        self.scalarrel = scalarrel
        
        if xcname == 'LDA':
            self.gga = False
            code = 117 # not used!
        else:
            self.gga = True
            if xcname == 'PBE':
                code = 0
            elif xcname == 'revPBE':
                code = 1
            elif xcname == 'RPBE':
                code = 2
            elif xcname.startswith('XC'):
                code = 3
            elif xcname == 'PBE0':
                code = 4
            elif xcname == 'PADE':
                code = 5
            elif xcname == 'EXX':
                code = 6
            else:
                raise TypeError('Unknown exchange-correlation functional')

        if code == 3:
            i = int(xcname[3])
            s0 = float(xcname[5:])
            self.xc = _gridpaw.XCFunctional(code, self.gga, scalarrel, s0, i)
        if code == 5:
            self.xc = _gridpaw.XCFunctional(code, self.gga, scalarrel,
                                            0.0, 0, num.array(parameters))
        if code == 3:
            self.xc = XXFunctional()
        else:
            self.xc = _gridpaw.XCFunctional(code, self.gga, scalarrel)

    def __getstate__(self):
        return self.xcname, self.scalarrel, self.parameters

    def __setstate__(self, state):
        xcname, scalarrel, parameters = state
        self.__init__(xcname, scalarrel, parameters)
    
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
            self.shape = tuple(gd.n_c)
            self.dv = gd.dv
            if xcfunc.gga:
                self.ddr = [Gradient(gd, c).apply for c in range(3)]
                self.dndr_cg = num.zeros((3,) + self.shape, num.Float)
                self.a2_g = num.zeros(self.shape, num.Float)
                self.deda2_g = num.zeros(self.shape, num.Float)
                if nspins == 2:
                    self.dnadr_cg = num.zeros((3,) + self.shape, num.Float)
                    self.dnbdr_cg = num.zeros((3,) + self.shape, num.Float)
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
                for c in range(3):
                    self.ddr[c](n_g, self.dndr_cg[c])
                self.a2_g[:] = num.sum(self.dndr_cg**2)

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
                tmp_g = self.dndr_cg[0]
                for c in range(3):
                    self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
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
                for c in range(3):
                    self.ddr[c](na_g, self.dnadr_cg[c])
                    self.ddr[c](nb_g, self.dnbdr_cg[c])
                self.dndr_cg[:] = self.dnadr_cg + self.dnbdr_cg
                self.a2_g[:] = num.sum(self.dndr_cg**2)
                self.aa2_g[:] = num.sum(self.dnadr_cg**2)
                self.ab2_g[:] = num.sum(self.dnbdr_cg**2)

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
                for c in range(3):
                    self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                    va_g -= 2.0 * tmp_g
                    vb_g -= 2.0 * tmp_g
                    self.ddr[c](self.dedaa2_g * self.dnadr_cg[c], tmp_g)
                    va_g -= 4.0 * tmp_g
                    self.ddr[c](self.dedab2_g * self.dnbdr_cg[c], tmp_g)
                    vb_g -= 4.0 * tmp_g
        else:
            self.xc.calculate_spinpolarized(self.e_g, na_g, va_g, nb_g, vb_g)
        if self.radial:
            return num.dot(self.e_g, self.dv_g)
        else:
            return num.sum(self.e_g.flat) * self.dv

    def get_second_derivatives(self, na_g, nb_g=None,scale=.001):
        """Second derivatives of Exc using a simple two point formula"""
        
        def invert(arr):
            """Invert the Numeric array with 1/0=0"""
            res=arr.copy()
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    for k in range(res.shape[2]):
                        if res[i,j,k]==0: pass
                        else            : res[i,j,k]=1./res[i,j,k]
            return res

        # unpolarised
        if nb_g is None:
            if self.xc.gga: # need derivatives at the actual density
                if self.radial:
                    self.rgd.derivative(na_g, self.dndr_g)
                    self.a2_g[:] = self.dndr_g**2
                else:
                    for c in range(3):
                        self.ddr[c](na_g, self.dndr_cg[c])
                    self.a2_g[:] = num.sum(self.dndr_cg**2)
            # small density difference
            dn_g=scale*na_g
            dinv_g=invert(dn_g)
            # upper and lower densities
            np1_g=na_g+dn_g
            vp1_g=na_g.copy()
            nm1_g=na_g-dn_g
            vm1_g=na_g.copy()
            if self.xc.gga:
                self.xc.calculate_spinpaired(self.e_g,
                                             np1_g, vp1_g,
                                             self.a2_g,
                                             self.deda2_g)
                if self.radial:
                    tmp_g = self.dndr_g
                    self.rgd.derivative2(self.dv_g * self.deda2_g *
                                         self.dndr_g, tmp_g)
                    tmp_g[1:] /= self.dv_g[1:]
                    tmp_g[0] = tmp_g[1]
                    vp1_g -= 2.0 * tmp_g
                else:
                    tmp_g = self.dndr_cg[0]
                    for c in range(3):
                        self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                        vp1_g -= 2.0 * tmp_g
                self.xc.calculate_spinpaired(self.e_g,
                                             nm1_g, vm1_g,
                                             self.a2_g,
                                             self.deda2_g)
                if self.radial:
                    tmp_g = self.dndr_g
                    self.rgd.derivative2(self.dv_g * self.deda2_g *
                                         self.dndr_g, tmp_g)
                    tmp_g[1:] /= self.dv_g[1:]
                    tmp_g[0] = tmp_g[1]
                    vm1_g -= 2.0 * tmp_g
                else:
                    tmp_g = self.dndr_cg[0]
                    for c in range(3):
                        self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                        vm1_g -= 2.0 * tmp_g
            else:
                self.get_energy_and_potential(np1_g,vp1_g)
                self.get_energy_and_potential(nm1_g,vm1_g)
            d2Edn2_g=[range(1)]
            d2Edn2_g[0]=.5*(vp1_g-vm1_g)*dinv_g
        # polarized
        else:
            pass
        return d2Edn2_g

