# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import array
import Numeric as num

from gpaw.grid_descriptor import RadialGridDescriptor
from gpaw.operators import Gradient
from gpaw.utilities import is_contiguous
from gpaw.exx import EXX, XXFunctional
from gpaw.kli import KLIFunctional
from gpaw.kli import GLLBFunctional
import _gpaw

"""
     A Short Description for 'xc':s

     paw.hamilton object has a member called xc which is of class XC3DGrid.
     There is also a class named XCRadialGrid. These classes calculate
     the derivatives for gga in these different coordinates. Both have the
     same superclass XCGrid which ensures that the arrays are contiguos.

     XC3DGrid has a member called xcfunc which is of class XCFunctional.
     This XCFunctional is a wrapper for real functional which initializes its
     member called xc, for the correct functional instance. So the actual
     xc-functional can be found at hamilton.xc.xcfunc.xc
"""

class XCFunctional:
    def __init__(self, xcname, parameters=None):
        self.xcname = xcname
        self.hybrid = 0.0
        self.parameters = parameters
        self.mgga = False
        self.gga = False 
        self.orbital_dependent = False

        if xcname == 'LDA':
            self.maxDerivativeLevel=2
            code = 117 # not used!
        elif xcname == 'LDAc':
            self.maxDerivativeLevel=2
            code = 7
        elif xcname == 'LDAx':
            code = 11
        elif xcname == 'KLI':  
            code = 15               
            self.orbital_dependent = True
        elif xcname == 'EXX':
            code = 6
            self.hybrid = 1.0
            self.orbital_dependent = True
        elif xcname == 'GLLB':
            self.orbital_dependent = True
            code = 16
        else:
            self.gga = True
            self.maxDerivativeLevel=1
            if xcname == 'PBE':
                code = 0
            elif xcname == 'revPBE':
                code = 1
            elif xcname == 'RPBE':
                code = 2
            elif xcname.startswith('XC'):
                code = 3
            elif xcname == 'PBE0':
                self.hybrid = 0.25
                code = 4
            elif xcname == 'PADE':
                code = 5
            elif xcname == 'revPBEx':
                code = 8
            elif xcname == 'RPBEx':
                code = 12
            elif xcname == 'TPSS':
                code = 9
                self.mgga = True
            elif xcname == 'PW91':
                code = 14
            else:
                raise TypeError('Unknown exchange-correlation functional')

        if code == 3:
            i = int(xcname[3])
            s0 = float(xcname[5:])
            self.xc = _gpaw.XCFunctional(code, self.gga, s0, i)
        elif code == 5:
            self.xc = _gpaw.XCFunctional(code, self.gga,
                                         0.0, 0, num.array(parameters))
        elif code == 6:
            self.xc = XXFunctional()
        elif code == 9:
            self.xc = _gpaw.MGGAFunctional(code)
        elif code == 15:
            self.xc = KLIFunctional()
        elif code == 16:
            self.xc = GLLBFunctional()
        else:
            self.xc = _gpaw.XCFunctional(code, self.gga)

    def __getstate__(self):
        return self.xcname, self.parameters

    def __setstate__(self, state):
        xcname, parameters = state
        self.__init__(xcname, parameters)

    # Returns true, if the orbital is orbital dependent.
    def is_non_local(self):
        return self.orbital_dependent

    def set_non_local_things(self, paw, energy_only=False):
        if not self.orbital_dependent:
            return

        if self.xcname == 'GLLB':
            self.xc.pass_stuff(paw.kpt_u, paw.finegd, paw.density.interpolate,
                               paw.nspins, paw.my_nuclei)

        if self.xcname == 'KLI':
            self.xc.pass_stuff(
                paw.kpt_u, paw.gd, paw.finegd, paw.density.interpolate,
                paw.hamiltonian.restrict, paw.hamiltonian.poisson,
                paw.my_nuclei, paw.ghat_nuclei,
                paw.nspins, paw.nmyu, paw.nbands,
                paw.kpt_comm, paw.domain.comm, paw.density.nt_sg)

        if self.hybrid > 0.0:
            if paw.typecode == num.Complex:
                raise NotImplementedError, 'k-point calculation with EXX'
            self.exx = EXX(paw.gd, paw.finegd, paw.density.interpolate,
                           paw.hamiltonian.restrict, paw.hamiltonian.poisson,
                           paw.my_nuclei, paw.ghat_nuclei,
                           paw.nspins, paw.nmyu, paw.nbands,
                           paw.kpt_comm, paw.domain.comm,
                           energy_only)

    def apply_non_local(self, kpt, Htpsit_nG=None, H_nn=None):
        if self.orbital_dependent:
            if self.hybrid > 0.0:
                self.exx.apply(kpt, Htpsit_nG, H_nn, self.hybrid)

    def get_non_local_energy(self):
        Exc = 0.0
        
        if self.orbital_dependent:
            if self.hybrid > 0.0:
                Exc += self.exx.Exx
        
        return Exc

    def get_non_local_kinetic_corrections(self):
        Ekin = 0.0
        if self.orbital_dependent:
            if self.hybrid > 0.0:
                Ekin += self.exx.Ekin

        return Ekin
    
    def adjust_non_local_residual(self, pR_G, dR_G, eps, u, s, k, n):
        if self.hybrid > 0.0:
            self.exx.adjust_residual(pR_G, dR_G, u, n)

    # For non-local functional, this function does the calculation for special
    # case of setup-generator. The processes for non-local in radial and 3D-grid
    # deviate so greatly that this is special treatment is needed.
    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j, v_xc):
        # Send the command one .xc up
        return self.xc.get_non_local_energy_and_potential1D(gd, u_j, f_j, e_j, l_j, v_xc)
        
    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g=None, deda2_g=None,
                             tau_g=None):
        if self.mgga:
            print "<calculate_spinpaired> type(v_g)=",type(v_g)
            if type(v_g) != 'vxcOperator':
                v_g = vxcOperator(v_g)
            print "<calculate_spinpaired> 2 type(v_g)=",type(v_g)
            print "<calculate_spinpaired> v_g=",v_g
            self.xc.calculate_spinpaired(e_g.flat, n_g, v_g, a2_g, deda2_g,
                                         tau_g)
        elif self.gga:
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

    def get_max_derivative_level(self):
        """maximal derivative level of Exc available""" 
        return self.maxDerivativeLevel
            
    def get_name(self):
        return self.xcname

    def exchange(self, rs, a2=0):
        return self.xc.exchange(rs, a2)

    def correlation(self, rs, zeta=0, a2=0):
        return self.xc.correlation(rs, zeta, a2)

class XCGrid:
    def __init__(self, xcfunc, gd, nspins):
        """Base class for XC3DGrid and XCRadialGrid."""

        self.gd = gd
        self.nspins = nspins
        
        if isinstance(xcfunc, str):
            xcfunc = XCFunctional(xcfunc)
        self.set_functional(xcfunc)

    def set_functional(self, xcfunc):
        self.xcfunc = xcfunc

    def get_functional(self):
        return self.xcfunc
    
    def get_energy_and_potential(self, na_g, va_g, nb_g=None, vb_g=None,
                                 taua_g=None, taub_g=None):
        assert is_contiguous(na_g, num.Float)
        assert is_contiguous(va_g, num.Float)
        assert na_g.shape == va_g.shape == self.shape
        if nb_g is None:
            return self.get_energy_and_potential_spinpaired(na_g, va_g, taua_g)
        else:
            assert is_contiguous(nb_g, num.Float)
            assert is_contiguous(vb_g, num.Float)
            assert nb_g.shape == vb_g.shape == self.shape
            return self.get_energy_and_potential_spinpolarized(na_g, va_g,
                                                               nb_g, vb_g)

class XC3DGrid(XCGrid):
    def __init__(self, xcfunc, gd, nspins=1):
        """XC-functional object for 3D uniform grids."""
        XCGrid.__init__(self, xcfunc, gd, nspins)

    def set_functional(self, xcfunc):
        XCGrid.set_functional(self, xcfunc)

        gd = self.gd
        self.shape = tuple(gd.n_c)
        self.dv = gd.dv
        if xcfunc.gga:
            self.ddr = [Gradient(gd, c).apply for c in range(3)]
            self.dndr_cg = gd.empty(3)
            self.a2_g = gd.empty()
            self.deda2_g = gd.empty()
            if self.nspins == 2:
                self.dnadr_cg = gd.empty(3)
                self.dnbdr_cg = gd.empty(3)
                self.aa2_g = gd.empty()
                self.ab2_g = gd.empty()
                self.dedaa2_g = gd.empty()
                self.dedab2_g = gd.empty()
        if xcfunc.mgga:
            self.taua_g = gd.empty()
            if self.nspins == 2:
                self.taub_g = gd.empty()
        self.e_g = gd.empty()

    # Calculates exchange energy and potential.
    # The energy density will be returned on reference e_g if it is specified.
    # Otherwise the method will use self.e_g
    def get_energy_and_potential_spinpaired(self, n_g, v_g, tau_g=None, e_g=None):
        if e_g == None:
            e_g = self.e_g
            
        if self.xcfunc.mgga:
            # derivatives of the density
            for c in range(3):
                self.ddr[c](n_g, self.dndr_cg[c])
            self.a2_g[:] = num.sum(self.dndr_cg**2)
            # derivatives of the tau
            
            self.xcfunc.calculate_spinpaired(e_g,
                                             n_g, v_g,
                                             self.a2_g,
                                             self.deda2_g)
            
        elif self.xcfunc.gga:
            for c in range(3):
                self.ddr[c](n_g, self.dndr_cg[c])
            self.a2_g[:] = num.sum(self.dndr_cg**2)

            self.xcfunc.calculate_spinpaired(e_g,
                                             n_g, v_g,
                                             self.a2_g,
                                             self.deda2_g)
            tmp_g = self.dndr_cg[0]
            for c in range(3):
                self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                v_g -= 2.0 * tmp_g
        else:
            self.xcfunc.calculate_spinpaired(e_g, n_g, v_g)
            
        return num.sum(self.e_g.flat) * self.dv

    def get_energy_and_potential_spinpolarized(self, na_g, va_g, nb_g, vb_g):
        if self.xcfunc.gga:
            for c in range(3):
                self.ddr[c](na_g, self.dnadr_cg[c])
                self.ddr[c](nb_g, self.dnbdr_cg[c])
            self.dndr_cg[:] = self.dnadr_cg + self.dnbdr_cg
            self.a2_g[:] = num.sum(self.dndr_cg**2)
            self.aa2_g[:] = num.sum(self.dnadr_cg**2)
            self.ab2_g[:] = num.sum(self.dnbdr_cg**2)

            self.xcfunc.calculate_spinpolarized(self.e_g,
                                                na_g, va_g,
                                                nb_g, vb_g,
                                                self.a2_g,
                                                self.aa2_g, self.ab2_g,
                                                self.deda2_g,
                                                self.dedaa2_g, self.dedab2_g)
            tmp_g = self.a2_g
            for c in range(3):
                self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                va_g -= 2.0 * tmp_g
                vb_g -= 2.0 * tmp_g
                self.ddr[c](self.dedaa2_g * self.dnadr_cg[c], tmp_g)
                va_g -= 4.0 * tmp_g
                self.ddr[c](self.dedab2_g * self.dnbdr_cg[c], tmp_g)
                vb_g -= 4.0 * tmp_g
        else:
            self.xcfunc.calculate_spinpolarized(self.e_g,
                                                na_g, va_g,
                                                nb_g, vb_g)
        return num.sum(self.e_g.flat) * self.dv

class XCRadialGrid(XCGrid):
    def __init__(self, xcfunc, gd, nspins=1):
        """XC-functional object for radial grids."""
        XCGrid.__init__(self, xcfunc, gd, nspins)

    def set_functional(self, xcfunc):
        XCGrid.set_functional(self, xcfunc)

        gd = self.gd
        
        self.shape = (len(gd.r_g),)
        assert self.shape[0] >= 4
        self.dv_g = gd.dv_g
            
        if xcfunc.gga:
            self.rgd = gd
            self.dndr_g = num.empty(self.shape, num.Float)
            self.a2_g = num.empty(self.shape, num.Float)
            self.deda2_g = num.empty(self.shape, num.Float)
            if self.nspins == 2:
                self.dnadr_g = num.empty(self.shape, num.Float)
                self.dnbdr_g = num.empty(self.shape, num.Float)
                self.aa2_g = num.empty(self.shape, num.Float)
                self.ab2_g = num.empty(self.shape, num.Float)
                self.dedaa2_g = num.empty(self.shape, num.Float)
                self.dedab2_g = num.empty(self.shape, num.Float)
        if xcfunc.mgga:
            self.taua_g = num.empty(self.shape, num.Float)
            if self.nspins == 2:
                self.taub_g = num.empty(self.shape, num.Float)
        
        self.e_g = num.empty(self.shape, num.Float) 

    # True, if this xc-potential depends on more than just density
    def is_non_local(self):
        return self.xcfunc.is_non_local()

    # This is called from all_electron.py
    # Special function for just 1D-case
    def get_non_local_energy_and_potential(self, u_j, f_j, e_j, l_j, v_xc):
        # Send the command one .xc up. Include also the grid descriptor.
        return self.xcfunc.get_non_local_energy_and_potential1D(self.gd, u_j, f_j, e_j, l_j, v_xc)

    def get_energy_and_potential_spinpaired(self, n_g, v_g, tau_g=None, e_g = None):

        if e_g == None:
            e_g = self.e_g
            
        if self.xcfunc.mgga:
            self.rgd.derivative(n_g, self.dndr_g)
            self.a2_g[:] = self.dndr_g**2

            print "<get_energy_and_potential_spinpaired> type=",type(v_g)
            
            self.xcfunc.calculate_spinpaired(e_g,
                                             n_g, v_g,
                                             self.a2_g,
                                             self.deda2_g,
                                             tau_g)
            
        elif self.xcfunc.gga:
            self.rgd.derivative(n_g, self.dndr_g)
            self.a2_g[:] = self.dndr_g**2

            self.xcfunc.calculate_spinpaired(e_g,
                                             n_g, v_g,
                                             self.a2_g,
                                             self.deda2_g)
            tmp_g = self.dndr_g
            self.rgd.derivative2(self.dv_g * self.deda2_g *
                                 self.dndr_g, tmp_g)
            tmp_g[1:] /= self.dv_g[1:]
            tmp_g[0] = tmp_g[1]
            v_g -= 2.0 * tmp_g
        else:
            self.xcfunc.calculate_spinpaired(e_g, n_g, v_g)

        return num.dot(self.e_g, self.dv_g)

    def get_energy_and_potential_spinpolarized(self, na_g, va_g, nb_g, vb_g):
        if self.xcfunc.gga:
            self.rgd.derivative(na_g, self.dnadr_g)
            self.rgd.derivative(nb_g, self.dnbdr_g)
            self.dndr_g[:] = self.dnadr_g + self.dnbdr_g
            self.a2_g[:] = self.dndr_g**2
            self.aa2_g[:] = self.dnadr_g**2
            self.ab2_g[:] = self.dnbdr_g**2

            self.xcfunc.calculate_spinpolarized(self.e_g,
                                                na_g, va_g,
                                                nb_g, vb_g,
                                                self.a2_g,
                                                self.aa2_g, self.ab2_g,
                                                self.deda2_g,
                                                self.dedaa2_g, self.dedab2_g)
            tmp_g = self.a2_g
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
            self.xcfunc.calculate_spinpolarized(self.e_g,
                                                na_g, va_g,
                                                nb_g, vb_g)

        return num.dot(self.e_g, self.dv_g)

class vxcOperator(list):
    """vxc as operator object"""
    def __init__(self, v):
        print "<vxcOperator::__init__> type(v)=",v
        print "<vxcOperator::__init__> type(v)=",v.shape
        # init the local part
        list.__init__(self,v)
        # lists for the operator part
        
