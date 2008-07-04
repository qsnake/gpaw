# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import array
import numpy as npy

from gpaw.grid_descriptor import RadialGridDescriptor
from gpaw.operators import Gradient
from gpaw.utilities import is_contiguous
from gpaw.utilities.timing import Timer
from gpaw.exx import EXX
from gpaw.gllb.nonlocalfunctionalfactory import NonLocalFunctionalFactory
from gpaw.libxc import Libxc
import _gpaw

"""
     A Short Description for 'xc':s

     paw.hamilton object has a member called xc which is of class XC3DGrid.
     There is also a class named XCRadialGrid. These classes calculate
     the derivatives for gga in these different coordinates. Both have the
     same superclass XCGrid which ensures that the arrays are contiguous.

     XC3DGrid has a member called xcfunc which is of class XCFunctional.
     This XCFunctional is a wrapper for real functional which initializes its
     member called xc, for the correct functional instance. So the actual
     xc-functional can be found at hamilton.xc.xcfunc.xc
"""


class ZeroFunctional:
    """Dummy XC functional"""
    def calculate_spinpaired(self, e_g, n_g, v_g):
        e_g[:] = 0.0

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g):
        e_g[:] = 0.0


class XCFunctional:
    def __init__(self, xcname, nspins=1, parameters=None):

        if isinstance(xcname, dict):
            parameters = xcname
            xcname = parameters.pop('name')

        self.xcname = xcname
        self.hybrid = 0.0
        self.parameters = parameters
        self.mgga = False
        self.gga = False
        self.gllb = False
        self.orbital_dependent = False
        self.uses_libxc = False
        self.nspins = nspins

        # Check if setup name has been set manually
        index = xcname.rfind('-setup')
        if index == -1:
            self.setupname = None
        else:
            self.setupname = xcname[index + len('-setup'):]
            xcname = xcname[:index]
            self.xcname = xcname

        # Special cases of functionals from libxc
        if xcname == 'XC_LB':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
        elif xcname.startswith('XC'): # MDTMP - case to be removed one day
            code = 3
            self.gga = True
            self.maxDerivativeLevel=1
        elif '-' in xcname: # functionals from libxc
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
        # Abbreviations for common functionals from libxc
        elif xcname == 'LDA':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X-C_PW'
        elif xcname == 'LDAx':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X-None'
        elif xcname == 'PBE':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_PBE-C_PBE'
        elif xcname == 'revPBE':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_PBE_R-C_PBE'
        elif xcname == 'revPBEx':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_PBE_R-None'
        elif xcname == 'RPBE':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_RPBE-C_PBE'
        elif xcname == 'RPBEx':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_RPBE-None'
        elif xcname == 'PW91':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_PW91-C_PW91'
        elif xcname == 'PBE0':
            assert (nspins is not None)
            code = 'lxc' # libxc
            self.uses_libxc = True
            xcname = 'X_PBE-C_PBE'
            self.orbital_dependent = True
            self.hybrid = 0.25
            if self.setupname is None:
                self.setupname = 'PBE'
        # End of: Abbreviations for common functionals from libxc
        elif xcname == 'oldLDA':
            self.maxDerivativeLevel=2
            code = 117 # not used!
            xcname = 'LDA'
        elif xcname == 'oldLDAc':
            self.maxDerivativeLevel=2
            code = 7
            xcname = 'LDAc'
        elif xcname == 'oldLDAx':
            code = 11
            xcname = 'LDAx'
        elif xcname == 'EXX':
            code = 6
            self.hybrid = 1.0
            self.orbital_dependent = True
            if self.setupname is None:
                self.setupname = 'LDA'
        elif xcname.startswith('GLLB') or xcname=='KLI':
            # GLLB type of functionals which use orbitals, require special
            # treatment at first iterations, where there is no orbitals
            # available. Therefore orbital_dependent = True!
            self.orbital_dependent = True
            self.gllb = True
            code = 'gllb'
        elif xcname == 'SAOP':
            self.reference = ( 'P.R.T. Schipper et al, ' +
                               'J Chem Phys 112 (2000) 1344' )
            self.orbital_dependent = True
            self.gllb = True
            self.gga = True
            code = 'gllb'
        else:
            self.gga = True
            self.maxDerivativeLevel=1
            if xcname == 'oldPBE':
                code = 0
                xcname = 'PBE'
            elif xcname == 'oldrevPBE':
                code = 1
                xcname = 'revPBE'
            elif xcname == 'oldRPBE':
                code = 2
                xcname = 'RPBE'
            elif xcname == 'oldPBE0':
                self.orbital_dependent = True
                self.hybrid = 0.25
                code = 4
                xcname = 'PBE0'
                if self.setupname is None:
                    self.setupname = 'oldPBE'
            elif xcname == 'PADE':
                code = 5
            elif xcname == 'oldrevPBEx':
                code = 8
                xcname = 'revPBEx'
            elif xcname == 'oldRPBEx':
                code = 12
                xcname = 'RPBEx'
            elif xcname == 'TPSS':
                code = 9
                self.mgga = True ## use real tau and local potential
                local_tau = False ## use Weiszacker term
                self.orbital_dependent = True

            elif xcname == 'oldPW91':
                code = 14
                xcname = 'PW91'
            elif xcname == 'LB94' or xcname == 'LBalpha':
                code = 17
                if xcname == 'LB94':
                    self.reference = ( 'R. van Leeuwen and E. J. Bearends, ' +
                                       'Phys Rev A 49 (1994) 2421' )
                    parameters = [1., 0.05] # alpha, beta
                else:
                    self.reference = ( 'P.R.T. Schipper et al, ' +
                                       'J Chem Phys 112 (2000) 1344' )
                    parameters = [1.19, 0.01] # alpha, beta
                if self.parameters:
                    for i, key in enumerate(['alpha', 'beta']):
                        if self.parameters.has_key(key):
                            parameters[i] = self.parameters[key]
            elif xcname == 'BEE1':
                code = 18
            else:
                raise TypeError('Unknown exchange-correlation functional')

        if code == 3:
            i = int(xcname[3])
            s0 = float(xcname[5:])
            self.xc = _gpaw.XCFunctional(code, self.gga, s0, i)
        elif code in [5, 17, 18]:
            self.xc = _gpaw.XCFunctional(code, self.gga,
                                         0.0, 0, npy.array(parameters))
        elif code == 6:
            self.xc = ZeroFunctional()
        elif code == 9:
            self.xc = _gpaw.MGGAFunctional(code,local_tau)
        elif code == 'gllb':
            # Get the correct functional from NonLocalFunctionalFactory
            self.xc = NonLocalFunctionalFactory().get_functional_by_name(xcname)
        elif code == 'lxc':
###            self.xcname = xcname # MDTMP: to get the lxc name for setup
            # find numeric identifiers of libxc functional based on xcname
            lxc_functional = Libxc.get_lxc_functional(
                Libxc(), Libxc.lxc_split_xcname(
                Libxc(), xcname))
            self.xc = _gpaw.lxcXCFunctional(
                lxc_functional[0], # exchange-correlation
                lxc_functional[1], # exchange
                lxc_functional[2], # correlation
                nspins,
                self.hybrid
                )
            self.mgga = bool(self.xc.is_mgga())
            self.gga = bool(self.xc.is_gga())
            if self.gga:
                self.maxDerivativeLevel=1
        else:
###            self.xcname = xcname # MDTMP: to get the xcname name for setup
            self.xc = _gpaw.XCFunctional(code, self.gga)
        self.timer = None

    def set_timer(self, timer):
        self.timer = timer

    def __getstate__(self):
        return self.xcname, self.nspins, self.parameters

    def __setstate__(self, state):
        xcname, nspins, parameters = state
        self.__init__(xcname, nspins, parameters)

    # Returns true, if the orbital is orbital dependent.
    def is_non_local(self):
        return self.orbital_dependent

    def is_gllb(self):
        return self.gllb

    # Initialize the GLLB functional, hopefully at this stage, the eigenvalues and functions are already available
    def initialize_gllb(self, paw):
        self.xc.pass_stuff(paw.hamiltonian.vt_sg, paw.density.nt_sg,
                           paw.kpt_u, paw.gd, paw.finegd,
                           paw.density.interpolate, paw.nspins,
                           paw.my_nuclei, paw.nuclei, paw.occupation,
                           paw.kpt_comm, paw.symmetry, paw.nvalence,
                           paw.eigensolver, paw.hamiltonian)

    def set_non_local_things(self, paw, energy_only=False):

        if not self.orbital_dependent:
            return

        if self.hybrid > 0.0:
            if paw.dtype == complex:
                raise NotImplementedError, 'k-point calculation with EXX'
            if self.parameters and self.parameters.has_key('finegrid'):
                use_finegrid = self.parameters['finegrid']
            else:
                use_finegrid = True

            self.exx = EXX(paw, energy_only, use_finegrid=use_finegrid)

        if self.xcname == 'TPSS':
            paw.density.initialize_kinetic()
            paw.density.update_kinetic(paw.kpt_u,paw.locfuncbcaster)
            if paw.nspins ==1:
                paw.hamiltonian.xc.taua_g = paw.density.taut_sg[0]
            if self.nspins == 2:
                paw.hamiltonian.xc.taua_g = paw.density.taut_sg[0]
                paw.hamiltonian.xc.taub_g = paw.density.taut_sg[1]
            for nucleus in paw.my_nuclei:
                nucleus.setup.xc_correction.initialize_kinetic(nucleus.setup.data)

    def apply_non_local(self, kpt, Htpsit_nG=None, H_nn=None):
        if self.orbital_dependent:
            if self.hybrid > 0.0:
                self.exx.apply(kpt, Htpsit_nG, H_nn, self.hybrid)

    def get_non_local_force(self, kpt):
        F_ac = 0.0
        if self.orbital_dependent:
            if self.hybrid > 0.0:
                F_ac = self.exx.force_kpoint(kpt, self.hybrid)
        return F_ac

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
    # case of setup-generator. The processes for non-local in radial and
    # 3D-grid deviate so greatly that this is special treatment is needed.
    def get_non_local_energy_and_potential1D(self, gd, u_j, f_j, e_j, l_j,
                                             v_xc, density=None):
        # Send the command one .xc up
        return self.xc.get_non_local_energy_and_potential1D(
            gd, u_j, f_j, e_j, l_j, v_xc, density=density)

    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g=None, deda2_g=None,
                             taua_g=None,dedtaua_g=None):
        if self.timer is not None:
            self.timer.start('Local xc')
        if self.mgga:
            self.xc.calculate_spinpaired(e_g.ravel(), n_g, v_g, a2_g, deda2_g,
                                         taua_g,dedtaua_g)
        elif self.gga:
            # e_g.ravel() !!!!! XXX
            self.xc.calculate_spinpaired(e_g.ravel(), n_g, v_g, a2_g, deda2_g)
        else:
            self.xc.calculate_spinpaired(e_g.ravel(), n_g, v_g)
        if self.timer is not None:
            self.timer.stop('Local xc')

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                               a2_g=None, aa2_g=None, ab2_g=None,
                               deda2_g=None, dedaa2_g=None, dedab2_g=None,
                                taua_g=None,taub_g=None,dedtaua_g=None,
                                dedtaub_g=None):
        if self.timer is not None:
            self.timer.start('Local xc')
        if self.mgga:
            #dedtau on the grid not used, only in xc_correction 
              self.xc.calculate_spinpolarized(e_g.ravel(), na_g, va_g, nb_g, vb_g,
                                           a2_g, aa2_g, ab2_g,
                                           deda2_g, dedaa2_g, dedab2_g,
                                              taua_g,taub_g,dedtaua_g,dedtaub_g)
        elif self.gga:
            self.xc.calculate_spinpolarized(e_g.ravel(), na_g, va_g, nb_g, vb_g,
                                           a2_g, aa2_g, ab2_g,
                                           deda2_g, dedaa2_g, dedab2_g)
        else:
            self.xc.calculate_spinpolarized(e_g.ravel(), na_g, va_g, nb_g, vb_g)
        if self.timer is not None:
            self.timer.stop('Local xc')

    def get_max_derivative_level(self):
        """maximal derivative level of Exc available"""
        return self.maxDerivativeLevel

    def get_name(self):
        return self.xcname

    def get_setup_name(self):
        if self.setupname is None:
            return self.get_name()
        else:
            return self.setupname

    def get_local_xc(self):
        if not self.orbital_dependent:
            return self

        if self.get_name() == 'EXX':
            return XCFunctional('LDAx', self.nspins)
        elif self.get_name() == 'oldPBE0':
            return XCFunctional('oldPBE', self.nspins)
        elif self.get_name() == 'PBE0':
            return XCFunctional('PBE', self.nspins)
        elif self.get_name().startswith('GLLB'):
            return XCFunctional('LDAx', self.nspins)
        else:
            raise RuntimeError('Orbital dependent xc-functional, but no local '
                               'functional set.')

    def exchange(self, rs, a2=0):
        return self.xc.exchange(rs, a2)

    def correlation(self, rs, zeta=0, a2=0):
        return self.xc.correlation(rs, zeta, a2)

    def calculate_xcenergy(self, na, nb,
                           sigma0=None, sigma1=None, sigma2=None,
                           taua=None,taub=None):
        # see c/libxc.c for the input and output values
        d_exc = npy.zeros(5)
        d_ex = npy.zeros(5)
        d_ec = npy.zeros(5)
        (exc, ex, ec,
         d_exc[0], d_exc[1],
         d_exc[2], d_exc[3], d_exc[4],
         d_ex[0], d_ex[1],
         d_ex[2], d_ex[3], d_ex[4],
         d_ec[0], d_ec[1],
         d_ec[2], d_ec[3], d_ec[4]
         ) = self.xc.calculate_xcenergy(na, nb,
                                        sigma0, sigma1, sigma2)
        return exc, ex, ec, d_exc, d_ex, d_ec

class XCGrid:
    def __init__(self, xcfunc, gd, nspins):
        """Base class for XC3DGrid and XCRadialGrid."""

        self.gd = gd
        self.nspins = nspins

        if isinstance(xcfunc, str):
            xcfunc = XCFunctional(xcfunc, self.nspins)
        self.set_functional(xcfunc)

        # flag is true if functional comes from libxc,
        # it is used in calculate_spinpolarized GGA's
        self.uses_libxc = self.get_functional().uses_libxc

    def set_functional(self, xcfunc):
        self.xcfunc = xcfunc

    def get_functional(self):
        return self.xcfunc

    def get_energy_and_potential(self, na_g, va_g, nb_g=None, vb_g=None):

        assert is_contiguous(na_g, float)
        assert is_contiguous(va_g, float)
        assert na_g.shape == va_g.shape == self.shape
        if nb_g is None:
            return self.get_energy_and_potential_spinpaired(na_g, va_g)
        else:
            assert is_contiguous(nb_g, float)
            assert is_contiguous(vb_g, float)
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
            self.temp = gd.empty()
        self.e_g = gd.empty()

    # Calculates exchange energy and potential.
    # The energy density will be returned on reference e_g if it is specified.
    # Otherwise the method will use self.e_g
    def get_energy_and_potential_spinpaired(self, n_g, v_g, e_g=None):

        if e_g == None:
            e_g = self.e_g

        if self.xcfunc.mgga:
            for c in range(3):
                self.ddr[c](n_g, self.dndr_cg[c])
            npy.sum(self.dndr_cg**2, axis=0, out=self.a2_g)
            self.xcfunc.calculate_spinpaired(e_g, n_g, v_g,
                                             self.a2_g,
                                             self.deda2_g,
                                             self.taua_g,self.temp)
            tmp_g = self.dndr_cg[0]
            for c in range(3):
                self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                v_g -= 2.0 * tmp_g

        elif self.xcfunc.gga:
            for c in range(3):
                self.ddr[c](n_g, self.dndr_cg[c])

            npy.sum(self.dndr_cg**2, axis=0, out=self.a2_g)

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

        return e_g.sum() * self.dv

    def get_energy_and_potential_spinpolarized(self, na_g, va_g, nb_g, vb_g, e_g=None):
        if e_g == None:
            e_g = self.e_g

        if self.xcfunc.mgga:
            for c in range(3):
                self.ddr[c](na_g, self.dnadr_cg[c])
                self.ddr[c](nb_g, self.dnbdr_cg[c])
            self.dndr_cg[:] = self.dnadr_cg + self.dnbdr_cg
            npy.sum(self.dndr_cg**2, axis=0, out=self.a2_g)
            npy.sum(self.dnadr_cg**2, axis=0, out=self.aa2_g)
            npy.sum(self.dnbdr_cg**2, axis=0, out=self.ab2_g)

            self.xcfunc.calculate_spinpolarized(e_g,
                                                na_g, va_g,
                                                nb_g, vb_g,
                                                self.a2_g,
                                                self.aa2_g, self.ab2_g,
                                                self.deda2_g,
                                                self.dedaa2_g,
                                                self.dedab2_g,
                                                self.taua_g,
                                                self.taub_g,
                                                self.temp,
                                                self.temp)
            tmp_g = self.a2_g
            for c in range(3):
                if not self.uses_libxc:
                    self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                    va_g -= 2.0 * tmp_g
                    vb_g -= 2.0 * tmp_g
                    self.ddr[c](self.dedaa2_g * self.dnadr_cg[c], tmp_g)
                    va_g -= 4.0 * tmp_g
                    self.ddr[c](self.dedab2_g * self.dnbdr_cg[c], tmp_g)
                    vb_g -= 4.0 * tmp_g

        elif self.xcfunc.gga:
            for c in range(3):
                self.ddr[c](na_g, self.dnadr_cg[c])
                self.ddr[c](nb_g, self.dnbdr_cg[c])
            self.dndr_cg[:] = self.dnadr_cg + self.dnbdr_cg
            npy.sum(self.dndr_cg**2, axis=0, out=self.a2_g)
            npy.sum(self.dnadr_cg**2, axis=0, out=self.aa2_g)
            npy.sum(self.dnbdr_cg**2, axis=0, out=self.ab2_g)
            self.xcfunc.calculate_spinpolarized(e_g,
                                                na_g, va_g,
                                                nb_g, vb_g,
                                                self.a2_g,
                                                self.aa2_g, self.ab2_g,
                                                self.deda2_g,
                                                self.dedaa2_g,
                                                self.dedab2_g)
            tmp_g = self.a2_g
            for c in range(3):
                if not self.uses_libxc:
                    self.ddr[c](self.deda2_g * self.dndr_cg[c], tmp_g)
                    va_g -= 2.0 * tmp_g
                    vb_g -= 2.0 * tmp_g
                    self.ddr[c](self.dedaa2_g * self.dnadr_cg[c], tmp_g)
                    va_g -= 4.0 * tmp_g
                    self.ddr[c](self.dedab2_g * self.dnbdr_cg[c], tmp_g)
                    vb_g -= 4.0 * tmp_g
                else: # libxc uses https://wiki.fysik.dtu.dk/gpaw/GGA
                    # see also:
                    # http://www.cse.scitech.ac.uk/ccg/dft/design.html
                    self.ddr[c](self.deda2_g * self.dnadr_cg[c], tmp_g)
                    vb_g -= tmp_g
                    self.ddr[c](self.deda2_g * self.dnbdr_cg[c], tmp_g)
                    va_g -= tmp_g
                    self.ddr[c](self.dedaa2_g * self.dnadr_cg[c], tmp_g)
                    va_g -= 2.0 * tmp_g
                    self.ddr[c](self.dedab2_g * self.dnbdr_cg[c], tmp_g)
                    vb_g -= 2.0 * tmp_g
        else:
            self.xcfunc.calculate_spinpolarized(e_g,
                                                na_g, va_g,
                                                nb_g, vb_g)
        return e_g.sum() * self.dv


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
            self.dndr_g = npy.empty(self.shape)
            self.a2_g = npy.empty(self.shape)
            self.deda2_g = npy.empty(self.shape)
            if self.nspins == 2:
                self.dnadr_g = npy.empty(self.shape)
                self.dnbdr_g = npy.empty(self.shape)
                self.aa2_g = npy.empty(self.shape)
                self.ab2_g = npy.empty(self.shape)
                self.dedaa2_g = npy.empty(self.shape)
                self.dedab2_g = npy.empty(self.shape)

        self.e_g = npy.empty(self.shape)

    # True, if this xc-potential depends on more than just density
    def is_non_local(self):
        return self.xcfunc.is_non_local()

    # This is called from all_electron.py
    # Special function for just 1D-case
    def get_non_local_energy_and_potential(self, u_j, f_j, e_j, l_j, v_xc, density=None):
        # Send the command one .xc up. Include also the grid descriptor.
        return self.xcfunc.get_non_local_energy_and_potential1D(self.gd, u_j, f_j, e_j, l_j, v_xc, density=density)

    def get_energy_and_potential_spinpaired(self, n_g, v_g, e_g = None):
        if e_g == None:
            e_g = self.e_g

        if self.xcfunc.mgga:
            self.rgd.derivative(n_g, self.dndr_g)
            self.a2_g[:] = self.dndr_g**2
            self.xcfunc.calculate_spinpaired(e_g,n_g, v_g,self.a2_g,
                                             self.deda2_g, self.taua_g)
            tmp_g = self.dndr_g
            self.rgd.derivative2(self.dv_g * self.deda2_g *
                                 self.dndr_g, tmp_g)
            tmp_g[1:] /= self.dv_g[1:]
            tmp_g[0] = tmp_g[1]
            v_g -= 2.0 * tmp_g

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

        return npy.dot(self.e_g.ravel(), self.dv_g)

    def get_energy_and_potential_spinpolarized(self, na_g, va_g, nb_g, vb_g):
        if self.xcfunc.mgga:
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
                                                self.dedaa2_g,
                                                self.dedab2_g,
                                                self.taua_g, self.taub_g)
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
                                                self.dedaa2_g,
                                                self.dedab2_g)
            tmp_g = self.a2_g

            if not self.uses_libxc:
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
                # libxc uses https://wiki.fysik.dtu.dk/gpaw/GGA
                # see also:
                # http://www.cse.scitech.ac.uk/ccg/dft/design.html
                self.rgd.derivative2(self.dv_g * self.deda2_g *
                                     self.dnadr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                vb_g -= tmp_g
                self.rgd.derivative2(self.dv_g * self.deda2_g *
                                     self.dnbdr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                va_g -= tmp_g
                self.rgd.derivative2(self.dv_g * self.dedaa2_g *
                                     self.dnadr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                va_g -= 2.0 * tmp_g
                self.rgd.derivative2(self.dv_g * self.dedab2_g *
                                     self.dnbdr_g, tmp_g)
                tmp_g[1:] /= self.dv_g[1:]
                tmp_g[0] = tmp_g[1]
                vb_g -= 2.0 * tmp_g
        else:
            self.xcfunc.calculate_spinpolarized(self.e_g,
                                                na_g, va_g,
                                                nb_g, vb_g)

        return npy.dot(self.e_g, self.dv_g)

class vxcOperator(list):
    """vxc as operator object"""
    def __init__(self, v):
        print "<vxcOperator::__init__> type(v)=",v
        print "<vxcOperator::__init__> type(v)=",v.shape
        # init the local part
        list.__init__(self,v)
        # lists for the operator part
