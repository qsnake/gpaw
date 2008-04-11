from math import exp
import numpy as npy
npy.seterr(all='raise')

from gpaw.xc_functional import XCFunctional, XC3DGrid, XCRadialGrid
from gpaw.gllb import SMALL_NUMBER
from gpaw.gllb.gllb1d import GLLB1D

class SAOPFunctional(GLLB1D):
    """The "Statistical Average of Orbital Potentials" functional.

    See: Schipper et al JChemPhys 112, 1344 (2000)"""
    def __init__(self, inner_xc_name, outer_xc_name):
        
        # ???
        self.relaxed_core_response = False

        # load xc's for the inner and outer reagions
##        print "<SAOPFunctional::__init__> inner_xc_name=", inner_xc_name
        self.inner_xc = XCFunctional(inner_xc_name)
        self.outer_xc = XCFunctional(outer_xc_name)

        self.xc1D = None
##        self.inner_xc3d = XC3DGrid(self.inner_xc, self.finegd, self.nspins)
##        self.outer_xc3d = XC3DGrid(self.outer_xc, self.finegd, self.nspins)

    def get_non_local_energy_and_potential1D(self, rgd, u_j, f_j, e_j, l_j,
                                             v_xc, njcore=None, density=None):
        """Used by setup generator to calculate the one dimensional potential

        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        rgd          Radial grid descriptor
        u_j         The wave functions
        f_j         The occupation numbers
        e_j         The eigenvalues
        l_j         The angular momentum quantum numbers
        v_xc        V_{xc} is added to this array.
        nj_core     If njcore is set, only response part will be returned for
                    wave functions u_j[:nj_core]
        density     If density is supplied, it overrides the density
                    calculated from orbitals.
                    This is used is setup-generation.
        =========== ==========================================================
        """

        # build the density if needed
        n_g = density
        if n_g is None:
            n_g = self.construct_density1D(rgd, u_j, f_j)
            
        # potential of inner functional
        # we assume only the inner potential to be orbital dependent

        v_inner = npy.zeros(v_xc.shape)
        self.inner_xc.xc.get_non_local_energy_and_potential1D(rgd, u_j, f_j,
                                                              e_j, l_j,
                                                              v_inner,
                                                              density=n_g)
        
        # potential and energy of outer functional

        # initialize radial grid if needed
        if self.xc1D is None:
            self.xc1D = XCRadialGrid(self.outer_xc, rgd)
        
        v_outer = npy.zeros(v_xc.shape)
        E_xc = self.xc1D.get_energy_and_potential(n_g, v_outer)

        # get weights for the outer part
        w_j = self.get_response_weights1D(u_j, f_j, e_j)
        

        # sum up the weighted contributions
        v = npy.zeros(v_xc.shape)
        for u, f, w in zip(u_j, f_j, w_j):
            u2 = u**2
            u2[1:] /= (4 * npy.pi * rgd.r_g[1:]**2)
            u2[0] = u2[1]
            v += f * ( w * v_outer + ( 1. - w ) * v_inner) * u2
        v /= n_g + SMALL_NUMBER

        # add this to the potential
        v_xc += v
        
        # return the energy
        return E_xc

    def exponential_weight(self, e):
        if e > 0:
            # this is an unoccupied state that does not contribute
            return 0
        else:
            return exp(e)
            
    def get_response_weights1D(self,  u_j, f_j, e_j):
        """
        Calculates the weights for the LBalpha in SAOP.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        u_j         The 1D-wave functions
        f_j         The occupation numbers
        e_j         Eigenvalues
        =========== ==========================================================
        """
        reference_level = self.find_reference_level1D(f_j, e_j)

        w_j = [ self.exponential_weight(e - reference_level) for e in e_j ]
        return w_j

