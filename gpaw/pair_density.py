from math import sqrt, pi
import numpy as npy

from gpaw.utilities import pack
from gpaw.utilities.tools import pick
from gpaw.lfc import LocalizedFunctionsCollection as LFC


class PairDensity2:
    def  __init__(self, density, atoms, finegrid):
        """Initialization needs a paw instance, and whether the compensated
        pair density should be on the fine grid (boolean)"""

        self.density = density
        self.finegrid = finegrid

        if not finegrid:
            density.Ghat = LFC(density.gd,
                               [setup.ghat_l
                                for setup in density.setups],
                               integral=sqrt(4 * pi))
            density.Ghat.set_positions(atoms.get_scaled_positions() % 1.0)

    def initialize(self, kpt, n1, n2):
        """Set wave function indices."""
        self.n1 = n1
        self.n2 = n2
        self.spin = kpt.s
        self.P_ani = kpt.P_ani
        self.psit1_G = pick(kpt.psit_nG, n1)
        self.psit2_G = pick(kpt.psit_nG, n2)

    def get_coarse(self, nt_G):
        """Get pair density"""
        npy.multiply(self.psit1_G.conj(), self.psit2_G, nt_G)

    def add_compensation_charges(self, nt_G, rhot_g):
        """Add compensation charges to input pair density, which
        is interpolated to the fine grid if needed."""

        if self.finegrid:
            # interpolate the pair density to the fine grid
            self.density.interpolator.apply(nt_G, rhot_g)
        else:
            # copy values
            rhot_g[:] = nt_G
        
        # Determine the compensation charges for each nucleus
        Q_aL = {}
        for a, P_ni in self.P_ani.items():
            # Generate density matrix
            P1_i = P_ni[self.n1]
            P2_i = P_ni[self.n2]
            D_ii = npy.outer(P1_i.conj(), P2_i)
            # allowed to pack as used in the scalar product with
            # the symmetric array Delta_pL
            D_p  = pack(D_ii, tolerance=1e30)
            
            # Determine compensation charge coefficients:
            Q_aL[a] = npy.dot(D_p, self.density.setups[a].Delta_pL)

        # Add compensation charges
        if self.finegrid:
            self.density.ghat.add(rhot_g, Q_aL)
        else:
            self.density.Ghat.add(rhot_g, Q_aL)


class PairDensity:
    def  __init__(self, paw):
        """basic initialisation knowing"""

        self.density = paw.density
        self.setups = paw.wfs.setups
        self.spos_ac = paw.atoms.get_scaled_positions()

    def initialize(self, kpt, i, j):
        """initialize yourself with the wavefunctions"""
        self.i = i
        self.j = j
        self.spin = kpt.s
        self.P_ani = kpt.P_ani
        
        self.wfi = kpt.psit_nG[i]
        self.wfj = kpt.psit_nG[j]

    def get(self, finegrid=False):
        """Get pair density"""
        nijt = self.wfi * self.wfj
        if not finegrid:
            return nijt 

        # interpolate the pair density to the fine grid
        nijt_g = self.density.finegd.empty()
        self.density.interpolator.apply(nijt, nijt_g)

        return nijt_g

    def with_compensation_charges(self, finegrid=False):
        """Get pair densisty including the compensation charges"""
        rhot = self.get(finegrid)

        # Determine the compensation charges for each nucleus
        Q_aL = {}
        for a, P_ni in self.P_ani.items():
            # Generate density matrix
            Pi_i = P_ni[self.i]
            Pj_i = P_ni[self.j]
            D_ii = npy.outer(Pi_i, Pj_i)
            # allowed to pack as used in the scalar product with
            # the symmetric array Delta_pL
            D_p  = pack(D_ii, tolerance=1e30)
            
            # Determine compensation charge coefficients:
            Q_aL[a] = npy.dot(D_p, self.setups[a].Delta_pL)

        # Add compensation charges
        if finegrid:
            self.density.ghat.add(rhot, Q_aL)
        else:
            if not hasattr(self.density, 'Ghat'):
                self.density.Ghat = LFC(self.density.gd,
                                        [setup.ghat_l
                                         for setup in self.setups],
                                        integral=sqrt(4 * pi))
                self.density.Ghat.set_positions(self.spos_ac)
            self.density.Ghat.add(rhot, Q_aL)
                
        return rhot
