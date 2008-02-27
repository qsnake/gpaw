from math import sqrt, pi
import numpy as npy

from gpaw.utilities import pack
from gpaw.utilities.tools import pick
from gpaw.localized_functions import create_localized_functions


class PairDensity2:
    def  __init__(self, paw, finegrid):
        """Initialization needs a paw instance, and whether the compensated
        pair density should be on the fine grid (boolean)"""

        self.interpolate = paw.density.interpolate
        self.finegrid = finegrid

        self.ghat_nuclei = paw.ghat_nuclei
        self.gd = paw.gd

        self.yes_I_have_done_the_Ghat_L = False

    def set_coarse_ghat(self):
        create = create_localized_functions
        for nucleus in self.ghat_nuclei:
            # Shape functions:
            ghat_l = nucleus.setup.ghat_l
            Ghat_L = create(ghat_l, self.gd, nucleus.spos_c,
                            forces=False)
            nucleus.Ghat_L = Ghat_L
    
            if Ghat_L is not None:
                assert nucleus.ghat_L is not None
                Ghat_L.set_communicator(nucleus.ghat_L.comm,
                                        nucleus.ghat_L.root)
    
        self.yes_I_have_done_the_Ghat_L = True
    
        for nucleus in self.ghat_nuclei:
            Ghat_L = nucleus.Ghat_L
            if Ghat_L is None:
                ghat_L = nucleus.ghat_L
                I_i = npy.zeros(ghat_L.ni)
                ghat_L.comm.sum(I_i)
            else:
                Ghat_L.normalize(sqrt(4 * pi))

    def initialize(self, kpt, n1, n2):
        """Set wave function indices."""
        if not self.finegrid and not self.yes_I_have_done_the_Ghat_L:
            # we need to set Ghat_L on the coarse grid
            self.set_coarse_ghat()

        self.n1 = n1
        self.n2 = n2
        self.u = kpt.u
        self.spin = kpt.s
        
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
            self.interpolate(nt_G, rhot_g)
        else:
            # copy values
            rhot_g[:] = nt_G
        
        # Determine the compensation charges for each nucleus
        for nucleus in self.ghat_nuclei:
            if nucleus.in_this_domain:
                # Generate density matrix
                P1_i = pick(nucleus.P_uni[self.u], self.n1)
                P2_i = pick(nucleus.P_uni[self.u], self.n2)
                D_ii = npy.outer(P1_i.conj(), P2_i)
                # allowed to pack as used in the scalar product with
                # the symmetric array Delta_pL
                D_p  = pack(D_ii, tolerance=1e30)
                    
                # Determine compensation charge coefficients:
                Q_L = npy.dot(D_p, nucleus.setup.Delta_pL)
            else:
                Q_L = None

            # Add compensation charges
            if self.finegrid:
                nucleus.ghat_L.add(rhot_g, Q_L, communicate=True)
            else:
                Ghat_L = nucleus.Ghat_L
                if Ghat_L is None:
                    ghat_L = nucleus.ghat_L
                    Q_L = npy.empty(ghat_L.ni)
                    ghat_L.comm.broadcast(Q_L, ghat_L.root)
                else:
                    Ghat_L.add(rhot_g, Q_L, communicate=True)


class PairDensity:
    def  __init__(self, paw):
        """basic initialisation knowing"""

        self.density = paw.density

        self.ghat_nuclei = paw.ghat_nuclei
        self.nuclei = paw.nuclei
        
        # we need to set Ghat_nuclei and Ghat_L
        # on the course grid if not initialized already
        if not hasattr(paw, 'yes_I_have_done_the_Ghat_L'):
            self.set_coarse_ghat(paw)

    def set_coarse_ghat(self, paw):
        create = create_localized_functions
        for nucleus in self.ghat_nuclei:
            # Shape functions:
            ghat_l = nucleus.setup.ghat_l
            Ghat_L = create(ghat_l, paw.gd, nucleus.spos_c,
                            forces=False)
            nucleus.Ghat_L = Ghat_L
    
            if Ghat_L is not None:
                assert nucleus.ghat_L is not None
                Ghat_L.set_communicator(nucleus.ghat_L.comm,
                                        nucleus.ghat_L.root)
    
        paw.yes_I_have_done_the_Ghat_L = True
    
        for nucleus in self.ghat_nuclei:
            Ghat_L = nucleus.Ghat_L
            if Ghat_L is None:
                ghat_L = nucleus.ghat_L
                I_i = npy.zeros(ghat_L.ni)
                ghat_L.comm.sum(I_i)
            else:
                Ghat_L.normalize(sqrt(4 * pi))

    def initialize(self, kpt, i, j):
        """initialize yourself with the wavefunctions"""
        self.i = i
        self.j = j
        self.u = kpt.u
        self.spin = kpt.s
        
        self.wfi = kpt.psit_nG[i]
        self.wfj = kpt.psit_nG[j]

    def get(self, finegrid=False):
        """Get pair density"""
        nijt = self.wfi * self.wfj
        if not finegrid:
            return nijt 

        # interpolate the pair density to the fine grid
        nijt_g = self.density.finegd.empty()
        self.density.interpolate(nijt, nijt_g)

        return nijt_g

    def with_compensation_charges(self, finegrid=False):
        """Get pair densisty including the compensation charges"""
        rhot = self.get(finegrid)

        ghat_nuclei = self.ghat_nuclei
        
        # Determine the compensation charges for each nucleus
        for nucleus in ghat_nuclei:
            if nucleus.in_this_domain:
                # Generate density matrix
                Pi_i = nucleus.P_uni[self.u, self.i]
                Pj_i = nucleus.P_uni[self.u, self.j]
                D_ii = npy.outer(Pi_i, Pj_i)
                # allowed to pack as used in the scalar product with
                # the symmetric array Delta_pL
                D_p  = pack(D_ii, tolerance=1e30)
                    
                # Determine compensation charge coefficients:
                Q_L = npy.dot(D_p, nucleus.setup.Delta_pL)
            else:
                Q_L = None

            # Add compensation charges
            if finegrid:
                nucleus.ghat_L.add(rhot, Q_L, communicate=True)
            else:
                Ghat_L = nucleus.Ghat_L
                if Ghat_L is None:
                    ghat_L = nucleus.ghat_L
                    Q_L = npy.empty(ghat_L.ni)
                    ghat_L.comm.broadcast(Q_L, ghat_L.root)
                else:
                    Ghat_L.add(rhot, Q_L, communicate=True)
                
        return rhot
