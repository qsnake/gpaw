import Numeric as num

from gpaw.utilities import pack
from gpaw.localized_functions import create_localized_functions


class PairDensity:
    def  __init__(self, paw):
        """basic initialisation knowing"""

        self.density = paw.density

        self.ghat_nuclei = paw.ghat_nuclei
        
        # we need to set Ghat_nuclei and Ghat_L
        # on the course grid if not initialized already
        if not hasattr(paw, 'Ghat_nuclei'):
            Ghat_nuclei = []
            create = create_localized_functions
            for nucleus in paw.nuclei:
                # Shape functions:
                ghat_l = nucleus.setup.ghat_l
                Ghat_L = create(ghat_l, paw.gd, nucleus.spos_c,
                                lfbc=paw.locfuncbcaster)
                nucleus.Ghat_L = Ghat_L
                if Ghat_L is not None:
                    Ghat_nuclei.append(nucleus)
                Ghat_nuclei.sort()
            paw.Ghat_nuclei = Ghat_nuclei

        self.Ghat_nuclei = paw.Ghat_nuclei

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

        if finegrid:
            ghat_nuclei = self.ghat_nuclei
        else:
            ghat_nuclei = self.Ghat_nuclei
        
        # Determine the compensation charges for each nucleus
        for nucleus in ghat_nuclei:
            if nucleus.in_this_domain:
                # Generate density matrix
                Pi_i = nucleus.P_uni[self.u, self.i]
                Pj_i = nucleus.P_uni[self.u, self.j]
                D_ii = num.outerproduct(Pi_i, Pj_i)
                # allowed to pack as used in the scalar product with
                # the symmetric array Delta_pL
                D_p  = pack(D_ii, tolerance=1e30)
                    
                # Determine compensation charge coefficients:
                Q_L = num.dot(D_p, nucleus.setup.Delta_pL)
            else:
                Q_L = None
                
            # Add compensation charges
            if finegrid:
                nucleus.ghat_L.add(rhot, Q_L, communicate=True)
            else:
                nucleus.Ghat_L.add(rhot, Q_L, communicate=True)
                
        return rhot 
