import Numeric as num

from gpaw.utilities import pack
from gpaw.localized_functions import create_localized_functions


class PairDensity:
    def __init__(self, density, kpt, i, j):
        self.i = i
        self.j = j
        self.u = kpt.u
        self.spin = kpt.u
        self.density = density
        
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
        
        # Determine the compensation charges for each nucleus
        for nucleus in self.density.ghat_nuclei:
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
                if not hasattr(nucleus, 'Ghat_L'):
                    # add course grid splines to this nucleus
                    create = create_localized_functions
                    nucleus.Ghat_L = create(nucleus.setup.ghat_l,
                                            self.density.gd, nucleus.spos_c)
                nucleus.Ghat_L.add(rhot, Q_L, communicate=True)
                
        return rhot 
