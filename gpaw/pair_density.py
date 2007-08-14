class PairDensity:
    def __init__(self,paw,i,j,spin=0):
        self.i=i
        self.j=j
        self.spin=spin
        self.paw=paw
        
        self.wfi = paw.kpt_u[spin].psit_nG[i]
        self.wfj = paw.kpt_u[spin].psit_nG[j]

    def get(self,finegrid=False):
        """Get pair density"""
        nijt = self.wfi*self.wfj
        if not finegrid:
            return nijt 

        # interpolate the pair density to the fine grid
        nijt_g = self.paw.finegd.new_array()
        self.paw.density.interpolate(nijt,nijt_g)

        return nijt_g

    def width_compensation_charges(self,finegrid=False):
        """Get pair densisty including the compensation charges"""
        rhot = self.GetPairDensity(finegrid)
        
        # Determine the compensation charges for each nucleus
        for nucleus in self.paw.ghat_nuclei:
            if nucleus.in_this_domain:
                # Generate density matrix
                Pi_i = nucleus.P_uni[self.vspin,self.i]
                Pj_i = nucleus.P_uni[self.vspin,self.j]
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
                                            self.gd, nucleus.spos_c,
                                            lfbc=self.paw.locfuncbcaster)
                nucleus.Ghat_L.add(rhot, Q_L, communicate=True)
                
        return rhot 
