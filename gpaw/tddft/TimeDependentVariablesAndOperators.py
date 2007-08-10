# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements classes for time-dependent variables and 
operators."""

# Hamiltonian
class TimeDependentHamiltonian:
    """ Time-dependent Hamiltonian, H(t)
    
    This class contains information required to apply time-dependent
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, pt_nuclei, hamiltonian, td_potential):
        """ Create the TimeDependentHamiltonian-object.
        
        The time-dependent potential object must (be None or) have a member
        function get_potential(self,time), which provides the
        time-dependent external potential on a fine grid at the given time.
        
        ============= ========================================================
        Parameters:
        ============= ========================================================
        pt_nuclei     projector functions (paw.pt_nuclei)
        hamiltonian   time-independent Hamiltonian (paw.hamiltonian)
        td_potential  time-dependent potential
        ============= ========================================================

        """
        
        self.pt_nuclei = pt_nuclei
        self.hamiltonian = hamiltonian
        self.td_potential = td_potential
        self.time = 0
        if ( hamiltonian.vext_g ):
            self.ti_vext_g = hamiltonian.vext_g
        else:
            self.ti_vext_g = hamiltonian.finegd.zeros()
        self.td_vext_g = hamiltonian.finegd.zeros()
        self.vext_g = hamiltonian.finegd.zeros()
        
        
    def update(self, density, time):
        """Updates the time-dependent Hamiltonian.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        density     the density at the given time 
                    (paw.density or TimeDependentDensity.get_density())
        time        the current time
        =========== ==========================================================

        """
        
        self.time = time
        if ( self.td_potential != None ):
            self.td_vext_g = self.td_potential.get_potential(self.time)
        self.vext_g = self.ti_vext_g + self.td_vext_g
        self.hamiltonian.vext_g = self.vext_g
        self.hamiltonian.update(density)
        
        
    def apply(self, kpt, psit, hpsit):
        """Applies the time-dependent Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        kpt         the current k-point (paw.kpt_u[index_of_k-pointt])
        psit        the wavefuntion (on a coarse grid) 
                    (paw.kpt_u[index_of_k-point].psit_nG[index_of_wavefunc])
        hpsit       the resulting "operated wavefunction" (H psit)
        =========== ==========================================================

        """
        kpt.apply_hamiltonian(self.hamiltonian,
                              psit[None, ...], hpsit[None, ...])



# Overlap
class TimeDependentOverlap:    
    """Time-dependent overlap operator S(t)
    
    This class contains information required to apply time-dependent
    overlap operator to a wavefunction.
    """
    
    def __init__(self, pt_nuclei):
        """Creates the TimeDependentOverlap-object.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        pt_nuclei   projector functions (paw.pt_nuclei)
        =========== ==========================================================

        """
        self.pt_nuclei = pt_nuclei
    
    def update(self):
        """Updates the time-dependent overlap operator. !Currently does nothing!
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        None
        =========== ==========================================================
        """
        # !!! FIX ME !!! update overlap operator/projectors/...
        pass
    
    def apply(self, kpt, psit, spsit):        
        """Applies the time-dependent overlap operator to the wavefunction 
        psit of the k-point kpt.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        kpt         the current k-point (paw.kpt_u[index_of_k-pointt])
        psit        the wavefuntion (on a coarse grid) 
                    (paw.kpt_u[index_of_k-point].psit_nG[index_of_wavefunc])
        spsit       the resulting "operated wavefunction" (S psit)
        =========== ==========================================================

        """
        kpt.apply_overlap(self.pt_nuclei, psit[None, ...], spsit[None, ...])
        


# Density
class TimeDependentDensity:
    """Time-dependent density rho(t)
    
    This class contains information required to get the time-dependent
    density.
    """
    
    def __init__(self, paw):
        """Creates the TimeDependentDensity-object.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        paw         the PAW-object
        =========== ==========================================================

        """
        self.paw = paw
        
    def update(self):
        """Updates the time-dependent density. !Currently does nothing!
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        None
        =========== ==========================================================

        """
        self.paw.density.update(self.paw.kpt_u, self.paw.symmetry)
        
    def get_density(self):
        """Returns the current density.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        None
        =========== ==========================================================

        """
        return self.paw.density
