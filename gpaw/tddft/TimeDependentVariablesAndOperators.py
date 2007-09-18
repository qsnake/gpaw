# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements classes for time-dependent variables and 
operators."""

import Numeric as num
from gpaw.polynomial import Polynomial

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
        
        if ( td_potential != None ) :
            self.td_vext_g = hamiltonian.finegd.zeros()
        else:
            self.td_vext_g = None
            
        if ( hamiltonian.vext_g ):
            self.ti_vext_g = hamiltonian.vext_g
        else:
            self.ti_vext_g = None
            
        self.vext_g = hamiltonian.finegd.zeros()
        
        self.vt_sG = num.zeros(hamiltonian.vt_sG.shape, num.Float)
        
        
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
        # if time-dependent and independent external potentials
        if ( (self.td_potential != None) and (self.ti_vext_g != None) ):
            self.td_vext_g = self.td_potential.get_potential(self.time)
            self.vext_g = self.ti_vext_g + self.td_vext_g
        # if only time-dependent external potential
        elif ( self.td_potential != None ):
            self.td_vext_g = self.td_potential.get_potential(self.time)
            self.vext_g = self.td_vext_g
        # if only time-independent external potential
        else:
            self.vext_g = self.ti_vext_g
            
        self.hamiltonian.vext_g = self.vext_g
        self.hamiltonian.update(density)
        
        
    def half_update(self, density, time):
        """Updates the time-dependent Hamiltonian, in such a way, that a
        half of the old Hamiltonian is kept and the other half is updated.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        density     the density at the given time 
                    (paw.density or TimeDependentDensity.get_density())
        time        the current time
        =========== ==========================================================

        """
        
        self.time = time
        # if time-dependent and independent external potentials
        if ( (self.td_potential != None) and (self.ti_vext_g != None) ):
            self.td_vext_g = self.td_potential.get_potential(self.time)
            self.vext_g = self.ti_vext_g + self.td_vext_g
        # if only time-dependent external potential
        elif ( self.td_potential != None ):
            self.td_vext_g = self.td_potential.get_potential(self.time)
            self.vext_g = self.td_vext_g
        # if only time-independent external potential
        else:
            self.vext_g = self.ti_vext_g
            
        self.hamiltonian.vext_g = self.vext_g
        
        self.vt_sG[:] = self.hamiltonian.vt_sG
        self.hamiltonian.update(density)
        self.hamiltonian.vt_sG += self.vt_sG
        self.hamiltonian.vt_sG *= .5
        
        
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


# AbsorptionKickHamiltonian
class AbsorptionKickHamiltonian:
    """ Absorption kick Hamiltonian, p.r
    
    This class contains information required to apply absorption kick 
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, pt_nuclei, strength = 1e-2, direction = [0.0, 0.0, 1.0]):
        """ Create the AbsorptionKickHamiltonian-object.
                
        ============= ========================================================
        Parameters:
        ============= ========================================================
        pt_nuclei     projector functions (paw.pt_nuclei)
        strength      strength of the field
        direction     (unnormalized) direction of the field
        ============= ========================================================

        """
        
        self.pt_nuclei = pt_nuclei        

        # normalized direction
        dir = num.array(direction, num.Float)
        p = strength * dir / num.sqrt(num.vdot(dir,dir))
        # iterations
        self.iterations = int(round(strength / 1.0e-3))
        if ( self.iterations == 0 ): self.iterations = 1
        # delta p
        self.dp = p / self.iterations

        # hamiltonian
        # FIXME: slow! Should use special class instead of Polynomial
        coords = [ [0.0, 0.0, 0.0], \
                   [1.0, 0.0, 0.0], \
                   [0.0, 1.0, 0.0], \
                   [0.0, 0.0, 1.0] ]
        values = [ 0.0, self.dp[0], self.dp[1], self.dp[2] ]
        self.hamiltonian = Polynomial(values, coords, order=1)

        
    def update(self, density, time):
        """Dummy function = does nothing. Required to have correct interface.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        density     the density at the given time or None (ignored)
        time        the current time (ignored)
        =========== ==========================================================

        """
        pass
        
    def half_update(self, density, time):
        """Dummy function = does nothing. Required to have correct interface.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        density     the density at the given time or None (ignored)
        time        the current time (ignored)
        =========== ==========================================================

        """
        pass
        
    def apply(self, kpt, psit, hpsit):
        """Applies the absorption kick Hamiltonian to the wavefunction psit of
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
        kpt.apply_scalar_function( self.pt_nuclei,
                                   psit[None, ...], hpsit[None, ...], 
                                   self.hamiltonian )




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
    
    def half_update(self):
        """Updates the time-dependent overlap operator, in such a way, 
        that a half of the old overlap operator is kept and the other half 
        is updated. !Currently does nothing!
        
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
        """Updates the time-dependent density.
        
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
