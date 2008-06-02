# Written by Lauri Lehtovaara, 2007

"""This module implements classes for time-dependent variables and 
operators."""

import numpy as npy

from gpaw.polynomial import Polynomial
from gpaw.external_potential import ExternalPotential
from gpaw.mpi import run


# Hamiltonian
class TimeDependentHamiltonian:
    """ Time-dependent Hamiltonian, H(t)
    
    This class contains information required to apply time-dependent
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, pt_nuclei, hamiltonian, td_potential):
        """ Create the TimeDependentHamiltonian-object.
        
        The time-dependent potential object must (be None or) have a member
        function strength(self,time), which provides the strength of the
        time-dependent external potential to x-direction at the given time.
        
        Parameters
        ----------
        pt_nuclei: List of ?LocalizedFunctions?
            projector functions (paw.pt_nuclei)
        hamiltonian: Hamiltonian
            time-independent Hamiltonian
        td_potential: TimeDependentPotential
            time-dependent potential
        """
        
        self.pt_nuclei = pt_nuclei
        self.hamiltonian = hamiltonian
        self.td_potential = td_potential
        self.time = self.old_time = 0
        
        # internal smooth potential
        self.vt_sG = hamiltonian.gd.zeros(n=hamiltonian.nspins)

        # Increase the accuracy of Poisson solver
        self.hamiltonian.poisson_eps = 1e-12

        # external potential
        #if hamiltonian.vext_g is None:
        #    hamiltonian.vext_g = hamiltonian.finegd.zeros()

        #self.ti_vext_g = hamiltonian.vext_g
        #self.td_vext_g = hamiltonian.finegd.zeros(n=hamiltonian.nspins)

        # internal PAW-potential
        self.H_asp = [
            npy.zeros(nucleus.H_sp.shape)
            for nucleus in hamiltonian.my_nuclei
            ]


    def update(self, density, time):
        """Updates the time-dependent Hamiltonian.
    
        Parameters
        ----------
        density: Density
            the density at the given time 
            (TimeDependentDensity.get_density())
        time: float
            the current time

        """

        self.old_time = self.time = time
        self.hamiltonian.update(density)
        
    def half_update(self, density, time):
        """Updates the time-dependent Hamiltonian, in such a way, that a
        half of the old Hamiltonian is kept and the other half is updated.
        
        Parameters
        ----------
        density: Density
            the density at the given time 
            (TimeDependentDensity.get_density())
        time: float
            the current time

        """
        
        self.old_time = self.time
        self.time = time

        # copy old        
        self.vt_sG[:] = self.hamiltonian.vt_sG
        for a in range(len(self.hamiltonian.my_nuclei)):
            self.H_asp[a][:] = self.hamiltonian.my_nuclei[a].H_sp
        # update
        self.hamiltonian.update(density)
        # average
        self.hamiltonian.vt_sG += self.vt_sG
        self.hamiltonian.vt_sG *= .5
        for a in range(len(self.hamiltonian.my_nuclei)):
            self.hamiltonian.my_nuclei[a].H_sp += self.H_asp[a] 
            self.hamiltonian.my_nuclei[a].H_sp *= .5

        
    def apply(self, kpt, psit, hpsit):
        """Applies the time-dependent Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grid
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[index_of_wavefunc])
        hpsit: List of coarse grid
            the resulting "operated wavefunctions" (H psit)

        """
        self.hamiltonian.apply(psit, hpsit, kpt)
        if self.td_potential is not None:
            strength = self.td_potential.strength
            ExternalPotential().add_linear_field( self.pt_nuclei, psit, hpsit,
                                                  .5*strength(self.time)
                                                  + .5*strength(self.old_time),
                                                  kpt )


# AbsorptionKickHamiltonian
class AbsorptionKickHamiltonian:
    """ Absorption kick Hamiltonian, p.r
    
    This class contains information required to apply absorption kick 
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, pt_nuclei, strength = [0.0, 0.0, 1e-4]):
        """ Create the AbsorptionKickHamiltonian-object.

        Parameters
        ----------
        pt_nuclei: List of ?LocalizedFunctions?
            projector functions (pt_nuclei)
        strength: float[3]
            strength of the delta field to different directions

        """

        self.pt_nuclei = pt_nuclei

        # magnitude
        magnitude = npy.sqrt(strength[0]*strength[0] 
                             + strength[1]*strength[1] 
                             + strength[2]*strength[2])
        # iterations
        self.iterations = int(round(magnitude / 1.0e-4))
        if self.iterations < 1:
            self.iterations = 1
        # delta p
        self.dp = strength / self.iterations

        # hamiltonian
        self.abs_hamiltonian = npy.array([self.dp[0], self.dp[1], self.dp[2]])
        
    def update(self, density, time):
        """Dummy function = does nothing. Required to have correct interface.
        
        Parameters
        ----------
        density: Density or None
            the density at the given time or None (ignored)
        time: Float or None
            the current time (ignored)

        """
        pass
        
    def half_update(self, density, time):
        """Dummy function = does nothing. Required to have correct interface.
        
        Parameters
        ----------
        density: Density or None
            the density at the given time or None (ignored)
        time: float or None
            the current time (ignored)

        """
        pass
        
    def apply(self, kpt, psit, hpsit):
        """Applies the absorption kick Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[index_of_wavefunc])
        hpsit: List of coarse grids
            the resulting "operated wavefunctions" (H psit)

        """
        hpsit[:] = 0.0
        ExternalPotential().add_linear_field( self.pt_nuclei, psit, hpsit,
                                              self.abs_hamiltonian, kpt )


# Overlap
class TimeDependentOverlap:
    """Time-dependent overlap operator S(t)
    
    This class contains information required to apply time-dependent
    overlap operator to a wavefunction.
    """
    
    def __init__(self, overlap):
        """Creates the TimeDependentOverlap-object.
        
        Parameters
        ----------
        pt_nuclei: List of ?LocalizedFunctions?   
            projector functions (pt_nuclei)

        """
        self.overlap = overlap
    
    def update(self):
        """Updates the time-dependent overlap operator. !Currently does nothing!
        
        Parameters
        ----------
        None
        """
        # !!! FIX ME !!! update overlap operator/projectors/...
        pass
    
    def half_update(self):
        """Updates the time-dependent overlap operator, in such a way, 
        that a half of the old overlap operator is kept and the other half 
        is updated. !Currently does nothing!

        Parameters
        ----------
        None
        """
        # !!! FIX ME !!! update overlap operator/projectors/...
        pass
    
    def apply(self, kpt, psit, spsit):
        """Applies the time-dependent overlap operator to the wavefunction 
        psit of the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[index_of_wavefunc])
        spsit: List of coarse grids
            the resulting "operated wavefunctions" (S psit)

        """
        self.overlap.apply(psit, spsit, kpt)



# Density
class TimeDependentDensity:
    """Time-dependent density rho(t)
    
    This class contains information required to get the time-dependent
    density.
    """
    
    def __init__(self, paw):
        """Creates the TimeDependentDensity-object.
        
        Parameters
        ----------
        paw: PAW
            the PAW-object
        """
        self.paw = paw
        
    def update(self):
        """Updates the time-dependent density.
        
        Parameters
        ----------
        None

        """
        for kpt in self.paw.kpt_u:
            run([nucleus.calculate_projections(kpt)
                 for nucleus in self.paw.pt_nuclei])
        self.paw.density.update(self.paw.kpt_u, self.paw.symmetry)
       
    def get_density(self):
        """Returns the current density.
        
        Parameters
        ----------
        None

        """
        return self.paw.density
