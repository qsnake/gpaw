# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements classes for time-dependent variables and 
operators."""

import numpy as npy

from gpaw.polynomial import Polynomial
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
        self.time = self.old_time = 0
        
#        if td_potential is not None:
#            self.td_vext_g = hamiltonian.finegd.zeros()
#        else:
#            self.td_vext_g = None
#
#        if hamiltonian.vext_g is not None:
#            self.ti_vext_g = hamiltonian.vext_g
#        else:
#            self.ti_vext_g = None
#
#        self.vext_g = hamiltonian.finegd.zeros()

        self.vt_sG = hamiltonian.gd.zeros(hamiltonian.nspins)
        self.H_asp = [
            npy.zeros(nucleus.H_sp.shape)
            for nucleus in hamiltonian.my_nuclei
            ]


#    def set_vext_g(self):
#        # if time-dependent and independent external potentials
#        if (self.td_potential is not None) and (self.ti_vext_g is not None):
#            self.td_vext_g = self.td_potential.get_potential(self.time)
#            self.vext_g = self.ti_vext_g + self.td_vext_g
#        # if only time-dependent external potential
#        elif self.td_potential is not None:
#            self.td_vext_g = self.td_potential.get_potential(self.time)
#            self.vext_g = self.td_vext_g
#        # if only time-independent external potential
#        else:
#            self.vext_g = self.ti_vext_g


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
        
        self.old_time = self.time = time
#        self.set_vext_g()
#        self.hamiltonian.vext_g = self.vext_g
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
        
        self.old_time = time
        self.time = time
#        self.set_vext_g()
#        self.hamiltonian.vext_g = self.vext_g

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
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        kpt         the current k-point (paw.kpt_u[index_of_k-pointt])
        psit        the wavefuntion (on a coarse grid) 
                    (paw.kpt_u[index_of_k-point].psit_nG[index_of_wavefunc])
        hpsit       the resulting "operated wavefunction" (H psit)
        =========== ==========================================================

        """
        p = npy.reshape(psit, (1,) + psit.shape)
        hp = npy.reshape(hpsit, (1,) + hpsit.shape)
        self.hamiltonian.apply(p, hp, kpt)
        if self.td_potential is not None:
            strength = self.td_potential.strength
            kpt.add_linear_xfield( self.pt_nuclei, p, hp,
                                   .5 * strength(self.time)
                                   + .5 * strength(self.old_time) )

        # FIX ME: VERY BAD = SLOW, MODIFY HAMILTONIAN INSTEAD
#        class TDPotentialAverage:
#            def __init__(self, value, t0, t1):
#                self.pot = value
#                self.t0 = t0
#                self.t1 = t1
#
#            def value(self, x,y,z):
#                return .5 * self.pot(x,y,z,self.t0) \
#                    + .5 * self.pot(x,y,z,self.t1)
#
#        kpt.apply_scalar_function( self.pt_nuclei, hp, hp, 
#                                   TDPotentialAverage( self.td_potential.value, 
#                                                       self.old_time, 
#                                                       self.time ) )

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
        dir = npy.array(direction, float)
        p = strength * dir / npy.sqrt(npy.vdot(dir,dir))
        # iterations
        self.iterations = int(round(strength / 1.0e-3))
        if self.iterations < 10:
            self.iterations = 10
        # delta p
        self.dp = p / self.iterations

        # hamiltonian
        # FIXME: slow! Should use special class instead of Polynomial
        coords = [ [0.0, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0] ]
        values = [ 0.0, self.dp[0], self.dp[1], self.dp[2] ]
        self.abs_hamiltonian = Polynomial(values, coords, order=1)

        
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
        p = npy.reshape(psit, (1,) + psit.shape)
        hp = npy.reshape(hpsit, (1,) + hpsit.shape)
        kpt.apply_scalar_function( self.pt_nuclei,
                                   #psit[None, ...], hpsit[None, ...], 
                                   p, hp,
                                   self.abs_hamiltonian )



# Overlap
class TimeDependentOverlap:
    """Time-dependent overlap operator S(t)
    
    This class contains information required to apply time-dependent
    overlap operator to a wavefunction.
    """
    
    def __init__(self, overlap):
        """Creates the TimeDependentOverlap-object.
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        pt_nuclei   projector functions (paw.pt_nuclei)
        =========== ==========================================================

        """
        self.overlap = overlap
    
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
        p = npy.reshape(psit, (1,) + psit.shape)
        sp = npy.reshape(spsit, (1,) + spsit.shape)
        self.overlap.apply(p, sp, kpt)
        # self.overlap.apply(psit[None, ...], hpsit[None, ...], kpt)




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
        for kpt in self.paw.kpt_u:
            run([nucleus.calculate_projections(kpt)
                 for nucleus in self.paw.pt_nuclei])
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
