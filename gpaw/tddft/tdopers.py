# Written by Lauri Lehtovaara, 2007

"""This module implements classes for time-dependent variables and
operators."""

import numpy as npy

from gpaw.polynomial import Polynomial
from gpaw.external_potential import ExternalPotential
from gpaw.mpi import run


# Hamiltonian
class TimeDependentHamiltonian:
    """Time-dependent Hamiltonian, H(t)
    
    This class contains information required to apply time-dependent
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, wfs, hamiltonian, td_potential):
        """Create the TimeDependentHamiltonian-object.
        
        The time-dependent potential object must (be None or) have a member
        function strength(self,time), which provides the strength of the
        time-dependent external potential to x-direction at the given time.
        
        Parameters
        ----------
        wfs: GridWaveFunctions
            time-independent grid-based wavefunctions
        hamiltonian: Hamiltonian
            time-independent Hamiltonian
        td_potential: TimeDependentPotential
            time-dependent potential
        """

        self.wfs = wfs
        self.hamiltonian = hamiltonian
        self.td_potential = td_potential
        self.time = self.old_time = 0
        
        # internal smooth potential
        self.vt_sG = hamiltonian.gd.zeros(hamiltonian.nspins)

        # Increase the accuracy of Poisson solver
        self.hamiltonian.poisson.eps = 1e-12

        # external potential
        #if hamiltonian.vext_g is None:
        #    hamiltonian.vext_g = hamiltonian.finegd.zeros()

        #self.ti_vext_g = hamiltonian.vext_g
        #self.td_vext_g = hamiltonian.finegd.zeros(n=hamiltonian.nspins)

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
        dH_asp = {}
        for a, dH_sp in self.hamiltonian.dH_asp.items():
            dH_asp[a] = dH_sp.copy()
        # update
        self.hamiltonian.update(density)
        # average
        self.hamiltonian.vt_sG += self.vt_sG
        self.hamiltonian.vt_sG *= .5
        for a, dH_sp in self.hamiltonian.dH_asp.items():
            dH_sp += dH_asp[a] 
            dH_sp *= 0.5
        
    def apply(self, kpt, psit, hpsit, calculate_P_ani=True):
        """Applies the time-dependent Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grid
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        hpsit: List of coarse grid
            the resulting "operated wavefunctions" (H psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """

        self.hamiltonian.apply(psit, hpsit, self.wfs, kpt, calculate_P_ani)

        if self.td_potential is not None:
            #TODO on shaky ground here...
            strength = self.td_potential.strength
            ExternalPotential().add_linear_field(psit, hpsit,
                                                 0.5 * strength(self.time) +
                                                 0.5 * strength(self.old_time),
                                                 kpt)


# AbsorptionKickHamiltonian
class AbsorptionKickHamiltonian:
    """Absorption kick Hamiltonian, p.r
    
    This class contains information required to apply absorption kick
    Hamiltonian to a wavefunction.
    """
    
    def __init__(self, wfs, atoms, strength=[0.0, 0.0, 1e-3]):
        """Create the AbsorptionKickHamiltonian-object.

        Parameters
        ----------
        wfs: GridWaveFunctions
            time-independent grid-based wavefunctions
        atoms: Atoms
            list of atoms
        strength: float[3]
            strength of the delta field to different directions

        """

        self.wfs = wfs
        self.spos_ac = atoms.get_scaled_positions() % 1.0
        
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
        
    def apply(self, kpt, psit, hpsit, calculate_P_ani=True):
        """Applies the absorption kick Hamiltonian to the wavefunction psit of
        the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        hpsit: List of coarse grids
            the resulting "operated wavefunctions" (H psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """
        hpsit[:] = 0.0

        #TODO on shaky ground here...
        ExternalPotential().add_linear_field(self.wfs, self.spos_ac,
                                             psit, hpsit,
                                             self.abs_hamiltonian, kpt)


# Overlap
class TimeDependentOverlap:
    """Time-dependent overlap operator S(t)
    
    This class contains information required to apply time-dependent
    overlap operator to a wavefunction.
    """
    
    def __init__(self, wfs):
        """Creates the TimeDependentOverlap-object.
        
        Parameters
        ----------
        wfs: GridWaveFunctions
            time-independent grid-based wavefunctions

        """
        self.wfs = wfs
        self.overlap = wfs.overlap

    def update_k_point_projections(self, kpt, psit=None):
        """Updates the projector function overlap integrals
        with the wavefunctions of a given k-point.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids (optional)
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])

        """
        if psit is not None:
            self.wfs.pt.integrate(psit, kpt.P_ani, kpt.q)
        else:
            self.wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)

    def update(self):
        """Updates the time-dependent overlap operator.
        
        Parameters
        ----------
        None

        """
        for kpt in self.wfs.kpt_u:
            self.update_k_point_projections(kpt)
    
    def half_update(self):
        """Updates the time-dependent overlap operator, in such a way,
        that a half of the old overlap operator is kept and the other half
        is updated. !Currently does nothing!

        Parameters
        ----------
        None

        """
        #for kpt in self.wfs.kpt_u:
        #    # copy old
        #    P_ani = {}
        #    for a,P_ni in kpt.P_ani.items():
        #        P_ani[a] = P_ni.copy()
        #    # update
        #    self.update_k_point_projections(kpt)
        #    # average
        #    for a,P_ni in P_ani.items():
        #        kpt.P_ani[a] += P_ni
        #        kpt.P_ani[a] *= .5

        # !!! FIX ME !!! update overlap operator/projectors/...
        pass
    
    def apply(self, kpt, psit, spsit, calculate_P_ani=True):
        """Apply the time-dependent overlap operator to the wavefunction
        psit of the k-point kpt.
        
        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        spsit: List of coarse grids
            the resulting "operated wavefunctions" (S psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_ani are used

        """
        self.overlap.apply(psit, spsit, self.wfs, kpt, calculate_P_ani)

    def apply_inverse(self, kpt, psit, sinvpsit, calculate_P_ani=True):
        """Apply the approximative time-dependent inverse overlap operator
        to the wavefunction psit of the k-point kpt.

        Parameters
        ----------
        kpt: Kpoint
            the current k-point (kpt_u[index_of_k-point])
        psit: List of coarse grids
            the wavefuntions (on coarse grid) 
            (kpt_u[index_of_k-point].psit_nG[indices_of_wavefunc])
        sinvpsit: List of coarse grids
            the resulting "operated wavefunctions" (S^(-1) psit)
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | psit> are calculated.
            When False, existing P_uni are used

        """
        self.overlap.apply_inverse(psit, sinvpsit, self.wfs, kpt,
                                   calculate_P_ani)


# DummyDensity
class DummyDensity:
    """Implements dummy (= does nothing) density for AbsorptionKick."""

    def __init__(self, wfs):
        """Placeholder Density object for AbsorptionKick.

        Parameters
        ----------
        wfs: GridWaveFunctions
            time-independent grid-based wavefunctions

        """
        self.wfs = wfs

    def update(self):
        pass

    def get_wavefunctions(self):
        return self.wfs

    def get_density(self):
        return None


# Density
class TimeDependentDensity(DummyDensity):
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
        DummyDensity.__init__(self, paw.wfs)
        self.density = paw.density

    def update(self):
        """Updates the time-dependent density.
        
        Parameters
        ----------
        None

        """
        #for kpt in self.wfs.kpt_u:
        #    self.wfs.pt.integrate(kpt.psit_nG, kpt.P_ani)
        self.density.update(self.wfs)
       
    def get_density(self):
        """Returns the current density.
        
        Parameters
        ----------
        None

        """
        return self.density
