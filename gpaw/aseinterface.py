# pylint: disable-msg=W0142,C0103,E0201

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""ASE-calculator interface."""

import os
import weakref
from math import sqrt, pi

import numpy as npy
import ase
#from ase.parallel import register_parallel_cleanup_function
from ase.units import Bohr, Hartree

from gpaw.paw import PAW

#register_parallel_cleanup_function()


class Calculator(PAW):
    """This is the ASE-calculator frontend for doing a PAW calculation.
    """

    def __init__(self, filename=None, **kwargs):
        # Set units to ASE units:
        self.a0 = Bohr
        self.Ha = Hartree

        PAW.__init__(self, filename, **kwargs)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        
        if atoms is None:
            atoms = self.atoms

        self.calculate(atoms)

        if force_consistent:
            # Free energy:
            return self.Ha * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return self.Ha * (self.Etot + 0.5 * self.S)

    def get_forces(self, atoms):
        """Return the forces for the current state of the ListOfAtoms."""
        if self.F_ac is None:
            if hasattr(self, 'nuclei') and not self.nuclei[0].ready:
                self.converged = False
        self.calculate(atoms)
        self.calculate_forces()
        return self.F_ac * (self.Ha / self.a0)
      
    def get_stress(self, atoms):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def get_number_of_bands(self):
        """Return the number of bands."""
        return self.nbands 
  
    def get_xc_functional(self):
        """Return the XC-functional identifier.
        
        'LDA', 'PBE', ..."""
        
        return self.xc 
 
    def get_bz_k_points(self):
        """Return the k-points."""
        return self.bzk_kc
 
    def get_number_of_spins(self):
        return self.nspins

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        return self.nspins == 2
    
    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.ibzk_kc

    def get_k_point_weights(self):
        """Weights of the k-points. 
        
        The sum of all weights is one."""
        
        return self.weight_k

    def get_pseudo_valence_density(self, pad=False):
        """Return pseudo-density array."""
        if pad:
            return self.gd.zero_pad(self.get_pseudo_valence_density(False))
        return self.density.get_density_array() / self.a0**3

    def get_pseudo_density_corrections(self):
        """Integrated density corrections.

        Returns the integrated value of the difference between the pseudo-
        and the all-electron densities at each atom.  These are the numbers
        you should add to the result of doing e.g. Bader analysis on the
        pseudo density."""
        if self.nspins == 1:
            return npy.array([n.get_density_correction(0, 1)
                              for n in self.nuclei])
        else:
            return npy.array([[n.get_density_correction(spin, 2)
                              for n in self.nuclei] for spin in range(2)])

    def get_all_electron_density(self, gridrefinement=2, pad=False):
        """Return reconstructed all-electron density array."""
        if pad:
            return self.gd.zero_pad(self.get_all_electron_density(
                gridrefinement, False))
        return self.density.get_all_electron_density(gridrefinement)\
               / self.a0**3

    def get_wigner_seitz_densities(self, spin):
        """Get the weight of the spin-density in Wigner-Seitz cells
        around each atom.

        The density assigned to each atom is relative to the neutral atom,
        i.e. the density sums to zero.
        """
        from gpaw.utilities import wignerseitz
        atom_index = self.gd.empty(dtype=int)
        atom_ac = npy.array([n.spos_c * self.gd.N_c for n in self.nuclei])
        wignerseitz(atom_index, atom_ac, self.gd.beg_c, self.gd.end_c)

        nt_G = self.density.nt_sG[spin]
        weight_a = npy.empty(len(self.nuclei))
        for a, nucleus in enumerate(self.nuclei):
            # XXX Optimize! No need to integrate in zero-region
            smooth = self.gd.integrate(npy.where(atom_index == a, nt_G, .0))
            correction = nucleus.get_density_correction(spin, self.nspins)
            weight_a[a] = smooth + correction
            
        return weight_a

    def get_dos(self, spin, npts=201, width=None):
        """The total DOS.

        Fold eigenvalues with Gaussians, and put on an energy grid."""
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        w_k = self.weight_k
        Nb = self.nbands
        energies = npy.empty(len(w_k) * Nb)
        weights  = npy.empty(len(w_k) * Nb)
        x = 0
        for k, w in enumerate(w_k):
            energies[x:x + Nb] = self.get_eigenvalues(k, spin)
            weights[x:x + Nb] = w
            x += Nb
            
        from gpaw.utilities.dos import fold
        return fold(energies, weights, npts, width)        

    def get_wigner_seitz_ldos(self, a, spin, npts=201, width=None):
        """The Local Density of States, using a Wigner-Seitz basis function.

        Project wave functions onto a Wigner-Seitz box at atom ``a``, and
        use this as weight when summing the eigenvalues."""
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        from gpaw.utilities.dos import raw_wignerseitz_LDOS, fold
        energies, weights = raw_wignerseitz_LDOS(self, a, spin)
        return fold(energies * self.Ha, weights, npts, width)        
    
    def get_orbital_ldos(self, a, spin, angular, npts=201, width=None):
        """The Local Density of States, using atomic orbital basis functions.

        Project wave functions onto an atom orbital at atom ``a``, and
        use this as weight when summing the eigenvalues.

        The atomic orbital has angular momentum ``angular``, which can be
        's', 'p', 'd', 'f', or any combination (e.g. 'sdf')."""
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        from gpaw.utilities.dos import raw_orbital_LDOS, fold
        energies, weights = raw_orbital_LDOS(self, a, spin, angular)
        return fold(energies * self.Ha, weights, npts, width)

    def get_molecular_ldos(self, mol, spin, npts=201, width=None,
                           lc=None, wf=None, P_uai=None):
        """The Projected Density of States, using either atomic
        orbital basis functions (lc) for a specified molecule (mol)
        or a molecular wavefunction(wf)."""
        
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        from gpaw.utilities.dos import molecular_LDOS, fold
        energies, weights = molecular_LDOS(self, mol, spin,
                                           lc=lc, wf=wf, P_uai=P_uai)
        return fold(energies * self.Ha, weights, npts, width)

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0, broadcast=True,
                                 pad=False):
        """Return pseudo-wave-function array."""
        if pad:
            return self.gd.zero_pad(self.get_pseudo_wave_function(
                band, kpt, spin, broadcast, False))
        psit_G = self.get_wave_function_array(band, kpt, spin)
        if broadcast:
            if not self.master:
                psit_G = self.gd.empty(dtype=self.dtype, global_array=True)
            self.world.broadcast(psit_G, 0)
            return psit_G / Bohr**1.5
        elif self.master:
            return psit_G / Bohr**1.5

    def get_eigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        result = self.collect_eigenvalues(kpt, spin)
        if result is not None:
            return result * self.Ha

    def initial_wannier(self, initialwannier, kpointgrid, fixedstates,
                        edf, spin):
        """Initial guess for the shape of wannier functions.

        Use initial guess for wannier orbitals to determine rotation
        matrices U and C.
        """
        raise NotImplementedError

        return c, U

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        c = dirG.tolist().index(1)
        G = self.bzk_kc[nextkpoint, c] - self.bzk_kc[kpoint, c] + G_I[c]

        return self.get_wannier_integrals(c, spin, kpoint, nextkpoint, G)

    def get_magnetic_moment(self, atoms=None):
        """Return the total magnetic moment."""
        return self.occupation.magmom

    def get_magnetic_moments(self, atoms=None):
        """Return the local magnetic moments within augmentation spheres"""
        return self.magmom_a
        
    def get_number_of_grid_points(self):
        return self.gd.N_c

    def GetEnsembleCoefficients(self):
        """Get BEE ensemble coefficients.

        See The ASE manual_ for details.

        .. _manual: https://wiki.fysik.dtu.dk/ase/Utilities
                    #bayesian-error-estimate-bee
        """

        E = self.GetPotentialEnergy()
        E0 = self.get_xc_difference('XC-9-1.0')
        coefs = (E + E0,
                 self.get_xc_difference('XC-0-1.0') - E0,
                 self.get_xc_difference('XC-1-1.0') - E0,
                 self.get_xc_difference('XC-2-1.0') - E0)
        self.text('BEE: (%.9f, %.9f, %.9f, %.9f)' % coefs)
        return npy.array(coefs)

    def get_electronic_temperature(self):
        return self.kT * self.Ha
