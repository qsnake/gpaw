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
            return Hartree * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return Hartree * (self.Etot + 0.5 * self.S)

    def get_forces(self, atoms):
        """Return the forces for the current state of the ListOfAtoms."""
        if self.F_ac is None:
            if hasattr(self, 'nuclei') and not self.nuclei[0].ready:
                self.converged = False
        self.calculate(atoms)
        self.calculate_forces()
        return self.F_ac * (Hartree / Bohr)
      
    def get_stress(self, atoms):
        """Return the stress for the current state of the ListOfAtoms."""
        raise NotImplementedError

    def calculation_required(self, atoms, quantities):
        if 'stress' in quantities:
            quantities.remove('stress')

        if len(quantities) == 0:
            return False

        if not (self.initialized and self.converged):
            return True

        if (len(atoms) != len(self.atoms) or
            (atoms.get_positions() != self.atoms.get_positions()).any() or
            (atoms.get_atomic_numbers() !=
             self.atoms.get_atomic_numbers()).any() or
            (atoms.get_cell() != self.atoms.get_cell()).any() or
            (atoms.get_pbc() != self.atoms.get_pbc()).any()):
            return True

        if 'forces' in quantities:
            return self.F_ac is None

        return False

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

    def get_pseudo_density(self, spin=None, pad=True):
        """Return pseudo-density array.

        If *spin* is not given, then the total density is returned.
        Otherwise, the spin up or down density is returned (spin=0 or
        1)."""
        
        nt_sG = self.density.nt_sG
        if self.nspins == 1:
            nt_G = nt_sG[0]
            if spin is not None:
                nt_G = 0.5 * nt_G
        else:
            if spin is None:
                nt_G = nt_sG.sum(axis=0)
            else:
                nt_G = nt_sG[spin]
        if pad:
            nt_G = self.gd.zero_pad(nt_G)
        return nt_G / Bohr**3

    get_pseudo_valence_density = get_pseudo_density  # Don't use this one!
    
    def get_effective_potential(self, spin=0, pad=True):
        """Return pseudo effective-potential."""
        vt_G = self.hamiltonian.vt_sG[spin]
        if pad:
            vt_G = self.gd.zero_pad(vt_G)
        return vt_G * Hartree
    
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

    def get_all_electron_density(self, spin=None, gridrefinement=2, pad=True):
        """Return reconstructed all-electron density array."""
        n_G = self.density.get_all_electron_density(gridrefinement)
        if n_G is None:
            return npy.array([0.]) # let the slave return something
        
        if self.nspins == 1 and spin is not None:
            n_G *= .5
        elif self.nspins == 2:
            if spin is None:
                n_G = n_G.sum(axis=0)
            else:
                n_G = n_G[spin]
        
        if pad:
            n_G = self.gd.zero_pad(n_G)
        return n_G / Bohr**3

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

    def get_dos(self, spin=0, npts=201, width=None):
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

    def get_wigner_seitz_ldos(self, a, spin=0, npts=201, width=None):
        """The Local Density of States, using a Wigner-Seitz basis function.

        Project wave functions onto a Wigner-Seitz box at atom ``a``, and
        use this as weight when summing the eigenvalues."""
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0:
            width = 0.1

        from gpaw.utilities.dos import raw_wignerseitz_LDOS, fold
        energies, weights = raw_wignerseitz_LDOS(self, a, spin)
        return fold(energies * Hartree, weights, npts, width)
    
    def get_orbital_ldos(self, a,
                         spin=0, angular='spdf', npts=201, width=None):
        """The Local Density of States, using atomic orbital basis functions.

        Project wave functions onto an atom orbital at atom ``a``, and
        use this as weight when summing the eigenvalues.

        The atomic orbital has angular momentum ``angular``, which can be
        's', 'p', 'd', 'f', or any combination (e.g. 'sdf').

        An integer value for ``angular`` can also be used to specify a specific
        projector function to project onto.
        """
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        from gpaw.utilities.dos import raw_orbital_LDOS, fold
        energies, weights = raw_orbital_LDOS(self, a, spin, angular)
        return fold(energies * Hartree, weights, npts, width)

    def get_all_electron_ldos(self, mol, spin=0, npts=201, width=None,
                              wf_k=None, P_aui=None, lc=None, raw=False):
        """The Projected Density of States, using all-electron wavefunctions.

        Projects onto a pseudo_wavefunctions (wf_k) corresponding to some band
        n and uses P_aui ([paw.nuclei[a].P_uni[:,n,:] for a in atoms]) to
        obtain the all-electron overlaps.
        Instead of projecting onto a wavefunctions a molecular orbital can
        be specified by a linear combination of weights (lc)
        """
        from gpaw.utilities.dos import all_electron_LDOS, fold

        if raw:
            return all_electron_LDOS(self, mol, spin, lc=lc,
                                     wf_k=wf_k, P_aui=P_aui)
        if width is None:
            width = self.get_electronic_temperature()
        if width == 0.0:
            width = 0.1

        energies, weights = all_electron_LDOS(self, mol, spin,
                                              lc=lc, wf_k=wf_k, P_aui=P_aui)
        return fold(energies * Hartree, weights, npts, width)

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0, broadcast=True,
                                 pad=True):
        """Return pseudo-wave-function array.

        Unit: 1/Angstrom^(3/2)
        """
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
            return result * Hartree

    def get_occupation_numbers(self, kpt=0, spin=0):
        """Return occupation array."""
        return self.collect_occupations(kpt, spin)

    get_occupations = get_occupation_numbers
    
    def initial_wannier(self, initialwannier, kpointgrid, fixedstates,
                        edf, spin):
        """Initial guess for the shape of wannier functions.

        Use initial guess for wannier orbitals to determine rotation
        matrices U and C.
        """
        if self.nkpts != 1:
            raise NotImplementedError
        from ase.dft.wannier import rotation_from_projection
        proj_knw = self.get_projections(initialwannier, spin)
        U_ww, C_ul = rotation_from_projection(proj_knw[0],
                                              fixedstates[0],
                                              ortho=True)
        return [C_ul], U_ww[npy.newaxis]

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        c = dirG.tolist().index(1)
        G = self.bzk_kc[nextkpoint, c] - self.bzk_kc[kpoint, c] - G_I[c]

        return self.get_wannier_integrals(c, spin, kpoint,
                                          nextkpoint, G, nbands)

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
        return self.kT * Hartree
