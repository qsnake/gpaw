# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a density class."""

from math import pi, sqrt

import numpy as np

from gpaw import debug, extra_parameters
from gpaw.mixer import BaseMixer, Mixer, MixerSum
from gpaw.transformers import Transformer
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.wavefunctions import LCAOWaveFunctions


class Density:
    """Density object.
    
    Attributes:
     =============== =====================================================
     ``gd``          Grid descriptor for coarse grids.
     ``finegd``      Grid descriptor for fine grids.
     ``interpolate`` Function for interpolating the electron density.
     ``mixer``       ``DensityMixer`` object.
     =============== =====================================================

    Soft and smooth pseudo functions on uniform 3D grids:
     ========== =========================================
     ``nt_sG``  Electron density on the coarse grid.
     ``nt_sg``  Electron density on the fine grid.
     ``nt_g``   Electron density on the fine grid.
     ``rhot_g`` Charge density on the fine grid.
     ``nct_G``  Core electron-density on the coarse grid.
     ========== =========================================
    """
    
    def __init__(self, gd, finegd, nspins, charge):
        """Create the Density object."""

        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.charge = float(charge)

        self.charge_eps = 1e-7
        self.D_asp = None
        self.Q_aL = None

        self.nct_G = None
        self.nt_sG = None
        self.rhot_g = None
        self.nt_sg = None
        self.nt_g = None

        self.rank_a = None

        self.mixer = BaseMixer()
        
    def initialize(self, setups, stencil, timer, magmom_a, hund):
        self.timer = timer
        self.setups = setups
        self.hund = hund
        self.magmom_a = magmom_a
        
        # Interpolation function for the density:
        self.interpolator = Transformer(self.gd, self.finegd, stencil)

        self.nct = LFC(self.gd, [[setup.nct] for setup in setups],
                       integral=[setup.Nct for setup in setups],
                       forces=True, cut=True)
        self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups],
                        integral=sqrt(4 * pi), forces=True)

    def set_positions(self, spos_ac, rank_a=None):
        self.nct.set_positions(spos_ac)
        self.ghat.set_positions(spos_ac)
        self.mixer.reset()

        self.nct_G = self.gd.zeros()
        self.nct.add(self.nct_G, 1.0 / self.nspins)
        #self.nt_sG = None
        self.nt_sg = None
        self.nt_g = None
        self.rhot_g = None
        self.Q_aL = None

        if self.D_asp is not None:
            requests = []
            D_asp = {}
            for a in self.nct.my_atom_indices:
                if a in self.D_asp:
                    D_asp[a] = self.D_asp.pop(a)
                else:
                    # Get matrix from old domain:
                    ni = self.setups[a].ni
                    D_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                    D_asp[a] = D_sp
                    requests.append(self.gd.comm.receive(D_sp, self.rank_a[a],
                                                         39, False))
            for a, D_sp in self.D_asp.items():
                # Send matrix to new domain:
                requests.append(self.gd.comm.send(D_sp, rank_a[a], 39, False))
            for request in requests:
                self.gd.comm.wait(request)
            self.D_asp = D_asp

        self.rank_a = rank_a

    def update(self, wfs):
        wfs.calculate_density(self)
        wfs.calculate_atomic_density_matrices(self)
        self.nt_sG += self.nct_G
        comp_charge = self.calculate_multipole_moments()
        
        if (isinstance(wfs, LCAOWaveFunctions) or
            extra_parameters.get('normalize')):
            self.normalize(comp_charge)

        self.mix(comp_charge)

    def normalize(self, comp_charge=None):
        """Normalize pseudo density."""
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()
        
        pseudo_charge = self.gd.integrate(self.nt_sG).sum()
        if pseudo_charge != 0:
            x = -(self.charge + comp_charge) / pseudo_charge
            self.nt_sG *= x

    def calculate_pseudo_charge(self, comp_charge):
        self.nt_g = self.nt_sg.sum(axis=0)
        self.rhot_g = self.nt_g.copy()
        self.ghat.add(self.rhot_g, self.Q_aL)

        if extra_parameters.get('normalize'):
            if comp_charge is None:
                comp_charge = self.calculate_multipole_moments()
            Rt = self.finegd.integrate(self.rhot_g)
            x = comp_charge / (comp_charge + self.charge + Rt)
            self.rhot_g *= x
            self.rhot_g += (1.0 - x) * self.nt_g

        if debug:
            charge = self.finegd.integrate(self.rhot_g) + self.charge
            if abs(charge) > self.charge_eps:
                raise RuntimeError('Charge not conserved: excess=%.9f' %
                                   charge)

    def mix(self, comp_charge):
        if not self.mixer.mix_rho:
            self.mixer.mix(self)
            comp_charge = None
          
        self.interpolate(comp_charge)
        self.calculate_pseudo_charge(comp_charge)

        if self.mixer.mix_rho:
            self.mixer.mix(self)

    def interpolate(self, comp_charge=None):
        """Interpolate pseudo density to fine grid."""
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()

        if self.nt_sg is None:
            self.nt_sg = self.finegd.empty(self.nspins)

        for s in range(self.nspins):
            self.interpolator.apply(self.nt_sG[s], self.nt_sg[s])

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if not self.gd.pbc_c.all():
            # With zero-boundary conditions in one or more directions,
            # this is not the case.
            pseudo_charge = -(self.charge + comp_charge)
            if abs(pseudo_charge) > 1.0e-14:
                x = pseudo_charge / self.finegd.integrate(self.nt_sg).sum()
                self.nt_sg *= x

    def calculate_multipole_moments(self):
        """Calculate multipole moments of compensation charges.

        Returns the total compenstion charge in units of electron
        charge, so the number will be negative because of the
        dominating contribution from the nuclear charge."""

        comp_charge = 0.0
        self.Q_aL = {}
        for a, D_sp in self.D_asp.items():
            Q_L = self.Q_aL[a] = np.dot(D_sp.sum(0), self.setups[a].Delta_pL)
            Q_L[0] += self.setups[a].Delta0
            comp_charge += Q_L[0]
        return self.gd.comm.sum(comp_charge) * sqrt(4 * pi)

    def initialize_from_atomic_densities(self, basis_functions):
        """Initialize density from atomic densities.

        The density is initialized from atomic orbitals, and will
        be constructed with the specified magnetic moments and
        obeying Hund's rules if ``hund`` is true."""

        f_sM = np.empty((self.nspins, basis_functions.Mmax))
        self.D_asp = {}
        f_asi = {}
        c = self.charge / len(self.setups)  # distribute charge on all atoms
        for a in basis_functions.atom_indices:
            f_si = self.setups[a].calculate_initial_occupation_numbers(
                    self.magmom_a[a], self.hund, charge=c)
            if a in basis_functions.my_atom_indices:
                self.D_asp[a] = self.setups[a].initialize_density_matrix(f_si)
            f_asi[a] = f_si

        self.nt_sG = self.gd.zeros(self.nspins)
        basis_functions.add_to_density(self.nt_sG, f_asi)
        self.nt_sG += self.nct_G

    def set_mixer(self, mixer, fixmom, width):
        if mixer is not None:
            if (self.nspins == 2 and (not fixmom or width != 0)
                and isinstance(mixer, Mixer)):
                raise RuntimeError('Cannot use Mixer in spin-polarized '
                                   'calculations without fixed moment '
                                   'nor with finite Fermi-width.')
            self.mixer = mixer
        else:
            if self.nspins == 2 and (not fixmom or width != 0):
                self.mixer = MixerSum()
            else:
                self.mixer = Mixer()

        if self.nspins == 1 and isinstance(mixer, MixerSum):
            raise RuntimeError('Cannot use MixerSum with nspins==1')

        self.mixer.initialize(self)
        
    def estimate_magnetic_moments(self):
        magmom_a = np.zeros_like(self.magmom_a)
        if self.nspins == 2:
            for a, D_sp in self.D_asp.items():
                magmom_a[a] = np.dot(D_sp[0] - D_sp[1], self.setups[a].N0_p)
            self.gd.comm.sum(magmom_a)
        return magmom_a

    def get_correction(self, a, spin):
        """Integrated atomic density correction.

        Get the integrated correction to the pseuso density relative to
        the all-electron density.
        """
        setup = self.setups[a]
        return sqrt(4 * pi) * (
            np.dot(self.D_asp[a][spin], setup.Delta_pL[:, 0])
            + setup.Delta0 / self.nspins)

    def get_density_array(self):
        XXX
        # XXX why not replace with get_spin_density and get_total_density?
        """Return pseudo-density array."""
        if self.nspins == 2:
            return self.nt_sG
        else:
            return self.nt_sG[0]
    
    def get_all_electron_density(self, atoms, gridrefinement=2):
        """Return real all-electron density array."""

        # Refinement of coarse grid, for representation of the AE-density
        if gridrefinement == 1:
            gd = self.gd
            n_sg = self.nt_sG.copy()
        elif gridrefinement == 2:
            gd = self.finegd
            if self.nt_sg is None:
                self.interpolate()
            n_sg = self.nt_sg.copy()
        elif gridrefinement == 4:
            # Extra fine grid
            gd = self.finegd.refine()
            
            # Interpolation function for the density:
            interpolator = Transformer(self.finegd, gd, 3)

            # Transfer the pseudo-density to the fine grid:
            n_sg = gd.empty(self.nspins)
            if self.nt_sg is None:
                self.interpolate()
            for s in range(self.nspins):
                interpolator.apply(self.nt_sg[s], n_sg[s])
        else:
            raise NotImplementedError

        # Add corrections to pseudo-density to get the AE-density
        splines = {}
        phi_aj = []
        phit_aj = []
        nc_a = []
        nct_a = []
        for a, id in enumerate(self.setups.id_a):
            if id in splines:
                phi_j, phit_j, nc, nct = splines[id]
            else:
                # Load splines:
                phi_j, phit_j, nc, nct = self.setups[a].get_partial_waves()[:4]
                splines[id] = (phi_j, phit_j, nc, nct)
            phi_aj.append(phi_j)
            phit_aj.append(phit_j)
            nc_a.append([nc])
            nct_a.append([nct])

        # Create localized functions from splines
        phi = LFC(gd, phi_aj)
        phit = LFC(gd, phit_aj)
        nc = LFC(gd, nc_a)
        nct = LFC(gd, nct_a)
        spos_ac = atoms.get_scaled_positions() % 1.0
        phi.set_positions(spos_ac)
        phit.set_positions(spos_ac)
        nc.set_positions(spos_ac)
        nct.set_positions(spos_ac)

        all_D_asp = []
        for a, setup in enumerate(self.setups):
            D_sp = self.D_asp.get(a)
            if D_sp is None:
                ni = setup.ni
                D_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
            if gd.comm.size > 1:
                gd.comm.broadcast(D_sp, self.rank_a[a])
            all_D_asp.append(D_sp)

        for s in range(self.nspins):
            I_a = np.zeros(len(atoms))
            nc.add1(n_sg[s], 1.0 / self.nspins, I_a)
            nct.add1(n_sg[s], -1.0 / self.nspins, I_a)
            phi.add2(n_sg[s], all_D_asp, s, 1.0, I_a)
            phit.add2(n_sg[s], all_D_asp, s, -1.0, I_a)
            for a, D_sp in self.D_asp.items():
                setup = self.setups[a]
                I_a[a] -= ((setup.Nc - setup.Nct) / self.nspins +
                           sqrt(4 * pi) *
                           np.dot(D_sp[s], setup.Delta_pL[:, 0]))
            gd.comm.sum(I_a)
            N_c = gd.N_c
            g_ac = np.around(N_c * spos_ac).astype(int) % N_c - gd.beg_c
            for I, g_c in zip(I_a, g_ac):
                if (g_c >= 0).all() and (g_c < gd.n_c).all():
                    n_sg[s][tuple(g_c)] -= I / gd.dv

        return n_sg, gd

    def initialize_kinetic(self, atoms):
        """Initial pseudo electron kinetic density."""

        self.tauct = LFC(self.gd, [[setup.tauct] for setup in self.setups],
                        forces=True, cut=True)

        spos_ac = atoms.get_scaled_positions() % 1.0
        self.tauct.set_positions(spos_ac)

        self.taut_sG = self.gd.empty(self.nspins)
        self.taut_sg = None

    def interpolate_kinetic(self, comp_charge=None):
        """Interpolate pseudo electron kinetic density to the fine grid."""

        if self.taut_sg is None:
            self.taut_sg = self.finegd.empty(self.nspins)

        for s in range(self.nspins):
            self.interpolator.apply(self.taut_sG[s], self.taut_sg[s])

        """
        if comp_charge is None:
            comp_charge = self.calculate_multipole_moments()

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if not self.gd.pbc_c.all():
            # With zero-boundary conditions in one or more directions,
            # this is not the case.
            pseudo_charge = -(self.charge + comp_charge)
            if abs(pseudo_charge) > 1.0e-14:
                work_sg = self.finegd.empty(self.nspins)
                for s in range(self.nspins):
                    self.interpolator.apply(self.nt_sG[s], work_sg[s])

                x = pseudo_charge / self.finegd.integrate(work_sg).sum()
                self.taut_sg *= x #TODO XXX scaling by same factor as nt_sg
        """

    def update_kinetic(self, wfs, symmetry=None):
        """Calculate pseudo electron kinetic density.
        The pseudo electron-density ``taut_sG`` is calculated from the
        wave functions, the occupation numbers and atomic contributions."""

        self.taut_sG[:] = 0.

        # Add contribution from all k-points
        for kpt in wfs.kpt_u:
            wfs.add_to_kinetic_density_from_k_point(self.taut_sG[kpt.s], kpt)
        wfs.band_comm.sum(self.taut_sG)
        wfs.kpt_comm.sum(self.taut_sG)

        # Add the pseudo core kinetic array
        for s in range(wfs.nspins):
            self.tauct.add(self.taut_sG[s], 1.0 / wfs.nspins)

        # For periodic boundary conditions
        if symmetry is not None:
            for taut_G in self.taut_sG:
                symmetry.symmetrize(taut_G, self.gd)

        return

    def estimate_memory(self, mem):
        nspins = self.nspins
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()

        arrays = mem.subnode('Arrays')
        for name, size in [('nt_sG', nbytes * nspins),
                           ('nt_sg', nfinebytes * nspins),
                           ('nt_g', nfinebytes),
                           ('rhot_g', nfinebytes),
                           ('nct_G', nbytes)]:
            arrays.subnode(name, size)

        lfs = mem.subnode('Localized functions')
        for name, obj in [('nct', self.nct),
                          ('ghat', self.ghat)]:
            obj.estimate_memory(lfs.subnode(name))
        self.mixer.estimate_memory(mem.subnode('Mixer'), self.gd)
        self.interpolator.estimate_memory(mem.subnode('Interpolator'))

