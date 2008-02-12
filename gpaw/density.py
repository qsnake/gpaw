# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a density class."""

import sys
from math import pi, sqrt, log
import time

from numpy import array, dot, newaxis, zeros, transpose
from numpy.linalg import solve

from gpaw.mixer import Mixer, MixerSum
from gpaw.transformers import Transformer
from gpaw.utilities import pack, unpack2
from gpaw.utilities.complex import cc, real


class Density:
    """Density object.
    
    Attributes:
     =============== =====================================================
     ``nuclei``      List of ``Nucleus`` objects.
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
    
    def __init__(self, paw, magmom_a):
        """Create the Density object."""

        p = paw.input_parameters
        self.hund = p['hund']
        self.idiotproof = p['idiotproof']
        
        self.magmom_a = magmom_a
        self.nspins = paw.nspins
        self.gd = paw.gd
        self.finegd = paw.finegd
        self.my_nuclei = paw.my_nuclei
        self.ghat_nuclei = paw.ghat_nuclei
        self.nuclei = paw.nuclei
        self.timer = paw.timer
        self.kpt_comm = paw.kpt_comm
        self.band_comm = paw.band_comm
        self.nvalence = paw.nvalence
        self.charge = float(p['charge'])
        self.charge_eps = 1e-7
        self.lcao = paw.eigensolver.lcao
        
        self.nvalence0 = self.nvalence + self.charge
        for nucleus in self.nuclei:
            setup = nucleus.setup
            self.charge += (setup.Z - setup.Nv - setup.Nc)
        
        # Allocate arrays for potentials and densities on coarse and
        # fine grids:
        self.nct_G = self.gd.empty()
        self.nt_sG = self.gd.empty(self.nspins)
        self.rhot_g = self.finegd.empty()
        self.nt_sg = self.finegd.empty(self.nspins)
        if self.nspins == 1:
            self.nt_g = self.nt_sg[0]
        else:
            self.nt_g = self.finegd.empty()

        # Number of neighbor grid points used for interpolation (1, 2, or 3):
        nn = p['stencils'][1]

        # Interpolation function for the density:
        self.interpolate = Transformer(self.gd, self.finegd, nn).apply
        
        # Density mixer
        self.set_mixer(paw, p['mixer'])
        
        self.initialized = False

    def initialize(self):
        """Initialize density.

        The density is initialized from atomic orbitals, and will
        be constructed with the specified magnetic moments and
        obeying Hund's rules if ``hund`` is true."""

        self.nt_sG[:] = self.nct_G
        for magmom, nucleus in zip(self.magmom_a, self.nuclei):
            nucleus.add_atomic_density(self.nt_sG, magmom, self.hund)

        # The nucleus.add_atomic_density() method should be improved
        # so that we don't have to do this scaling: XXX
        if self.nvalence != self.nvalence0:
            x = float(self.nvalence) / self.nvalence0
            for nucleus in self.my_nuclei:
                nucleus.D_sp *= x
            self.nt_sG *= x
                
        # We don't have any occupation numbers.  The initial
        # electron density comes from overlapping atomic densities
        # or from a restart file.  We scale the density to match
        # the compensation charges:

        for nucleus in self.nuclei:
            nucleus.calculate_multipole_moments()

        comm = self.gd.comm
        
        if self.nspins == 1:
            Q = 0.0
            Q0 = 0.0
            for nucleus in self.my_nuclei:
                Q += nucleus.Q_L[0]
                Q0 += nucleus.setup.Delta0
            Q = sqrt(4 * pi) * comm.sum(Q)
            Q0 = sqrt(4 * pi) * comm.sum(Q0)
            Nt = self.gd.integrate(self.nt_sG[0])
            # Nt + Q must be equal to minus the total charge:
            if Q0 - Q != 0:
                x = (Nt + Q0 + self.charge) / (Q0 - Q)
                for nucleus in self.my_nuclei:
                    nucleus.D_sp *= x

                for nucleus in self.nuclei:
                    nucleus.calculate_multipole_moments()
            else:
                x = -(self.charge + Q) / Nt
                self.nt_sG *= x

        else:
            Q_s = array([0.0, 0.0])
            for nucleus in self.my_nuclei:
                s = nucleus.setup
                Q_s += 0.5 * s.Delta0 + dot(nucleus.D_sp, s.Delta_pL[:, 0])
            Q_s *= sqrt(4 * pi)
            comm.sum(Q_s)
            Nt_s = self.gd.integrate(self.nt_sG)

            M = sum(self.magmom_a)
            x = 1.0
            y = 1.0
            if Nt_s[0] == 0:
                if Nt_s[1] != 0:
                    y = -(self.charge + Q_s[0] + Q_s[1]) / Nt_s[1]
            else:
                if Nt_s[1] == 0:
                    x = -(self.charge + Q_s[0] + Q_s[1]) / Nt_s[0]
                else:
                    x, y = solve(array([[Nt_s[0],  Nt_s[1]],
                                        [Nt_s[0], -Nt_s[1]]]),
                                 array([-Q_s[0] - Q_s[1] - self.charge,
                                        -Q_s[0] + Q_s[1] + M]))

            if self.charge == 0:
                if abs(x - 1.0) > 0.17 or abs(y - 1.0) > 0.17:
                    warning = ('Bad initial density.  Scaling factors: %f, %f'
                               % (x, y))
                    if self.idiotproof:
                        raise RuntimeError(warning)
                    else:
                        print(warning)

            self.nt_sG[0] *= x
            self.nt_sG[1] *= y

        self.mixer.mix(self.nt_sG)

        self.interpolate_pseudo_density()

        self.initialized = True

    def interpolate_pseudo_density(self):
        """Transfer the density from the coarse to the fine grid."""
        for s in range(self.nspins):
            self.interpolate(self.nt_sG[s], self.nt_sg[s])

        # With periodic boundary conditions, the interpolation will
        # conserve the number of electrons.
        if False in self.gd.domain.pbc_c:
            # With zero-boundary conditions in one or more directions,
            # this is not the case.
            for s in range(self.nspins):
                Nt0 = self.gd.integrate(self.nt_sG[s])
                Nt = self.finegd.integrate(self.nt_sg[s])
                if Nt != 0.0:
                    self.nt_sg[s] *= Nt0 / Nt

    def set_mixer(self, paw, mixer):
        if mixer is not None:
            self.mixer = mixer
        else:
            if self.nspins == 2 and (not paw.fixmom or paw.kT != 0):
                self.mixer = MixerSum()#mix, self.gd)
            else:
                self.mixer = Mixer()#mix, self.gd, self.nspins)

        self.mixer.initialize(self.gd, self.nspins)
        
    def update_pseudo_charge(self):
        if self.nspins == 2:
            self.nt_g[:] = self.nt_sg[0]
            self.nt_g += self.nt_sg[1]

        Q = 0.0
        for nucleus in self.nuclei:
            nucleus.calculate_multipole_moments()
            Q += nucleus.Q_L[0] * sqrt(4 * pi)

        if self.lcao:
            Nt = self.finegd.integrate(self.nt_g)
            scale = -Q / Nt
            assert abs(scale - 1.0) < 0.01
            self.nt_g *= scale
            
        self.rhot_g[:] = self.nt_g

        for nucleus in self.ghat_nuclei:
            nucleus.add_compensation_charge(self.rhot_g)
            
        charge = self.finegd.integrate(self.rhot_g) + self.charge
        if abs(charge) > self.charge_eps:
            raise RuntimeError('Charge not conserved: excess=%.7f' % charge ) 

    def update(self, kpt_u, symmetry):
        """Calculate pseudo electron-density.

        The pseudo electron-density ``nt_sG`` is calculated from the
        wave functions, the occupation numbers, and the smooth core
        density ``nct_G``, and finally symmetrized and mixed."""

        self.nt_sG[:] = 0.0

        # Add contribution from all k-points:
        for kpt in kpt_u:
            kpt.add_to_density(self.nt_sG[kpt.s])

        self.band_comm.sum(self.nt_sG)
        self.kpt_comm.sum(self.nt_sG)

        # add the smooth core density:
        self.nt_sG += self.nct_G

        # Compute atomic density matrices:
        for nucleus in self.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            D_sii = zeros((self.nspins, ni, ni))
            for kpt in kpt_u:
                P_ni = nucleus.P_uni[kpt.u]
                D_sii[kpt.s] += real(dot(cc(transpose(P_ni)),
                                             P_ni * kpt.f_n[:, newaxis]))
            nucleus.D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.band_comm.sum(nucleus.D_sp)
            self.kpt_comm.sum(nucleus.D_sp)

        comm = self.gd.comm
        
        if symmetry is not None:
            for nt_G in self.nt_sG:
                symmetry.symmetrize(nt_G, self.gd)

            D_asp = []
            for nucleus in self.nuclei:
                if comm.rank == nucleus.rank:
                    D_sp = nucleus.D_sp
                    comm.broadcast(D_sp, nucleus.rank)
                else:
                    ni = nucleus.get_number_of_partial_waves()
                    np = ni * (ni + 1) / 2
                    D_sp = zeros((self.nspins, np))
                    comm.broadcast(D_sp, nucleus.rank)
                D_asp.append(D_sp)

            for s in range(self.nspins):
                D_aii = [unpack2(D_sp[s]) for D_sp in D_asp]
                for nucleus in self.my_nuclei:
                    nucleus.symmetrize(D_aii, symmetry.maps, s)

        self.mixer.mix(self.nt_sG)

        self.interpolate_pseudo_density()

    def move(self):
        self.mixer.reset(self.my_nuclei)

        # Set up smooth core density:
        self.nct_G[:] = 0.0
        for nucleus in self.nuclei:
            nucleus.add_smooth_core_density(self.nct_G, self.nspins)

    def calculate_local_magnetic_moments(self):
        # XXX remove this?
        spindensity = self.nt_sg[0] - self.nt_sg[1]

        for nucleus in self.nuclei:
            nucleus.calculate_magnetic_moments()
            
        #locmom = 0.0
        for nucleus in self.nuclei:
            #locmom += nucleus.mom[0]
            mom = array([0.0])
            if nucleus.stepf is not None:
                nucleus.stepf.integrate(spindensity, mom)
                nucleus.mom = array(nucleus.mom + mom[0])
            nucleus.comm.broadcast(nucleus.mom, nucleus.rank)

    def get_density_array(self):
        """Return pseudo-density array."""
        if self.nspins == 2:
            return self.nt_sG
        else:
            return self.nt_sG[0]
    
    def get_all_electron_density(self, gridrefinement=2):
        """Return real all-electron density array."""

        # Refinement of coarse grid, for representation of the AE-density
        if gridrefinement == 1:
            gd = self.gd
            n_sg = self.nt_sG.copy()
        elif gridrefinement == 2:
            gd = self.finegd
            n_sg = self.nt_sg.copy()
        elif gridrefinement == 4:
            # Extra fine grid
            gd = self.finegd.refine()
            
            # Interpolation function for the density:
            interpolator = Transformer(self.finegd, gd, 3)

            # Transfer the pseudo-density to the fine grid:
            n_sg = gd.empty(self.nspins)
            for s in range(self.nspins):
                interpolator.apply(self.nt_sg[s], n_sg[s])
        else:
            raise NotImplementedError

        # Add corrections to pseudo-density to get the AE-density
        splines = {}
        for nucleus in self.nuclei:
            nucleus.add_density_correction(n_sg, self.nspins, gd, splines)
        
        # Return AE-(spin)-density
        if self.nspins == 2:
            return n_sg
        else:
            return n_sg[0]

    def initialize_kinetic(self):
        """Initial pseudo electron kinetic density."""
        """flag to use local variable in tpss.c"""

        self.taut_sG = self.gd.empty(self.nspins)
        self.taut_sg = self.finegd.empty(self.nspins)

    def update_kinetic(self, kpt_u):
        """Calculate pseudo electron kinetic density.
        The pseudo electron-density ``taut_sG`` is calculated from the
        wave functions, the occupation numbers,
        the peusdo core density ``tauct_G``, is not included"""

        ## Add contribution from all k-points:
        for kpt in kpt_u:
            kpt.add_to_kinetic_density(self.taut_sG[kpt.s])
        self.band_comm.sum(self.taut_sG)
        self.kpt_comm.sum(self.taut_sG)

        """Transfer the density from the coarse to the fine grid."""
        for s in range(self.nspins):
            self.interpolate(self.taut_sG[s], self.taut_sg[s])

        return 
