# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

import sys
import time
from math import pi, sqrt, log

import Numeric as num

from gpaw.localized_functions import LocFuncBroadcaster
from gpaw.operators import Laplace
from gpaw.pair_potential import PairPotential
from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.xc_functional import XC3DGrid
from gpaw.mpi import run


class Hamiltonian:
    """Hamiltonian object.

    Attributes:
     =============== =====================================================
     ``xc``          ``XC3DGrid`` object.
     ``nuclei``      List of ``Nucleus`` objects.
     ``pairpot``     ``PairPotential`` object.
     ``poisson``     ``PoissonSolver``.
     ``gd``          Grid descriptor for coarse grids.
     ``finegd``      Grid descriptor for fine grids.
     ``restrict``    Function for restricting the effective potential.
     =============== =====================================================

    Soft and smooth pseudo functions on uniform 3D grids:
     ========== =========================================
     ``vHt_g``  Hartree potential on the fine grid.
     ``vt_sG``  Effective potential on the coarse grid.
     ``vt_sg``  Effective potential on the fine grid.
     ========== =========================================
    """

    def __init__(self, paw):
        """Create the Hamiltonian."""

        self.nspins = paw.nspins
        self.gd = paw.gd
        self.finegd = paw.finegd
        self.my_nuclei = paw.my_nuclei
        self.pt_nuclei = paw.pt_nuclei
        self.ghat_nuclei = paw.ghat_nuclei
        self.nuclei = paw.nuclei
        self.timer = paw.timer

        # Allocate arrays for potentials and densities on coarse and
        # fine grids:
        self.vt_sG = self.gd.empty(self.nspins)
        self.vHt_g = self.finegd.zeros()
        self.vt_sg = self.finegd.empty(self.nspins)

        # The external potential
        vext_g = paw.input_parameters['external']
        if vext_g is not None:
            assert num.alltrue(vext_g.shape ==
                               self.finegd.get_size_of_global_array())
            self.vext_g = self.finegd.zeros()
            self.finegd.distribute(vext_g, self.vext_g)
        else:
            self.vext_g = None

        p = paw.input_parameters
        stencils = p['stencils']

        # Number of neighbor grid points used for finite difference
        # Laplacian in the Schrödinger equation (1, 2, ...):
        nn = stencils[0]

        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, nn, paw.typecode)

        # Number of neighbor grid points used for interpolation (1, 2,
        # or 3):
        nn = stencils[2]

        # Restrictor function for the potential:
        self.restrict = Transformer(self.finegd, self.gd, nn).apply

        # Number of neighbor grid points used for finite difference
        # Laplacian in the Poisson equation (1, 2, ...):
        self.poisson_stencil = nn = stencils[1]

        # Solver for the Poisson equation:
        self.poisson = PoissonSolver(self.finegd, nn, p['poissonsolver'])

        # Pair potential for electrostatic interacitons:
        self.pairpot = PairPotential(paw.setups)

        self.npoisson = 0 #???

        # Exchange-correlation functional object:
        self.xc = XC3DGrid(paw.xcfunc, self.finegd, self.nspins)

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""

        self.timer.start('Hamiltonian')
        vt_g = self.vt_sg[0]
        vt_g[:] = 0.0

        density.update_pseudo_charge()

        for nucleus in self.ghat_nuclei:
            nucleus.add_localized_potential(vt_g)

        Ebar = num.vdot(vt_g, density.nt_g) * self.finegd.dv

        for nucleus in self.ghat_nuclei:
            nucleus.add_hat_potential(vt_g)

        Epot = num.vdot(vt_g, density.nt_g) * self.finegd.dv - Ebar

        Eext = 0.0
        if self.vext_g is not None:
            vt_g += self.vext_g
            Eext = num.vdot(vt_g, density.nt_g) * self.finegd.dv - Ebar - Epot

        if self.nspins == 2:
            self.vt_sg[1] = vt_g

        if self.nspins == 2:
            Exc = self.xc.get_energy_and_potential(
                density.nt_sg[0], self.vt_sg[0],
                density.nt_sg[1], self.vt_sg[1])
        else:
            Exc = self.xc.get_energy_and_potential(
                density.nt_sg[0], self.vt_sg[0])

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_g, density.rhot_g,
                                           charge=-density.charge)
        self.timer.stop('Poisson')

        Epot += 0.5 * num.vdot(self.vHt_g, density.rhot_g) * self.finegd.dv
        Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, density.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            Ekin -= num.vdot(vt_G, nt_G - density.nct_G) * self.gd.dv

        # Calculate atomic hamiltonians:
        self.timer.start('Atomic Hamiltonians')
        iters = []
        for nucleus in self.ghat_nuclei:
            # Energy corections due to external potential.
            # Potential is assumed to be constant inside augmentation spheres.
            if self.vext_g is not None and nucleus.in_this_domain:
                g_c = nucleus.get_nearest_grid_point(density.finegd)
                g_c -= (g_c == density.finegd.n_c) # force point to this domain
                vext = self.vext_g[R_c]
            else:
                vext = None

            iters.append(nucleus.calculate_hamiltonian(density.nt_g,
                                                       self.vHt_g, vext))
        if len(iters) != 0:
            k, p, b, v, x = num.sum(run(iters))
            Ekin += k
            Epot += p
            Ebar += b
            Eext += v
            Exc += x

        self.timer.stop('Atomic Hamiltonians')

        comm = self.gd.comm
        self.Ekin = comm.sum(Ekin)
        self.Epot = comm.sum(Epot)
        self.Ebar = comm.sum(Ebar)
        self.Eext = comm.sum(Eext)
        self.Exc = comm.sum(Exc)

        self.timer.stop('Hamiltonian')
