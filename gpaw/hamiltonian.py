# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

import sys
import time
from math import pi, sqrt, log

import Numeric as num

from gpaw.exx import get_exx
from gpaw.localized_functions import LocFuncBroadcaster
from gpaw.operators import Laplace
from gpaw.pair_potential import PairPotential
from gpaw.poisson_solver import PoissonSolver
from gpaw.transformers import Restrictor
from gpaw.xc_functional import XC3DGrid


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
    
    def __init__(self, gd, finegd, xcfunc, nspins, typecode, stencils, relax,
                 timer, my_nuclei, pt_nuclei, ghat_nuclei, nuclei,
                 setups, exx):
        """Create the Hamiltonian."""

        self.nspins = nspins
        self.gd = gd
        self.finegd = finegd
        self.my_nuclei = my_nuclei
        self.pt_nuclei = pt_nuclei
        self.ghat_nuclei = ghat_nuclei
        self.nuclei = nuclei
        self.timer = timer
        self.exx = exx

        # Allocate arrays for potentials and densities on coarse and
        # fine grids:
        self.vt_sG = gd.new_array(nspins)
        self.vHt_g = finegd.new_array()        
        self.vt_sg = finegd.new_array(nspins)

        # Number of neighbor grid points used for finite difference
        # Laplacian in the Schrödinger equation (1, 2, ...):
        nn = stencils[0]

        # Kinetic energy operator:
        self.kin = Laplace(gd, -0.5, nn, typecode)

        # exchange-correlation functional object:
        self.xc = XC3DGrid(xcfunc, finegd, nspins)

        # Number of neighbor grid points used for interpolation (1, 2,
        # or 3):
        nn = stencils[2]

        # Restrictor function for the potential:
        self.restrict = Restrictor(finegd, nn, num.Float).apply

        # Number of neighbor grid points used for finite difference
        # Laplacian in the Poisson equation (1, 2, ...):
        self.poisson_stencil = nn = stencils[1]

        # Solver for the Poisson equation:
        self.poisson = PoissonSolver(finegd, nn, relax)
   
        # Pair potential for electrostatic interacitons:
        self.pairpot = PairPotential(setups)

        self.npoisson = 0

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potentials are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""

        vt_g = self.vt_sg[0]
        vt_g[:] = 0.0

        density.update_pseudo_charge()
        
        for nucleus in self.ghat_nuclei:
            nucleus.add_localized_potential(vt_g)

        Ebar = num.vdot(vt_g, density.nt_g) * self.finegd.dv 

        for nucleus in self.ghat_nuclei:
            nucleus.add_hat_potential(vt_g)

        Epot = num.vdot(vt_g, density.nt_g) * self.finegd.dv - Ebar
        
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
        self.timer.stop()
        
        Epot += 0.5 * num.vdot(self.vHt_g, density.rhot_g) * self.finegd.dv
        Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, density.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            Ekin -= (1 - .5 * self.xc.xcfunc.hybrid) * num.vdot( # EXX hack
                vt_G, nt_G - density.nct_G) * self.gd.dv

        # Exact-exchange correction
        if self.exx is not None:
            Exx = self.exx.Exx
            Exc += Exx
            Ekin -= Exx
            
        # Calculate atomic hamiltonians:
        for nucleus in self.ghat_nuclei:
            k, p, b, x = nucleus.calculate_hamiltonian(density.nt_g,
                                                       self.vHt_g)
            Ekin += k
            Epot += p
            Ebar += b
            Exc += x

        comm = self.gd.comm
        Ekin = comm.sum(Ekin)
        Epot = comm.sum(Epot)
        Ebar = comm.sum(Ebar)
        Exc = comm.sum(Exc)
        
        return Ekin, Epot, Ebar, Exc

        
