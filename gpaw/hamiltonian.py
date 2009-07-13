# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

from math import pi, sqrt

import numpy as np

from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.xc_functional import XCFunctional, xcgrid
from gpaw.lfc import LFC
from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full


class Hamiltonian:
    """Hamiltonian object.

    Attributes:
     =============== =====================================================
     ``xc``          ``XC3DGrid`` object.
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

    Energy contributions and forces:

    =========== ==========================================
                Description
    =========== ==========================================
    ``Ekin``    Kinetic energy.
    ``Epot``    Potential energy.
    ``Etot``    Total energy.
    ``Exc``     Exchange-Correlation energy.
    ``Eext``    Energy of external potential
    ``Eref``    Reference energy for all-electron atoms.
    ``S``       Entropy.
    ``Ebar``    Should be close to zero!
    =========== ==========================================

    """

    def __init__(self, gd, finegd, nspins, setups, stencil, timer, xcfunc,
                 psolver, vext_g):
        """Create the Hamiltonian."""
        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.setups = setups
        self.timer = timer
        self.xcfunc = xcfunc
        
        # Solver for the Poisson equation:
        if psolver is None:
            if gd.is_non_orthogonal():
                psolver = PoissonSolver(nn=2, relax='J')
            else:
                psolver = PoissonSolver(nn='M', relax='J')
        self.poisson = psolver
        self.poisson.set_grid_descriptor(finegd)

        self.dH_asp = None

        # The external potential
        self.vext_g = vext_g

        self.vt_sG = None
        self.vHt_g = None
        self.vt_sg = None
        self.vbar_g = None

        self.rank_a = None

        # Restrictor function for the potential:
        self.restrictor = Transformer(self.finegd, self.gd, stencil,
                                      allocate=False)
        self.restrict = self.restrictor.apply

        # Exchange-correlation functional object:
        self.xc = xcgrid(xcfunc, finegd, nspins)

        self.vbar = LFC(self.finegd, [[setup.vbar] for setup in setups],
                        forces=True)

        self.Ekin0 = None
        self.Ekin = None
        self.Epot = None
        self.Ebar = None
        self.Eext = None
        self.Exc = None
        self.Etot = None
        self.S = None
        self.allocated = False

    def allocate(self):
        # TODO We should move most of the gd.empty() calls here
        assert not self.allocated
        self.restrictor.allocate()
        self.xc.allocate()
        self.allocated = True

    def set_positions(self, spos_ac, rank_a=None):
        if not self.allocated:
            self.allocate()
        self.vbar.set_positions(spos_ac)
        if self.vbar_g is None:
            self.vbar_g = self.finegd.empty()
        self.vbar_g[:] = 0.0
        self.vbar.add(self.vbar_g)

        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        if (self.rank_a is not None and rank_a is not None and
            self.dH_asp is None and (rank_a == self.gd.comm.rank).any()):
            self.dH_asp = {}

        if self.dH_asp is not None:
            requests = []
            dH_asp = {}
            for a in self.vbar.my_atom_indices: #XXX a better way to obtain?
                if a in self.dH_asp:
                    dH_asp[a] = self.dH_asp.pop(a)
                else:
                    # Get matrix from old domain:
                    ni = self.setups[a].ni
                    dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                    dH_asp[a] = dH_sp
                    requests.append(self.gd.comm.receive(dH_sp, self.rank_a[a],
                                                         tag=a, block=False))
            for a, dH_sp in self.dH_asp.items():
                # Send matrix to new domain:
                requests.append(self.gd.comm.send(dH_sp, rank_a[a],
                                                  tag=a, block=False))
            for request in requests:
                self.gd.comm.wait(request)
            self.dH_asp = dH_asp

        if self.xc.xcfunc.mgga:
            self.xc.set_positions(spos_ac)
            
        self.rank_a = rank_a

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""


        self.timer.start('Hamiltonian')

        if self.vt_sg is None:
            self.vt_sg = self.finegd.empty(self.nspins)
            self.vHt_g = self.finegd.zeros()
            self.vt_sG = self.gd.empty(self.nspins)
            self.poisson.initialize()

        Ebar = self.finegd.integrate(self.vbar_g, density.nt_g,
                                     global_integral=False) 

        vt_g = self.vt_sg[0]
        vt_g[:] = self.vbar_g

        Eext = 0.0
        if self.vext_g is not None:
            vt_g += self.vext_g.get_potential(self.finegd)
            Eext = np.vdot(vt_g, density.nt_g) * self.finegd.dv - Ebar

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

        Epot = 0.5 * self.finegd.integrate(self.vHt_g, density.rhot_g,
                                           global_integral=False)
        Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, density.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            Ekin -= self.gd.integrate(vt_G, nt_G - density.nct_G,
                                      global_integral=False)
            
        # Calculate atomic hamiltonians:
        self.timer.start('Hamiltonian: atomic')
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((self.setups[a].lmax + 1)**2)
        self.timer.start('Hamiltonian: atomic: integrate ghat')
        density.ghat.integrate(self.vHt_g, W_aL)
        self.timer.stop('Hamiltonian: atomic: integrate ghat')
        self.dH_asp = {}
        for a, D_sp in density.D_asp.items():
            W_L = W_aL[a]
            setup = self.setups[a]

            D_p = D_sp.sum(0)
            dH_p = (setup.K_p + setup.M_p +
                    setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                    np.dot(setup.Delta_pL, W_L))
            Ekin += np.dot(setup.K_p, D_p) + setup.Kc
            Ebar += setup.MB + np.dot(setup.MB_p, D_p)
            Epot += setup.M + np.dot(D_p, (setup.M_p + np.dot(setup.M_pp, D_p)))

            if setup.HubU is not None:
##                 print '-----'
                nspins = len(self.D_sp)
                i0 = setup.Hubi
                i1 = i0 + 2 * setup.Hubl + 1
                for D_p, H_p in zip(self.D_sp, self.H_sp): # XXX self.H_sp ??
                    N_mm = unpack2(D_p)[i0:i1, i0:i1] / 2 * nspins 
                    Eorb = setup.HubU/2. * (N_mm - np.dot(N_mm,N_mm)).trace()
                    Vorb = setup.HubU * (0.5 * np.eye(i1-i0) - N_mm)
##                     print '========='
##                     print 'occs:',np.diag(N_mm)
##                     print 'Eorb:',Eorb
##                     print 'Vorb:',np.diag(Vorb)
##                     print '========='
                    Exc += Eorb                    
                    Htemp = unpack(H_p)
                    Htemp[i0:i1,i0:i1] += Vorb
                    H_p[:] = pack2(Htemp)

            if 0:#vext is not None:
                # Tailor expansion to the zeroth order
                Eext += vext[0][0] * (sqrt(4 * pi) * self.Q_L[0] + setup.Z)
                dH_p += vext[0][0] * sqrt(4 * pi) * setup.Delta_pL[:, 0]
                if len(vext) > 1:
                    # Tailor expansion to the first order
                    Eext += sqrt(4 * pi / 3) * np.dot(vext[1], self.Q_L[1:4])
                    # there must be a better way XXXX
                    Delta_p1 = np.array([setup.Delta_pL[:, 1],
                                          setup.Delta_pL[:, 2],
                                          setup.Delta_pL[:, 3]])
                    dH_p += sqrt(4 * pi / 3) * np.dot(vext[1], Delta_p1)

            self.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            self.timer.start('Hamiltonian: xc_correction')
            Exc += setup.xc_correction.calculate_energy_and_derivatives(
                D_sp, dH_sp, a)
            self.timer.stop('Hamiltonian: xc_correction')
            dH_sp += dH_p

            Ekin -= (D_sp * dH_sp).sum()

        self.timer.stop('Hamiltonian: atomic')

        # Make corrections due to non-local xc:
        xcfunc = self.xc.xcfunc
        self.Enlxc = xcfunc.get_non_local_energy()
        self.Enlkin = xcfunc.get_non_local_kinetic_corrections()
        if self.Enlxc != 0 or self.Enlkin != 0:
            print 'Where should we do comm.sum() ?'

        energies = np.array([Ekin, Epot, Ebar, Eext, Exc])
        self.gd.comm.sum(energies)
        (self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc) = energies

        self.Exc += self.Enlxc
        self.Ekin0 += self.Enlkin

        self.timer.stop('Hamiltonian')

    def get_energy(self, occupations):
        self.Ekin = self.Ekin0 + occupations.Eband
        self.S = occupations.S  # entropy

        # Total free energy:
        self.Etot = (self.Ekin + self.Epot + self.Eext + 
                     self.Ebar + self.Exc - self.S)
        #print self.Etot, self.Ekin, self.Epot, self.Eext, self.Ebar, self.Exc,
        #print self.S, self.Enlxc,self.Enlkin

        return self.Etot

    def apply_local_potential(self, psit_nG, Htpsit_nG, s):
        """Apply the Hamiltonian operator to a set of vectors.

        XXX Parameter description is deprecated!
        
        Parameters:

        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting H times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_projections: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_uni are used
        local_part_only: bool
            When True, the non-local atomic parts of the Hamiltonian
            are not applied and calculate_projections is ignored.
        
        """
        vt_G = self.vt_sG[s]
        if psit_nG.ndim == 3:
            Htpsit_nG += psit_nG * vt_G
        else:
            for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
                Htpsit_G += psit_G * vt_G

    def apply(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply the Hamiltonian operator to a set of vectors.

        Parameters:

        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        wfs: WaveFunctions
            Wave-function object defined in wavefunctions.py
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used
        
        """

        wfs.kin.apply(a_xG, b_xG, kpt.phase_cd)
        self.apply_local_potential(a_xG, b_xG, kpt.s)
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani: #TODO calculate_P_ani=False is experimental
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            dH_ii = unpack(self.dH_asp[a][kpt.s])
            P_axi[a] = np.dot(P_xi, dH_ii)
        wfs.pt.add(b_xG, P_axi, kpt.q)

    def get_xc_difference(self, xcname, wfs, density, atoms):
        """Calculate non-selfconsistent XC-energy difference."""
        xc = self.xc
        oldxcfunc = xc.xcfunc

        if isinstance(xcname, str):
            newxcfunc = XCFunctional(xcname, self.nspins)
        else:
            newxcfunc = xcname
        
        newxcfunc.set_non_local_things(density, self, wfs, atoms,
                                       energy_only=True)

        xc.set_functional(newxcfunc)
        xc.set_positions(atoms.get_scaled_positions() % 1.0)
        for setup in self.setups.setups.values():
            setup.xc_correction.xc.set_functional(newxcfunc)
            if newxcfunc.mgga:
                setup.xc_correction.initialize_kinetic(setup.data)

        if newxcfunc.hybrid > 0.0 and not self.nuclei[0].ready: #bugged?
            self.set_positions(npy.array([n.spos_c * self.domain.cell_c
                                          for n in self.nuclei]), self.rank_a)
        if newxcfunc.hybrid > 0.0:
            for nucleus in self.my_nuclei:
                nucleus.allocate_non_local_things(self.nmyu,self.mynbands)
        
        vt_g = self.finegd.empty()  # not used for anything!
        if density.nt_sg is None:
            density.interpolate()
        nt_sg = density.nt_sg
        if self.nspins == 2:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g, nt_sg[1], vt_g)
        else:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g)

        for a, D_sp in density.D_asp.items():
            setup = self.setups[a]
            Exc += setup.xc_correction.calculate_energy_and_derivatives(
                D_sp, np.zeros_like(D_sp), a)

        Exc = self.gd.comm.sum(Exc)

        for kpt in wfs.kpt_u:
            newxcfunc.apply_non_local(kpt)
        Exc += newxcfunc.get_non_local_energy()

        xc.set_functional(oldxcfunc)
        for setup in self.setups.setups.values():
            setup.xc_correction.xc.set_functional(oldxcfunc)

        return Exc - self.Exc

    def get_vxc(self, density, wfs):
        """Calculate matrix elements of the xc-potential."""
        dtype = wfs.dtype
        nbands = wfs.nbands
        nu = len(wfs.kpt_u)
        if density.nt_sg is None:
            density.interpolate()

        # Allocate space for result matrix
        Vxc_unn = np.empty((nu, nbands, nbands), dtype=dtype)

        # Get pseudo xc potential on the coarse grid
        Vxct_sG = self.gd.empty(self.nspins)
        Vxct_sg = self.finegd.zeros(self.nspins)
        if nspins == 1:
            self.xc.get_energy_and_potential(density.nt_sg[0], Vxct_sg[0])
        else:
            self.xc.get_energy_and_potential(density.nt_sg[0], Vxct_sg[0],
                                             density.nt_sg[1], Vxct_sg[1])
        for Vxct_G, Vxct_g in zip(Vxct_sG, Vxct_sg):
            self.restrict(Vxct_g, Vxct_G)
        del Vxct_sg

        # Get atomic corrections to the xc potential
        Vxc_asp = {}
        for a, D_sp in density.D_asp.items():
            Vxc_asp[a] = np.zeros_like(D_sp)
            self.setups[a].xc_correction.calculate_energy_and_derivatives(
                D_sp, Vxc_asp[a])

        # Project potential onto the eigenstates
        for kpt, Vxc_nn in xip(wfs.kpt_u, Vxc_unn):
            s, q = kpt.s, kpt.q
            psit_nG = kpt.psit_nG

            # Project pseudo part
            r2k(.5 * self.gd.dv, psit_nG, Vxct_sG[s] * psit_nG, 0.0, Vxc_nn)
            tri2full(Vxc_nn, 'L')
            self.gd.comm.sum(Vxc_nn)

            # Add atomic corrections
            for a, P_ni in kpt.P_ani.items():
                Vxc_ii = unpack(Vxc_asp[a][s])
                Vxc_nn += np.dot(P_ni.conj(), np.dot(H_ii, P_ni.T))
        return Vxc_unn

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()
        arrays = mem.subnode('Arrays', 0)
        arrays.subnode('vHt_g', nfinebytes)
        arrays.subnode('vt_sG', self.nspins * nbytes)
        arrays.subnode('vt_sg', self.nspins * nfinebytes)
        self.restrictor.estimate_memory(mem.subnode('Restrictor'))
        self.xc.estimate_memory(mem.subnode('XC 3D grid'))
        self.poisson.estimate_memory(mem.subnode('Poisson'))
        self.vbar.estimate_memory(mem.subnode('vbar'))
