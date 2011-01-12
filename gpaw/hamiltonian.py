# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""

from math import pi, sqrt

import numpy as np

from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.lfc import LFC
from gpaw.utilities import pack2,unpack,unpack2
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

    def __init__(self, gd, finegd, nspins, setups, stencil, timer, xc,
                 psolver, vext_g):
        """Create the Hamiltonian."""
        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.setups = setups
        self.timer = timer
        self.xc = xc
        
        # Solver for the Poisson equation:
        if psolver is None:
            psolver = PoissonSolver(nn=3, relax='J')
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
        self.allocated = True

    def set_positions(self, spos_ac, rank_a=None):
        self.spos_ac = spos_ac
        if not self.allocated:
            self.allocate()
        self.vbar.set_positions(spos_ac)
        if self.vbar_g is None:
            self.vbar_g = self.finegd.empty()
        self.vbar_g[:] = 0.0
        self.vbar.add(self.vbar_g)

        self.xc.set_positions(spos_ac)
        
        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        if (self.rank_a is not None and rank_a is not None and
            self.dH_asp is None and (rank_a == self.gd.comm.rank).any()):
            self.dH_asp = {}

        if self.rank_a is not None and self.dH_asp is not None:
            self.timer.start('Redistribute')
            requests = []
            flags = (self.rank_a != rank_a)
            my_incoming_atom_indices = np.argwhere(np.bitwise_and(flags, \
                rank_a == self.gd.comm.rank)).ravel()
            my_outgoing_atom_indices = np.argwhere(np.bitwise_and(flags, \
                self.rank_a == self.gd.comm.rank)).ravel()

            for a in my_incoming_atom_indices:
                # Get matrix from old domain:
                ni = self.setups[a].ni
                dH_sp = np.empty((self.nspins, ni * (ni + 1) // 2))
                requests.append(self.gd.comm.receive(dH_sp, self.rank_a[a],
                                                     tag=a, block=False))
                assert a not in self.dH_asp
                self.dH_asp[a] = dH_sp

            for a in my_outgoing_atom_indices:
                # Send matrix to new domain:
                dH_sp = self.dH_asp.pop(a)
                requests.append(self.gd.comm.send(dH_sp, rank_a[a],
                                                  tag=a, block=False))
            self.gd.comm.waitall(requests)
            self.timer.stop('Redistribute')

        self.rank_a = rank_a

    def aoom(self, DM, a, l, scale=1):
        """Atomic Orbital Occupation Matrix.
        
        Determine the Atomic Orbital Occupation Matrix (aoom) for a
        given l-quantum number.
        
        This operation, takes the density matrix (DM), which for
        example is given by unpack2(D_asq[i][spin]), and corrects for
        the overlap between the selected orbitals (l) upon which the
        the density is expanded (ex <p|p*>,<p|p>,<p*|p*> ).

        Returned is only the "corrected" part of the density matrix,
        which represents the orbital occupation matrix for l=2 this is
        a 5x5 matrix.
        """
        S=self.setups[a]
        l_j = S.l_j
        n_j = S.n_j
        lq  = S.lq
        nl  = np.where(np.equal(l_j, l))[0]
        V = np.zeros(np.shape(DM))
        if len(nl) == 2:
            aa = (nl[0])*len(l_j)-((nl[0]-1)*(nl[0])/2)
            bb = (nl[1])*len(l_j)-((nl[1]-1)*(nl[1])/2)
            ab = aa+nl[1]-nl[0]
            
            if(scale==0 or scale=='False' or scale =='false'):
                lq_a  = lq[aa]
                lq_ab = lq[ab]
                lq_b  = lq[bb]
            else:
                lq_a  = 1
                lq_ab = lq[ab]/lq[aa]
                lq_b  = lq[bb]/lq[aa]
 
            # and the correct entrances in the DM
            nn = (2*np.array(l_j)+1)[0:nl[0]].sum()
            mm = (2*np.array(l_j)+1)[0:nl[1]].sum()
            
            # finally correct and add the four submatrices of NC_DM
            A = DM[nn:nn+2*l+1,nn:nn+2*l+1]*(lq_a)
            B = DM[nn:nn+2*l+1,mm:mm+2*l+1]*(lq_ab)
            C = DM[mm:mm+2*l+1,nn:nn+2*l+1]*(lq_ab)
            D = DM[mm:mm+2*l+1,mm:mm+2*l+1]*(lq_b)
            
            V[nn:nn+2*l+1,nn:nn+2*l+1]=+(lq_a)
            V[nn:nn+2*l+1,mm:mm+2*l+1]=+(lq_ab)
            V[mm:mm+2*l+1,nn:nn+2*l+1]=+(lq_ab)
            V[mm:mm+2*l+1,mm:mm+2*l+1]=+(lq_b)
 
            return  A+B+C+D, V
        else:
            nn =(2*np.array(l_j)+1)[0:nl[0]].sum()
            A=DM[nn:nn+2*l+1,nn:nn+2*l+1]*lq[-1]
            V[nn:nn+2*l+1,nn:nn+2*l+1]=+lq[-1]
            return A,V

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""

        self.timer.start('Hamiltonian')

        if self.vt_sg is None:
            self.timer.start('Initialize Hamiltonian')
            self.vt_sg = self.finegd.empty(self.nspins)
            self.vHt_g = self.finegd.zeros()
            self.vt_sG = self.gd.empty(self.nspins)
            self.poisson.initialize()
            self.timer.stop('Initialize Hamiltonian')

        self.timer.start('vbar')
        Ebar = self.finegd.integrate(self.vbar_g, density.nt_g,
                                     global_integral=False)

        vt_g = self.vt_sg[0]
        vt_g[:] = self.vbar_g
        self.timer.stop('vbar')

        Eext = 0.0
        if self.vext_g is not None:
            vt_g += self.vext_g.get_potential(self.finegd)
            Eext = self.finegd.integrate(vt_g, density.nt_g,
                                         global_integral=False) - Ebar

        if self.nspins == 2:
            self.vt_sg[1] = vt_g

        self.timer.start('XC 3D grid')
        Exc = self.xc.calculate(self.finegd, density.nt_sg, self.vt_sg)
        Exc /= self.gd.comm.size
        self.timer.stop('XC 3D grid')

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_g, density.rhot_g,
                                           charge=-density.charge)
        self.timer.stop('Poisson')

        self.timer.start('Hartree integrate/restrict')
        Epot = 0.5 * self.finegd.integrate(self.vHt_g, density.rhot_g,
                                           global_integral=False)
        Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, density.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            Ekin -= self.gd.integrate(vt_G, nt_G - density.nct_G,
                                      global_integral=False)
        self.timer.stop('Hartree integrate/restrict')
            
        # Calculate atomic hamiltonians:
        self.timer.start('Atomic')
        W_aL = {}
        for a in density.D_asp:
            W_aL[a] = np.empty((self.setups[a].lmax + 1)**2)
        density.ghat.integrate(self.vHt_g, W_aL)
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
            Epot += setup.M + np.dot(D_p, (setup.M_p +
                                           np.dot(setup.M_pp, D_p)))

            if self.vext_g is not None:
                vext = self.vext_g.get_taylor(spos_c=self.spos_ac[a, :])
                # Tailor expansion to the zeroth order
                Eext += vext[0][0] * (sqrt(4 * pi) * density.Q_aL[a][0]
                                      + setup.Z)
                dH_p += vext[0][0] * sqrt(4 * pi) * setup.Delta_pL[:, 0]
                if len(vext) > 1:
                    # Tailor expansion to the first order
                    Eext += sqrt(4 * pi / 3) * np.dot(vext[1],
                                                      density.Q_aL[a][1:4])
                    # there must be a better way XXXX
                    Delta_p1 = np.array([setup.Delta_pL[:, 1],
                                          setup.Delta_pL[:, 2],
                                          setup.Delta_pL[:, 3]])
                    dH_p += sqrt(4 * pi / 3) * np.dot(vext[1], Delta_p1)

            self.dH_asp[a] = dH_sp = np.zeros_like(D_sp)
            self.timer.start('XC Correction')
            Exc += setup.xc_correction.calculate(self.xc, D_sp, dH_sp, a)
            self.timer.stop('XC Correction')

            if setup.HubU is not None:
                nspins = len(D_sp)
                
                l_j = setup.l_j
                l   = setup.Hubl
                nl  = np.where(np.equal(l_j,l))[0]
                nn  = (2*np.array(l_j)+1)[0:nl[0]].sum()
                
                for D_p, H_p in zip(D_sp, self.dH_asp[a]):
                    [N_mm,V] =self.aoom(unpack2(D_p),a,l)
                    N_mm = N_mm / 2 * nspins
                     
                    Eorb = setup.HubU / 2. * (N_mm - np.dot(N_mm,N_mm)).trace()
                    Vorb = setup.HubU * (0.5 * np.eye(2*l+1) - N_mm)
                    Exc += Eorb
                    if nspins == 1:
                        # add contribution of other spin manyfold
                        Exc += Eorb
                    
                    if len(nl)==2:
                        mm  = (2*np.array(l_j)+1)[0:nl[1]].sum()
                        
                        V[nn:nn+2*l+1,nn:nn+2*l+1] *= Vorb
                        V[mm:mm+2*l+1,nn:nn+2*l+1] *= Vorb
                        V[nn:nn+2*l+1,mm:mm+2*l+1] *= Vorb
                        V[mm:mm+2*l+1,mm:mm+2*l+1] *= Vorb
                    else:
                        V[nn:nn+2*l+1,nn:nn+2*l+1] *= Vorb
                    
                    Htemp = unpack(H_p)
                    Htemp += V
                    H_p[:] = pack2(Htemp)

            dH_sp += dH_p

            Ekin -= (D_sp * dH_sp).sum()

        self.timer.stop('Atomic')

        # Make corrections due to non-local xc:
        #xcfunc = self.xc.xcfunc
        self.Enlxc = 0.0#XXXxcfunc.get_non_local_energy()
        Ekin += self.xc.get_kinetic_energy_correction() / self.gd.comm.size

        energies = np.array([Ekin, Epot, Ebar, Eext, Exc])
        self.timer.start('Communicate energies')
        self.gd.comm.sum(energies)
        self.timer.stop('Communicate energies')
        (self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc) = energies

        #self.Exc += self.Enlxc
        #self.Ekin0 += self.Enlkin

        self.timer.stop('Hamiltonian')

    def get_energy(self, occupations):
        self.Ekin = self.Ekin0 + occupations.e_band
        self.S = occupations.e_entropy

        # Total free energy:
        self.Etot = (self.Ekin + self.Epot + self.Eext +
                     self.Ebar + self.Exc - self.S)

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

    def get_xc_difference(self, xc, density):
        """Calculate non-selfconsistent XC-energy difference."""
        if density.nt_sg is None:
            density.interpolate()
        nt_sg = density.nt_sg
        if hasattr(xc, 'hybrid'):
            xc.calculate_exx()
        Exc = xc.calculate(density.finegd, nt_sg) / self.gd.comm.size
        for a, D_sp in density.D_asp.items():
            setup = self.setups[a]
            Exc += setup.xc_correction.calculate(xc, D_sp)
        Exc = self.gd.comm.sum(Exc)
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
            # H_ij = \int dr phi_i(r) Ĥ phi_j^*(r)
            # P_ni = \int dr psi_n(r) pt_i^*(r)
            # Vxc_nm = \int dr phi_n(r) vxc(r) phi_m^*(r)
            #      + sum_ij P_ni H_ij P_mj^*
            for a, P_ni in kpt.P_ani.items():
                Vxc_ii = unpack(Vxc_asp[a][s])
                Vxc_nn += np.dot(P_ni, np.inner(H_ii, P_ni).conj())
        return Vxc_unn

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()
        arrays = mem.subnode('Arrays', 0)
        arrays.subnode('vHt_g', nfinebytes)
        arrays.subnode('vt_sG', self.nspins * nbytes)
        arrays.subnode('vt_sg', self.nspins * nfinebytes)
        self.restrictor.estimate_memory(mem.subnode('Restrictor'))
        self.xc.estimate_memory(mem.subnode('XC'))
        self.poisson.estimate_memory(mem.subnode('Poisson'))
        self.vbar.estimate_memory(mem.subnode('vbar'))
