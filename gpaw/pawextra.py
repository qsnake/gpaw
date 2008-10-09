# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module will go away some dat in the future.  The few methods
in the PAWExtra class should be moved to the PAW class or other
modules."""

import sys

import numpy as npy

import gpaw.io
import gpaw.mpi as mpi
from gpaw.xc_functional import XCFunctional
from gpaw.density import Density
from gpaw.utilities import pack
from gpaw.mpi import run, MASTER


class PAWExtra:
    def get_fermi_level(self):
        """Return the Fermi-level."""
        e = self.occupation.get_fermi_level()
        if e is None:
            # Zero temperature calculation - return vacuum level:
            e = 0.0
        return e * self.Ha

    def write(self, filename, mode=''):
        """use mode='all' to write the wave functions"""
        self.timer.start('IO')
        gpaw.io.write(self, filename, mode)
        self.timer.stop('IO')
        
    def get_reference_energy(self):
        return self.Eref * self.Ha
    
    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)
        nn, band_rank = divmod(n, self.band_comm.size)

        psit_nG = self.kpt_u[u].psit_nG
        if psit_nG is None:
            raise RuntimeError('This calculator has no wave functions!')

        if self.world.size == 1:
            return psit_nG[nn][:]

        if self.kpt_comm.rank == kpt_rank:
            if self.band_comm.rank == band_rank:
                psit_G = self.gd.collect(psit_nG[nn][:])

                if kpt_rank == MASTER and band_rank == MASTER:
                    if self.master:
                        return psit_G

                # Domain master send this to the global master
                if self.domain.comm.rank == MASTER:
                    self.world.send(psit_G, MASTER, 1398)

        if self.master:
            # allocate full wavefunction and receive
            psit_G = self.gd.empty(dtype=self.dtype, global_array=True)
            world_rank = kpt_rank * self.domain.comm.size * self.band_comm.size + band_rank * self.domain.comm.size
            self.world.receive(psit_G, world_rank, 1398)
            return psit_G

    def collect_eigenvalues(self, k=0, s=0):
        """Return eigenvalue array.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)
        # Does this work correctly? Strides?
        return self.collect_array(self.kpt_u[u].eps_n, kpt_rank)

        if kpt_rank == MASTER:
            if self.band_comm.size == 1:
                return self.kpt_u[u].eps_n
                

        if self.kpt_comm.rank == kpt_rank:
            if not (kpt_rank == MASTER and self.master):
                # Domain master send this to the global master
                if self.domain.comm.rank == MASTER:
                    self.world.send(self.kpt_u[u].eps_n, MASTER, 1301)
        elif self.master:
            eps_all_n = npy.zeros(self.nbands)
            nstride = self.band_comm.size
            eps_n = npy.zeros(self.nmybands)
            r0 = 0
            if kpt_rank == MASTER:
                # Master has already the first slice
                eps_all_n[0::nstride] = self.kpt_u[u].eps_n
                r0 = 1
            for r in range(r0, self.band_comm.size):
                world_rank = kpt_rank * self.domain.comm.size * self.band_comm.size + r * self.domain.comm.size
                self.world.receive(eps_n, world_rank, 1301)
                eps_all_n[r::nstride] = eps_n

            return eps_all_n

    def collect_occupations(self, k=0, s=0):
        """Return occupation array.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)
        return self.collect_array(self.kpt_u[u].f_n, kpt_rank)

        if kpt_rank == MASTER:
            if self.band_comm.size == 1:
                return self.kpt_u[u].f_n

        if self.kpt_comm.rank == kpt_rank:
            if not (kpt_rank == MASTER and self.master):
                # Domain master send this to the global master
                if self.domain.comm.rank == MASTER:
                    self.world.send(self.kpt_u[u].f_n, MASTER, 1313)
        elif self.master:
            f_all_n = npy.zeros(self.nbands)
            nstride = self.band_comm.size
            f_n = npy.zeros(self.nbands)
            r0 = 0
            if kpt_rank == MASTER:
                # Master has already the first slice
                f_all_n[0::nstride] = self.kpt_u[u].f_n
                r0 = 1
            for r in range(r0, self.band_comm.size):
                world_rank = kpt_rank * self.domain.comm.size * self.band_comm.size + r * self.domain.comm.size
                self.world.receive(f_n, world_rank, 1313)
                f_all_n[r::nstride] = f_n
            return f_all_n

    def collect_array(self, a_n, kpt_rank):
        """Helper method for collect_eigenvalues and collect_occupations."""
        if kpt_rank == 0:
            if self.band_comm.size == 1:
                return a_n
            
            if self.band_comm.rank == 0:
                b_n = npy.zeros(self.nbands)
            else:
                b_n = None
            self.band_comm.gather(a_n, 0, b_n)
            return b_n

        if self.kpt_comm.rank == kpt_rank:
            # Domain master send this to the global master
            if self.domain.comm.rank == 0:
                if self.band_comm.size == 1:
                    self.kpt_comm.send(a_n, 0, 1301)
                else:
                    if self.band_comm.rank == 0:
                        b_n = npy.zeros(self.nbands)
                    else:
                        b_n = None
                    self.band_comm.gather(a_n, 0, b_n)
                    if self.band_comm.rank == 0:
                        self.kpt_comm.send(b_n, 0, 1301)

        elif self.master:
            b_n = npy.zeros(self.nbands)
            self.kpt_comm.receive(b_n, kpt_rank, 1301)
            return b_n

        # return something also on the slaves
        # might be nicer to have the correct array everywhere XXXX 
        return a_n

    def get_wannier_integrals(self, c, s, k, k1, G, nbands=None):
        """Calculate integrals for maximally localized Wannier functions."""

        assert s <= self.nspins

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)
        kpt_rank1, u1 = divmod(k1 + self.nkpts * s, self.nmyu)

        # XXX not for the kpoint/spin parallel case
        assert self.kpt_comm.size == 1

        # If calc is a save file, read in tar references to memory
        sel.initialize_wave_functions()
        
        # Get pseudo part
        Z_nn = self.gd.wannier_matrix(self.kpt_u[u].psit_nG,
                                      self.kpt_u[u1].psit_nG, c, G, nbands)

        # Add corrections
        for nucleus in self.my_nuclei:
            Z_nn += nucleus.wannier_correction(G, c, u, u1, nbands)

        self.gd.comm.sum(Z_nn, 0)
            
        return Z_nn

    def get_xc_difference(self, xcname):
        """Calculate non-selfconsistent XC-energy difference."""
        xc = self.hamiltonian.xc
        oldxcfunc = xc.xcfunc

        if isinstance(xcname, str):
            newxcfunc = XCFunctional(xcname, self.nspins)
        else:
            newxcfunc = xcname
        
        newxcfunc.set_non_local_things(self, energy_only=True)

        xc.set_functional(newxcfunc)
        for setup in self.setups:
            setup.xc_correction.xc.set_functional(newxcfunc)

        if newxcfunc.hybrid > 0.0 and not self.nuclei[0].ready: #bugged?
            self.set_positions(npy.array([n.spos_c * self.domain.cell_c
                                          for n in self.nuclei]))
        if newxcfunc.hybrid > 0.0:
            for nucleus in self.my_nuclei:
                nucleus.allocate_non_local_things(self.nmyu,self.nmybands)
        
        vt_g = self.finegd.empty()  # not used for anything!
        nt_sg = self.density.nt_sg
        if self.nspins == 2:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g, nt_sg[1], vt_g)
        else:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g)

        for nucleus in self.my_nuclei:
            D_sp = nucleus.D_sp
            H_sp = npy.zeros(D_sp.shape) # not used for anything!
            xc_correction = nucleus.setup.xc_correction
            Exc += xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)

        Exc = self.domain.comm.sum(Exc)

        for kpt in self.kpt_u:
            newxcfunc.apply_non_local(kpt)
        Exc += newxcfunc.get_non_local_energy()

        xc.set_functional(oldxcfunc)
        for setup in self.setups:
            setup.xc_correction.xc.set_functional(oldxcfunc)

        return self.Ha * (Exc - self.Exc)

    def get_grid_spacings(self):
        return self.a0 * self.gd.h_c

    def get_exact_exchange(self):
        dExc = self.get_xc_difference('EXX') / self.Ha
        Exx = self.Exc + dExc
        for nucleus in self.nuclei:
            Exx += nucleus.setup.xc_correction.Exc0
        return Exx

    def get_weights(self):
        return self.weight_k #???

    def initialize_from_wave_functions(self):
        """Initialize density and Hamiltonian from wave functions"""
        
        self.set_positions()
        self.density.move()
        self.density.update(self.kpt_u, self.symmetry)
##         if self.wave_functions_initialized:
##             self.density.move()
##             self.density.update(self.kpt_u, self.symmetry)
##         else:
##             # no wave-functions: restart from LCAO
##             self.initialize_wave_functions()

    def totype(self, dtype):
        """Converts all the dtype dependent quantities of Paw
        (Laplacian, wavefunctions etc.) to dtype"""

        from gpaw.operators import Laplace

        if dtype not in [float, complex]:
            raise RuntimeError('PAW can be converted only to Float or Complex')

        self.dtype = dtype

        # Hamiltonian
        nn = self.stencils[0]
        self.hamiltonian.kin = Laplace(self.gd, -0.5, nn, dtype)

        # Nuclei
        for nucleus in self.nuclei:
            nucleus.dtype = dtype
            nucleus.ready = False

        # reallocate only my_nuclei (as the others are not allocated at all)
        for nucleus in self.my_nuclei:
            nucleus.reallocate(self.nmybands)

        self.set_positions()

        # Wave functions
        for kpt in self.kpt_u:
            kpt.dtype = dtype
            kpt.psit_nG = npy.array(kpt.psit_nG[:], dtype)

        # Eigensolver
        # !!! FIX ME !!!
        # not implemented yet...

    def read_wave_functions(self, mode='gpw'):
        """Read wave functions one by one from seperate files"""

        for u in range(self.nmyu):
            kpt = self.kpt_u[u]
            kpt.psit_nG = self.gd.empty(self.nbands, self.dtype)
            # Read band by band to save memory
            s = kpt.s
            k = kpt.k
            for n, psit_G in enumerate(kpt.psit_nG):
                psit_G[:] = gpaw.io.read_wave_function(self.gd, s, k, n, mode)
                
    def warn(self, string=None):
        if not string:
            string = "somethings wrong"
        print >> self.txt, "WARNING >>"
        print >> self.txt, string
        print >> self.txt, "WARNING <<"
                
    def wave_function_volumes(self):
        """Return the volume needed by the wave functions"""
        nu = self.nkpts * self.nspins
        volumes = npy.empty((nu,self.nbands))

        for k in range(nu):
            for n, psit_G in enumerate(self.kpt_u[k].psit_nG):
                volumes[k, n] = self.gd.integrate(psit_G**4)

                # atomic corrections
                for nucleus in self.my_nuclei:
                    # make sure the integrals are there
                    nucleus.setup.four_phi_integrals()
                    P_i = nucleus.P_uni[k, n]
                    ni = len(P_i)
                    P_ii = npy.outer(P_i, P_i)
                    P_p = pack(P_ii)
                    I = 0
                    for i1 in range(ni):
                        for i2 in range(ni):
                            I += P_ii[i1, i2] * npy.dot(P_p,
                                             nucleus.setup.I4_iip[i1, i2])
                volumes[k, n] += I
                
        return 1. / volumes

    def get_homo_lumo(self):
        """Return HOMO and LUMO eigenvalues.

        Works for zero-temperature Gamma-point calculations only.
        """
        if len(self.bzk_kc) != 1 or self.kT != 0:
            raise RuntimeError
        occ = self.collect_occupations()
        eig = self.collect_eigenvalues()
        lumo = self.nbands - occ[::-1].searchsorted(0, side='right')
        homo = lumo - 1
        e_homo = eig[homo]
        e_lumo = eig[lumo]
        if self.nspins == 2:
            # Spin polarized: check if frontier orb is in minority spin
            occ = self.collect_occupations(s=1)
            eig = self.collect_eigenvalues(s=1)
            lumo = self.nbands - occ[::-1].searchsorted(0, side='right')
            homo = lumo - 1
            if homo >= 0:
                e_homo = max(e_homo, eig[homo])
            e_lumo = min(e_lumo, eig[lumo])

        return e_homo * self.Ha, e_lumo * self.Ha

    def get_projections(self, locfun):
        """Project wave functions onto localized functions

        Determine the projections of the Koh-Sham eigenstates
        onto specified localized functions of the format::

          locfun = [[spos_c, l, a], [...]]

        Return format is::

          f_kni = <psi_kn | f_i>

        where psi_kn are the wave functions, and f_i are the specified
        localized functions.
        """
        from gpaw.localized_functions import create_localized_functions
        from gpaw.spline import Spline

        nbf = 0
        for spos_c, l, a in locfun:
            nbf += 2 * l + 1
        f_kni = npy.zeros((len(self.ibzk_kc), self.nbands, nbf), complex)

        nbf = 0
        for spos_c, l, a in locfun:
            if type(spos_c) is int:
                spos_c = self.nuclei[spos_c].spos_c

            a /= self.a0
            cutoff = 10 * a
            x = npy.arange(0.0, cutoff, cutoff / 500.0)
            rad_g = npy.exp(-x * x / a)
            rad_g[-2:] = 0.0
            functions = [Spline(l, cutoff, rad_g)]
            lf = create_localized_functions(functions, self.gd, spos_c,
                                            dtype=complex)
            lf.set_phase_factors(self.ibzk_kc)
            nlf = 2 * l + 1
            nbands = self.nbands
            nkpts = len(self.ibzk_kc)
            for k in range(nkpts):
                lf.integrate(self.kpt_u[k].psit_nG[:],
                             f_kni[k, :, nbf:nbf + nlf], k=k)
            nbf += nlf
        return f_kni.conj()

