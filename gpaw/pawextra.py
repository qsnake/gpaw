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
        return self.occupation.get_fermi_level() * self.Ha

    def get_homo_lumo(self):
        """Return HOMO and LUMO eigenvalues."""
        return self.occupation.get_homo_lumo(self.kpt_u) * self.Ha

    def write(self, filename, mode=''):
        """use mode='all' to write the wave functions"""
        self.timer.start('IO')
        gpaw.io.write(self, filename, mode)
        self.timer.stop('IO')
        
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
        kpt_u = self.wfs.kpt_u
        
        # Does this work correctly? Strides?
        return self.collect_array(kpt_u[u].eps_n, kpt_rank)

        if kpt_rank == MASTER:
            if self.band_comm.size == 1:
                return kpt_u[u].eps_n
                

        if self.kpt_comm.rank == kpt_rank:
            if not (kpt_rank == MASTER and self.master):
                # Domain master send this to the global master
                if self.domain.comm.rank == MASTER:
                    self.world.send(kpt_u[u].eps_n, MASTER, 1301)
        elif self.master:
            eps_all_n = npy.zeros(self.nbands)
            nstride = self.band_comm.size
            eps_n = npy.zeros(self.mynbands)
            r0 = 0
            if kpt_rank == MASTER:
                # Master has already the first slice
                eps_all_n[0::nstride] = kpt_u[u].eps_n
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
        kpt_u = self.wfs.kpt_u
        return self.collect_array(kpt_u[u].f_n, kpt_rank)

        if kpt_rank == MASTER:
            if self.band_comm.size == 1:
                return kpt_u[u].f_n

        if self.kpt_comm.rank == kpt_rank:
            if not (kpt_rank == MASTER and self.master):
                # Domain master send this to the global master
                if self.domain.comm.rank == MASTER:
                    self.world.send(kpt_u[u].f_n, MASTER, 1313)
        elif self.master:
            f_all_n = npy.zeros(self.nbands)
            nstride = self.band_comm.size
            f_n = npy.zeros(self.nbands)
            r0 = 0
            if kpt_rank == MASTER:
                # Master has already the first slice
                f_all_n[0::nstride] = kpt_u[u].f_n
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
        self.density.update(self.wfs.kpt_u, self.symmetry)
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
            nucleus.reallocate(self.mynbands)

        self.set_positions()

        # Wave functions
        for kpt in self.wfs.kpt_u:
            kpt.dtype = dtype
            kpt.psit_nG = npy.array(kpt.psit_nG[:], dtype)

        # Eigensolver
        # !!! FIX ME !!!
        # not implemented yet...

    def read_wave_functions(self, mode='gpw'):
        """Read wave functions one by one from seperate files"""

        for u, kpt in enumerate(self.wfs.kpt_u):
            #kpt = self.kpt_u[u]
            kpt.psit_nG = self.gd.empty(self.nbands, self.dtype)
            # Read band by band to save memory
            s = kpt.s
            k = kpt.k
            for n, psit_G in enumerate(kpt.psit_nG):
                psit_G[:] = gpaw.io.read_wave_function(self.gd, s, k, n, mode)
                
    def wave_function_volumes(self):
        """Return the volume needed by the wave functions"""
        nu = self.nkpts * self.nspins
        volumes = npy.empty((nu, self.nbands))

        for k in range(nu):
            for n, psit_G in enumerate(self.wfs.kpt_u[k].psit_nG):
                volumes[k, n] = self.gd.integrate(psit_G**4)

                # atomic corrections
                for nucleus in self.my_nuclei:
                    I4_pp = nucleus.setup.four_phi_integrals()
                    P_i = nucleus.P_uni[k, n]
                    P_ii = npy.outer(P_i, P_i)
                    P_p = pack(P_ii)
                    volumes[k, n] += npy.dot(P_p, npy.dot(I4_pp, P_p))
                
        return 1. / volumes

