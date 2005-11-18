"""Module defining a ``WaveFunctions`` class."""

import sys
import os
from math import pi, sqrt, log
import time

import Numeric as num
from ASE.ChemicalElements.symbol import symbols
from ASE.Units import Convert

from gridpaw.kpoint import KPoint
from gridpaw.utilities.complex import cc, real
from gridpaw.utilities import run_threaded, pack, unpack2
from gridpaw.utilities.timing import Timer
from gridpaw.operators import Laplace
import gridpaw.occupations as occupations
from gridpaw.preconditioner import Preconditioner


MASTER = 0


class WaveFunctions:
    """Class for handling wave functions.

    This object is a container for **k**-points (there may only be one
    **k**-point).  A wave-function object does not do any work it self
    - it delegates work (diagonalization, orthonormalization, ...) to
    a list of ``KPoint`` objects (the **k**-point object stores the
    actual wave functions, occupation numbers and eigenvalues).  Each
    **k**-point object can be either spin up, spin down or no spin
    (spin-saturated calculation).  Example: For a spin-polarized
    calculation on an isolated molecule, the **k**-point list will
    have length two (assuming the calculation is not parallelized over
    **k**-points/spin)."""
    
    def __init__(self, gd, nvalence, nbands, nspins,
                 typecode, kT,
                 bzk_kc, ibzk_kc, weights_k,
                 myspins, myibzk_kc, myweights_k, kpt_comm):
        """Construct wave-function object.

         =============== ===================================================
         ``gd``          Descriptor for wave-function grid.
         ``nvalence``    Number of valence electrons.
         ``nbands``      Number of bands.
         ``nspins``      Number of spins.
         ``typecode``    Data type of wave functions (``Float`` or
                         ``Complex``).
         ``kT``          Temperature for Fermi-distribution.
         ``bzk_ki``      Scaled **k**-points used for sampling the whole
                         Brillouin zone - values scaled to [-0.5, 0.5).  
         ``ibzk_ki``     Scaled **k**-points in the irreducible part of the
                         Brillouin zone.
         ``myspins``     List of spin-indices for this CPU.
         ``weights_k``   Weights of the **k**-points in the irreducible part
                         of the Brillouin zone (summing up to 1).
         ``myibzk_ki``   Scaled **k**-points in the irreducible part of the
                         Brillouin zone for this CPU.
         ``myweights_k`` Weights of the **k**-points on this CPU.
         ``kpt_comm``    MPI-communicator for parallelization over
                         **k**-points.
         =============== ===================================================

        Attributes:

         ================== =================================================
         ``kpt_u``          List of **k**-point objects.
         ``kin``            Finite-difference Laplacian (times -0.5).
         ``preconditioner`` Preconditioner object.
         ``occupation``     Occupation-number object.
         ``nkpts``          Number of irreducible **k**-points.
         ``nmykpts``        Number of irreducible **k**-points on *this* CPU.
         ================== =================================================
        """

        self.nvalence = nvalence
        self.nbands = nbands
        self.nspins = nspins
        self.typecode = typecode
        self.bzk_kc = bzk_kc
        self.ibzk_kc = ibzk_kc
        self.myibzk_kc = myibzk_kc
        self.weights_k = weights_k
        self.kpt_comm = kpt_comm
        
        self.nkpts = len(ibzk_kc)
        self.nmykpts = len(myibzk_kc)

        self.kpt_u = []
        u = 0
        for s in myspins: 
            for k, k_c in enumerate(myibzk_kc):
                weight = myweights_k[k] * 2 / nspins
                self.kpt_u.append(KPoint(gd, weight, s, k, u, k_c, typecode))
                u += 1
        
        # Kinetic energy operator:
        self.kin = Laplace(gd, -0.5, 2, typecode)

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(gd, self.kin, typecode)

        # Move all this occupation number stuff to occupation.py XXX
        # Create object for occupation numbers:
        if kT == 0 or 2 * nbands == nvalence:
            self.occupation = occupations.ZeroKelvin(nvalence, nspins)
        else:
            self.occupation = occupations.FermiDirac(nvalence, nspins, kT)
        self.occupation.set_communicator(kpt_comm)
        self.occupation.kT = kT # XXX

    def initialize_from_atomic_orbitals(self, nuclei, my_nuclei, out):
        # count the total number of atomic orbitals (bands):
        nao = 0
        for nucleus in nuclei:
            nao += nucleus.get_number_of_atomic_orbitals()

        print >> out, self.nbands, 'band%s.' % 's'[:self.nbands != 1]
        n = min(self.nbands, nao)
        if n == 1:
            string = 'Initializing one band from'
        else:
            string = 'Initializing %d bands from' % n
        if nao == 1:
            string += ' one atomic orbital.'
        else:
            string += ' linear combination of %d atomic orbitals.' % nao
        print >> out, string

        for nucleus in my_nuclei:
            # XXX already allocated once, but with wrong size!!!
            D_sp = nucleus.D_sp # XXXXXXXXXXX
            nucleus.allocate(self.nspins, self.nmykpts, nao)
            nucleus.D_sp = D_sp # XXXXXXXXXXX

        for kpt in self.kpt_u:
            kpt.create_atomic_orbitals(nao, nuclei)

    def calculate_occupation_numbers(self):
        return self.occupation.calculate(self.kpt_u)

    def calculate_electron_density(self, nt_sG, nct_G, symmetry, gd):
        """Calculate pseudo electron-density.

        The pseudo electron-density ``nt_sG`` is calculated from the
        wave functions, the occupation numbers, and the smoot core
        density ``nct_G``, and finally symmetrized."""
        
        nt_sG[:] = 0.0

        # Add contribution from all k-points:
        for kpt in self.kpt_u:
            kpt.add_to_density(nt_sG[kpt.s])

        self.kpt_comm.sum(nt_sG)

        # add the smooth core density:
        nt_sG += nct_G

        if symmetry is not None:
            for nt_G in nt_sG:
                symmetry.symmetrize(nt_G, gd)

    def calculate_projections_and_orthogonalize(self, p_nuclei, my_nuclei):
        for kpt in self.kpt_u:
            for nucleus in p_nuclei:
                nucleus.calculate_projections(kpt)

        run_threaded([kpt.orthonormalize(my_nuclei) for kpt in self.kpt_u])

    def diagonalize(self, vt_sG, my_nuclei):
        """Apply Hamiltonian and do subspace diagonalization."""
##        for kpt in self.kpt_u:
##            kpt.diagonalize(self.kin, vt_sG, my_nuclei, self.nbands)
        run_threaded([kpt.diagonalize(self.kin, vt_sG, my_nuclei, self.nbands)
                      for kpt in self.kpt_u])

    def sum_eigenvalues(self):
        Eeig = 0.0
        for kpt in self.kpt_u:
            Eeig += num.dot(kpt.f_n, kpt.eps_n)    
        return self.kpt_comm.sum(Eeig)

    def calculate_residuals(self, p_nuclei):
        error = 0.0
        for kpt in self.kpt_u:
            error += kpt.calculate_residuals(p_nuclei)
        return self.kpt_comm.sum(error) / self.nvalence

    def rmm_diis(self, p_nuclei, vt_sG):
        for kpt in self.kpt_u:
            kpt.rmm_diis(p_nuclei, self.preconditioner, self.kin, vt_sG)
    
    def calculate_force_contribution(self, p_nuclei, my_nuclei):
        for kpt in self.kpt_u:
            for nucleus in p_nuclei:
                nucleus.calculate_force_kpoint(kpt)

        for nucleus in my_nuclei:
            self.kpt_comm.sum(nucleus.F_c)

    def calculate_atomic_density_matrices(self, my_nuclei, nuclei,
                                          comm, symmetry):
        for nucleus in my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            D_sii = num.zeros((self.nspins, ni, ni), num.Float)
            for kpt in self.kpt_u:
                P_ni = nucleus.P_uni[kpt.u]
                D_sii[kpt.s] += real(num.dot(cc(num.transpose(P_ni)),
                                               P_ni * kpt.f_n[:, None]))
            nucleus.D_sp[:] = [pack(D) for D in D_sii]

            self.kpt_comm.sum(nucleus.D_sp)

        if symmetry is not None:
            D_asp = []
            for nucleus in nuclei:
                if comm.rank == nucleus.rank:
                    assert nucleus.domain_overlap == 3 # EVERYTHING
                    D_sp = nucleus.D_sp
                    comm.broadcast(D_sp, nucleus.rank)
                else:
                    ni = nucleus.get_number_of_partial_waves()
                    np = ni * (ni + 1) / 2
                    D_sp = num.zeros((self.nspins, np), num.Float)
                    comm.broadcast(D_sp, nucleus.rank)
                D_asp.append(D_sp)

            for s in range(self.nspins):
                D_aii = [unpack2(D_sp[s]) for D_sp in D_asp]
                for nucleus in my_nuclei:
                    nucleus.symmetrize(D_aii, symmetry.maps, s)

    def print_eigenvalues(self, out, Ha):
        if (self.kpt_comm.size > 1 or
            self.kpt_comm.size * self.nmykpts / self.nspins > 1):
            # not implemented yet:
            return
        
        if self.nspins == 1:
            print >> out, ' band     eps        occ'
            kpt = self.kpt_u[0]
            for n in range(self.nbands):
                print >> out, ('%4d %10.5f %10.5f' %
                               (n, Ha * kpt.eps_n[n], kpt.f_n[n]))
        else:
            print >> out, '                up                   down'
            print >> out, ' band     eps        occ        eps        occ'
            epsa_n = self.kpt_u[0].eps_n
            epsb_n = self.kpt_u[1].eps_n
            fa_n = self.kpt_u[0].f_n
            fb_n = self.kpt_u[1].f_n
            for n in range(self.nbands):
                print >> out, ('%4d %10.5f %10.5f %10.5f %10.5f' %
                               (n,
                                Ha * epsa_n[n], fa_n[n],
                                Ha * epsb_n[n], fb_n[n]))
