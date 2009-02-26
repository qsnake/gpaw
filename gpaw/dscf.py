# Copyright (C) 2008  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module is used in delta self-consistent field (dSCF) calculations

dSCF is a simple 'ad hoc' method to estimating excitation energies within
DFT. The only difference to ordinary DFT is that one or more electrons(s)
are forced to occupy one or more predefined orbitals. The only restriction
on these orbitals is that they must be linear combinations of available
Kohn-Sham orbitals.

"""

import copy
import numpy as np
from gpaw.occupations import OccupationNumbers, FermiDirac
import gpaw.mpi as mpi

def dscf_calculation(paw, orbitals):
    """Helper function to prepare a calculator for a dSCF calculation

    Parameters
    ==========
    orbitals: list of lists
        Orbitals which one wants to occupy. The format is
        orbitals = [[1.0,orb1,0],[1.0,orb2,1],...], where 1.0 is the no.
        of electrons, orb1 and orb2 are the orbitals (see MolecularOrbitals
        below for an example of an orbital class). 0 and 1 represents the
        spin (up and down). This number is ignored in a spin-paired
        calculation.

    Example
    =======
    
    >>> atoms.set_calculator(calc)
    >>> e_gs = atoms.get_potential_energy() #ground state energy
    >>> sigma_star=MolecularOrbitals(calc, molecule=[0,1],
    >>>                              w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]])
    >>> dscf_calculation(calc, [[1.0,sigma_star,1]])
    >>> e_exc = atoms.get_potential_energy() #excitation energy

    """

    occ = paw.occupations

    if occ.kT == 0:
        occ.kT = 1e-6
    if isinstance(occ, OccupationsDSCF):
        paw.occupations.orbitals = orbitals
    else:
        new_occ = OccupationsDSCF(occ.ne, occ.nspins, occ.kT, orbitals, paw)
        new_occ.set_communicator(occ.kpt_comm)
        paw.occupations = new_occ

    # if the calculator has already converged (for the ground state),
    # reset self-consistency and let the density be updated right away
    if paw.scf.converged:
        paw.scf.niter_fixdensity = 0
        paw.scf.reset()

class OccupationsDSCF(FermiDirac):
    """Occupation class.

    Corresponds to the ordinary FermiDirac class in occupation.py. Only
    difference is that it forces some electrons in the supplied orbitals
    in stead of placing all the electrons by a Fermi-Dirac distribution.
    """

    def __init__(self, ne, nspins, kT, orbitals, paw):
        FermiDirac.__init__(self, ne, nspins, kT)
        
        self.orbitals = orbitals
        self.norbitals = len(self.orbitals)

        self.cnoe = 0.
        for orb in self.orbitals:
            self.cnoe += orb[0]
        self.ne -= self.cnoe

    def calculate_band_energy(self, kpt_u):
        # Sum up all eigenvalues weighted with occupation numbers
        Eband = 0.0
        for kpt in kpt_u:
            Eband += np.dot(kpt.f_n, kpt.eps_n)

            if hasattr(kpt, 'c_on'):
                for c_n in kpt.c_on:
                    Eband += np.dot(np.abs(c_n)**2, kpt.eps_n)

        self.Eband = self.kpt_comm.sum(Eband)

    def calculate(self, kpts):
        FermiDirac.calculate(self, kpts)

        # Get the expansion coefficients c_un for each dscf-orbital
        # and incorporate their respective occupations into kpt.c_on
        c_oun = []
        for orb in self.orbitals:
            c_oun.append(orb[1].expand(self.epsF, kpts))

        for u, kpt in enumerate(kpts):
            kpt.c_on = np.zeros((self.norbitals,len(kpt.f_n)), np.complex)

            for o, orb in enumerate(self.orbitals):
                kpt.c_on[o,:] = orb[0]**0.5 * c_oun[o][u]

                if self.nspins == 2:
                    assert orb[2] in range(2), 'Invalid spin index'

                    if orb[2] == kpt.s:
                        kpt.c_on[o,:] *= kpt.weight**0.5
                    else:
                        kpt.c_on[o,:] = 0.0
                else:
                    kpt.c_on[o,:] = (0.5*kpt.weight)**0.5

        self.calculate_band_energy(kpts)
        
        # Correct the magnetic moment
        for orb in self.orbitals:
            if orb[2] == 0:
                self.magmom += orb[0]
            elif orb[2] == 1:
                self.magmom -= orb[0]
        
class MolecularOrbital:
    """Class defining the orbitals that should be filled in a dSCF calculation.
    
    An orbital is defined through a linear combination of the atomic
    partial waves. In each self-consistent cycle the method expand
    is called. This method take the Kohn-Sham orbitals fulfilling the
    criteria given by Estart, Eend and nos and return the best
    possible expansion of the orbital in this basis. The integral
    of the Kohn-Sham all-electron wavefunction ``|u,n>`` (u being local spin
    and kpoint index) and the partial wave ``|\phi_i^a>`` is appoximated
    by::

      wfs.kpt_u[u].P_ani = <\tilde p_i^a|\tilde\psi_{un}>.
    
    Parameters
    ----------
    paw: gpaw calculator instance
        The calculator used in the dSCF calculation.
    molecule: list of integers
        The atoms, which are a part of the molecule.
    Estart: float
        Kohn-Sham orbitals with an energy above Efermi+Estart are used
        in the linear expansion.
    Eend: float
        Kohn-Sham orbitals with an energy below Efermi+Eend are used
        in the linear expansion.
    nos: int
        The maximum Number Of States used in the linear expansion.
    w: list
        The weights of the atomic projector functions corresponding to
        a linear combination of atomic orbitals.
        Format::

          [[weight of 1. projector function of the 1. atom,
            weight of 2. projector function of the 1. atom, ...],
           [weight of 1. projector function of the 2. atom,
            weight of 2. projector function of the 2. atom, ...],
           ...]
    """

    def __init__(self, paw, molecule=[0,1], Estart=0.0, Eend=1.e6,
                 nos=None, w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]]):

        self.nspins = paw.wfs.nspins
        self.fixmom = paw.input_parameters.fixmom
        self.mol = molecule
        self.w = w
        self.Estart = Estart
        self.Eend = Eend
        self.nos = nos

    def expand(self, epsF, kpts):

        if self.nspins == 1:
            epsF = [epsF]
        elif not self.fixmom:
            epsF = [epsF, epsF]
            
        if self.nos == None:
            self.nos = len(kpts[0].f_n)

        # Get P_uni from the relevent nuclei
        P_auni = [[kpt.P_ani[a] for kpt in kpts] for a in self.mol]

        c_un = []
        for u, kpt in enumerate(kpts):
            Porb_n = np.zeros(len(kpt.f_n), np.complex)
            for atom in range(len(self.mol)):
                for pw_no in range(len(self.w[atom])):
                    Porb_n += (self.w[atom][pw_no] *
                               np.swapaxes(P_auni[atom][u], 0, 1)[pw_no])
            Porb_n = np.swapaxes(Porb_n, 0, 1)

            #print 'Kpt:', kpt.k, ' Spin:', kpt.s, \
            #      ' Sum_n|wi<pi|nks>|^2:', sum(abs(Porb_n)**2)/len(self.mol)

            c_n = np.zeros(len(kpt.f_n), np.complex)

            # Starting from KS orbitals with largest overlap,
            # fill in the expansion coeffients as <n|a> where
            # |n> is the n'th all-electron KS state for the
            # given k-point and |a> is the all-electron orbital.

            nos = 0
            bandpriority = np.argsort(abs(Porb_n)**2)[::-1]

            for n in bandpriority:
                if (kpt.eps_n[n] > epsF[kpt.s] + self.Estart and
                    kpt.eps_n[n] < epsF[kpt.s] + self.Eend):
                    c_n[n] = Porb_n[n].conj() #XXX BAD
                    nos += 1
                if nos == self.nos:
                    break

            c_n /= np.sqrt(sum(abs(c_n)**2))
            c_un.append(c_n)
        return c_un
                    
class AEOrbital:
    """Class defining the orbitals that should be filled in a dSCF calculation.
    
    An orbital is defined through a linear combination of KS orbitals
    which is determined by this class as follows: For each kpoint and spin
    we calculate the quantity ``c_n = <n|a>`` where ``|n>`` is the
    all-electron KS states in the calculation and ``|a>`` is the
    all-electron resonant state to be kept occupied.  We can then
    write ``|a> = Sum(c_n|n>)`` and in each self-consistent cycle the
    method expand is called. This method take the Kohn-Sham
    orbitals fulfilling the criteria given by Estart, Eend and
    nos (Number Of States) and return the best possible expansion of
    the orbital in this basis.

    Parameters
    ----------
    paw: gpaw calculator instance
        The calculator used in the dSCF calculation.
    molecule: list of integers
        The atoms, which are a part of the molecule.
    Estart: float
        Kohn-Sham orbitals with an energy above Efermi+Estart are used
        in the linear expansion.
    Eend: float
        Kohn-Sham orbitals with an energy below Efermi+Eend are used
        in the linear expansion.
    nos: int
        The maximum Number Of States used in the linear expansion.
    wf_u: list of wavefunction arrays
        Wavefunction to be occupied on the kpts on this processor.
    P_aui: list of two-dimensional arrays.
        [[kpt.P_ani[a,n,:] for kpt in kpts] for a in molecule]
        Projector overlaps with the wavefunction to be occupied for each
        kpoint. These are used when correcting to all-electron wavefunction
        overlaps.
    """

    def __init__(self, paw, wf_u, P_aui, Estart=0.0, Eend=1.e6,
                 molecule=[0,1], nos=None):
    
        self.nspins = paw.wfs.nspins
        self.fixmom = paw.input_parameters.fixmom
        self.gd = paw.wfs.gd
        self.dtype = paw.wfs.dtype
        self.nbands = paw.wfs.nbands
        self.setups = paw.wfs.setups

        self.wf_u = wf_u
        self.P_aui = P_aui
        self.Estart = Estart
        self.Eend = Eend
        self.mol = molecule
        self.nos = nos

    def expand(self, epsF, kpts):
        
        if self.nspins == 1:
            epsF = [epsF]
        elif not self.fixmom:
            epsF = [epsF, epsF]

        if self.nos == None:
            self.nos = len(kpts[0].f_n)
        
        # Check dimension of lists
        if len(self.wf_u) == len(kpts):
            wf_u = self.wf_u
            P_aui = self.P_aui
        else:
            raise RuntimeError('List of wavefunctions has wrong size')

        c_un = []
        for u, kpt in enumerate(kpts):

            # Inner product of pseudowavefunctions
            wf = np.reshape(wf_u[u], -1)
            Wf_n = kpt.psit_nG
            Wf_n = np.reshape(Wf_n, (len(kpt.f_n), -1))
            Porb_n = np.dot(Wf_n.conj(), wf) * self.gd.dv
            
            # Correction to obtain inner product of AE wavefunctions
            P_ani = [kpt.P_ani[a] for a in self.mol]
            for P_ni, a, b in zip(P_ani, self.mol, range(len(self.mol))):
                for n in range(self.nbands):
                    for i in range(len(P_ni[0])):
                        for j in range(len(P_ni[0])):
                            Porb_n[n] += (P_ni[n][i].conj() *
                                       self.setups[a].O_ii[i][j] *
                                       P_aui[b][u][j])

##             self.gd.comm.sum(Porb_n)

            print 'Kpt:', kpt.k, ' Spin:', kpt.s, \
                  ' Sum_n|<orb|nks>|^2:', sum(abs(Porb_n)**2)
            
            if self.dtype == float:
                c_n = np.zeros(len(kpt.f_n), np.float)
            else:
                c_n = np.zeros(len(kpt.f_n), np.complex)

            # Starting from KS orbitals with largest overlap,
            # fill in the expansion coeffients as <n|a> where
            # |n> is the n'th all-electron KS state for the
            # given k-point and |a> is the all-electron orbital.

            nos = 0
            bandpriority = np.argsort(abs(Porb_n)**2)[::-1]

            for n in bandpriority:
                if (kpt.eps_n[n] > epsF[kpt.s] + self.Estart and
                    kpt.eps_n[n] < epsF[kpt.s] + self.Eend):
                    c_n[n] = Porb_n[n] #XXX BAD - correct without .conj()
                    nos += 1
                if nos == self.nos:
                    break

            # Normalize expansion coefficients
            c_n /= np.sqrt(sum(abs(c_n)**2))
            
            c_un.append(c_n)
            
        return c_un
