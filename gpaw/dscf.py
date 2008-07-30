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
import numpy as npy
from gpaw.occupations import ZeroKelvin, FermiDirac
import gpaw.mpi as mpi

def dscf_calculation(calc, orbitals, atoms=None):
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
    >>> dscf_calculation(calc, [[1.0,sigma_star,1]], atoms)
    >>> e_exc=atoms.get_potential_energy() #excitation energy

    """

    # if the calculator has not been initialized it does not have an
    # occupation object
    if not hasattr(calc, 'occupation'):
       calc.initialize(atoms)
    occ = calc.occupation
    if isinstance(occ, FermiDirac) and not isinstance(occ, FermiDiracDSCF):
        n_occ = FermiDiracDSCF(occ.ne, occ.nspins, occ.kT, orbitals, calc)
        n_occ.set_communicator(occ.kpt_comm)
        calc.occupation = n_occ
        calc.converged = False
    elif (isinstance(occ, ZeroKelvinDSCF) and
          not isinstance(occ, ZeroKelvinDSCF)):
        n_occ = ZeroKelvinDSCF(occ.ne, occ.nspins, orbitals ,calc)
        n_occ.set_communicator(occ.kpt_comm)
        calc.occupation = n_occ
        calc.converged = False
    else:
        calc.occupation.orbitals = orbitals 

        
class ZeroKelvinDSCF(ZeroKelvin):
    """ Occupation class

    Corresponds to the ordinary ZeroKelvin class in occupation.py. Only
    difference is that it forces some electrons in the supplied orbitals
    in stead of placing all the eletrons in the lowest lying states.
    """

    def __init__(self, ne, nspins, orbitals, paw):
        ZeroKelvin.__init__(self, ne, nspins)
        self.orbitals = orbitals
        self.paw=paw

        # sum up the total number of controlled electrons
        self.cnoe = 0.
        for orb in orbitals:
            self.cnoe += orb[0]
        self.ne -= self.cnoe

    def calculate_band_energy(self, kpt_u):
        # Sum up all eigenvalues weighted with occupation numbers:
        Eband = 0.0
        for kpt in kpt_u:
            Eband += npy.dot(kpt.f_n, kpt.eps_n)
            if hasattr(kpt, 'ft_omn'):
                for i in range(len(kpt.ft_omn)):
                    Eband += npy.dot(npy.diagonal(kpt.ft_omn[i]).real,
                                     kpt.eps_n)
        self.Eband = self.kpt_comm.sum(Eband)

    def calculate(self, kpts):
        # place all non-controlled electrons in the lowest lying states
        ZeroKelvin.calculate(self, kpts)
        
        # Estimate fermi energy, which is used when collecting the
        # KS-orbitals used in the linear expansion of orbitals
        epsF = 0.
        for kpt in kpts:
            ffi = npy.argsort(kpt.f_n)[0]
            epsF += 0.5 * (kpt.eps_n[ffi-1] + kpt.eps_n[ffi])
        self.epsF = self.kpt_comm.sum(epsF) / self.nspins

        # Get the expansion coefficients for the dscf-orbital(s)
        # and create the density matrices, kpt.ft_mn
        ft_okm = []
        for orb in self.orbitals:
            ft_okm.append(orb[1].get_ft_km(self.epsF))
            
        for kpt in self.paw.kpt_u:
            kpt.ft_omn = npy.zeros((len(self.orbitals),
                                    len(kpt.f_n), len(kpt.f_n)), npy.complex)
            for o in range(len(self.orbitals)):
                ft_m = ft_okm[o][kpt.u]
                for n1 in range(len(kpt.f_n)):
                     for n2 in range(len(kpt.f_n)):
                         kpt.ft_omn[o,n1,n2] = (self.orbitals[o][0] *
                                                ft_m[n1] *
                                                npy.conjugate(ft_m[n2]))

                if self.nspins == 2 and self.orbitals[o][2] == kpt.s:
                    kpt.ft_omn[o] *= kpt.weight
                elif self.nspins == 2 and self.orbitals[o][2] < 2:
                    kpt.ft_omn[o] *= 0.
                else:
                    kpt.ft_omn[o] *= 0.5 * kpt.weight

        self.calculate_band_energy(kpts)
        

class FermiDiracDSCF(FermiDirac):
    """Occupation class.

    Corresponds to the ordinary FermiDirac class in occupation.py. Only
    difference is that it forces some electrons in the supplied orbitals
    in stead of placing all the electrons by a Fermi-Dirac distribution.
    """

    def __init__(self, ne, nspins, kT, orbitals, paw):
        FermiDirac.__init__(self, ne, nspins, kT)
        self.orbitals = orbitals
        self.paw=paw

        self.cnoe = 0.
        for orb in orbitals:
            self.cnoe += orb[0]
        self.ne -= self.cnoe

    def calculate_band_energy(self, kpt_u):
        # Sum up all eigenvalues weighted with occupation numbers:
        Eband = 0.0
        for kpt in kpt_u:
            Eband += npy.dot(kpt.f_n, kpt.eps_n)
            if hasattr(kpt, 'ft_omn'):
                for i in range(len(kpt.ft_omn)):
                    Eband += npy.dot(npy.diagonal(kpt.ft_omn[i]).real,
                                     kpt.eps_n)
        self.Eband = self.kpt_comm.sum(Eband)

    def calculate(self, kpts):

        if self.epsF is None:
            # Fermi level not set.  Make a good guess:
            self.guess_fermi_level(kpts)

        # Now find the correct Fermi level for the non-controlled electrons
        self.find_fermi_level(kpts)

        # Get the expansion coefficients for the dscf-orbital(s)
        # and create the density matrices, kpt.ft_mn
        ft_okm = []
        for orb in self.orbitals:
            ft_okm.append(orb[1].get_ft_km(self.epsF))
            
        for kpt in self.paw.kpt_u:
            kpt.ft_omn = npy.zeros((len(self.orbitals),
                                    len(kpt.f_n), len(kpt.f_n)), npy.complex)
            for o in range(len(self.orbitals)):
                ft_m = ft_okm[o][kpt.u]
                for n1 in range(len(kpt.f_n)):
                     for n2 in range(len(kpt.f_n)):
                         kpt.ft_omn[o,n1,n2] = (self.orbitals[o][0] *
                                                ft_m[n1] *
                                                npy.conjugate(ft_m[n2]))

                if self.nspins == 2 and self.orbitals[o][2] == kpt.s:
                    kpt.ft_omn[o] *= kpt.weight
                elif self.nspins == 2 and self.orbitals[o][2] < 2:
                    kpt.ft_omn[o] *= 0.
                else:
                    kpt.ft_omn[o] *= 0.5 * kpt.weight

        S = 0.0
        for kpt in kpts:
            if self.fixmom:
                x = npy.clip((kpt.eps_n - self.epsF[kpt.s]) / self.kT,
                             -100.0, 100.0)
            else:
                x = npy.clip((kpt.eps_n - self.epsF) / self.kT, -100.0, 100.0)
            y = npy.exp(x)
            z = y + 1.0
            y *= x
            y /= z
            y -= npy.log(z)
            S -= kpt.weight * npy.sum(y)

        self.S = self.kpt_comm.sum(S) * self.kT
        self.calculate_band_energy(kpts)



class MolecularOrbitals:
    """Class defining the orbitals that should be filled in a dSCF calculation.
    
    An orbital is defined through a linear combination of the atomic
    projector functions. In each self-consistent cycle the method get_ft_km
    is called. This method take the Kohn-Sham orbittals forfilling the
    criteria given by Estart, Eend and no_of_states and return the best
    possible expansion of the orbital in this basis.

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
    no_of_states: int
        The maximum number of Kohn-Sham orbitals used in the linear expansion.
    w: list
        The weights of the atomic projector functions.
        Format::

          [[weight of 1. projector function of the 1. atom,
            weight of 2. projector function of the 1. atom, ...],
           [weight of 1. projector function of the 2. atom,
            weight of 2. projector function of the 2. atom, ...],
           ...]
    """

    def __init__(self, paw, molecule=[0,1], Estart=0.0, Eend=1.e6,
                 no_of_states=None,w=[[1.,0.,0.,0.],[-1.,0.,0.,0.]]):
        self.paw = paw
        self.mol = molecule

        self.w = w
        self.Estart = Estart
        self.Eend = Eend
        self.nos = no_of_states

    def get_ft_km(self, epsF):

        # get P_uni from the relevent nuclei
        P_auni = []
        for atom_no in self.mol:
            nucleus = self.paw.nuclei[atom_no]

            no_waves = npy.zeros(1, npy.int)
            if nucleus.in_this_domain:
                no_waves[0] = nucleus.get_number_of_partial_waves()
            self.paw.domain.comm.broadcast(no_waves, nucleus.rank)

            if nucleus.in_this_domain:
                P_uni = nucleus.P_uni
            else:
                shape = (self.paw.nmyu, self.paw.nbands, no_waves[0])
                if self.paw.dtype == float:
                    P_uni = npy.zeros(shape, npy.float)
                else:
                    P_uni = npy.zeros(shape, npy.complex)
            self.paw.domain.comm.broadcast(P_uni, nucleus.rank)

            P_auni.append(P_uni)

        if self.paw.nspins == 1:
            epsF = [epsF]
        elif not self.paw.fixmom:
            epsF = [epsF,epsF]
        if self.nos == None:
            self.nos = len(self.paw.kpt_u[0].f_n)
            
        ft_km = []
        for kpt in self.paw.kpt_u:
            Porb_n = npy.zeros(npy.swapaxes(P_auni[0][0],0,1)[0].shape,
                                   npy.complex)
            for nuc in range(len(self.mol)):
                for pw_no in range(len(self.w[nuc])):
                    Porb_n += (self.w[nuc][pw_no] *
                               npy.swapaxes(P_auni[nuc][kpt.u],0,1)[pw_no])
            Porb_n = npy.swapaxes(Porb_n,0,1)

            Pabs_n = abs(Porb_n)**2
            argsort = npy.argsort(Pabs_n)

            assert(len(kpt.f_n) == len(Pabs_n))

            ft_m = npy.zeros(len(kpt.f_n), npy.complex)
            for m in argsort[::-1]:
                if self.hole:
                    if Pabs_n[m] < kpt.f_n[m] / kpt.weight:
                        ft_m[m] = npy.conjugate(Porb_n[m])
                else:
                    nosf = 0
                    if (kpt.eps_n[m] > epsF[kpt.s] + self.Estart and
                        kpt.eps_n[m] < epsF[kpt.s] + self.Eend):
                        ft_m[m] = Porb_n[m]
                        nosf += 1
                    if nosf == self.nos:
                        break

            ft_m /= npy.sqrt(sum(abs(ft_m)**2))

            ft_km.append(ft_m)

        return ft_km
                    
class WaveFunction:
    """Class defining the orbitals that should be filled in a dSCF calculation.
    
    An orbital is defined through a linear combination of KS orbitals
    which is determined by this class as follows: For each kpoint we
    calculate the quantity ``ft_m = <m|a>`` where ``|m>`` is the
    all-electron KS states in the calculation and ``|a>`` is the
    all-electron resonant state to be kept occupied.  We can then
    write ``|a> = Sum(ft_m|m>)`` and in each self-consistent cycle the
    method get_ft_km is called. This method take the Kohn-Sham
    orbitals fulfilling the criteria given by Estart, Eend and
    no_of_states and return the best possible expansion of the orbital
    in this basis.

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
    no_of_states: int
        The maximum number of Kohn-Sham orbitals used in the linear expansion.
    wf_u: list of wavefunction arrays
        Wavefunction to be occupied on the kpts on this processor.
    P_aui: list of two-dimensional arrays.
        Calulator.nuclei[a].P_uni[:,n,:] for a in molecule
        Projector overlaps with the wavefunction to be occupied for each
        kpoint. These are used when correcting to all-electron wavefunction
        overlaps. wf_u and P_uai can also be given as full lists
        corresponding to the all the kpoints in the calculation.
    """

    def __init__(self, paw, wf_u, P_aui, Estart=0.0, Eend=1.e6,
                 molecule=[0,1], no_of_states=None):
    
        self.paw = paw
        self.wf_u = wf_u
        self.P_aui = P_aui
        self.Estart = Estart
        self.Eend = Eend
        self.mol = molecule
        self.nos = no_of_states
        if no_of_states < 0:
            self.hole = True

    def get_ft_km(self, epsF):

        if self.paw.nspins == 1:
            epsF = [epsF]
        elif not self.paw.fixmom:
            epsF = [epsF, epsF]
        if self.nos == None:
            self.nos = len(self.paw.kpt_u[0].f_n)

        if len(self.wf_u) == len(self.paw.kpt_u):
            wf_u = self.wf_u
            P_aui = self.P_aui
        elif len(self.wf_u) == self.paw.nkpts * self.paw.nspins:
            wf_u = []
            P_uai = []
            for kpt in self.paw.kpt_u:
                k = kpt.s * self.paw.nkpts + kpt.k
                wf_u.append(self.wf_u[k])
                P_uai.append(self.P_uai[k])
            P_aui = npy.swapaxes(P_uai, 0, 1)
        else:
            raise RuntimeError('List of wavefunctions has wrong size')

        ft_km = []
        for kpt in self.paw.kpt_u:
            
            # Inner product of pseudowavefunctions
            wf = npy.reshape(wf_u[kpt.u], -1) * self.paw.a0**1.5
            if kpt.psit_nG is None:
                a,b,c = self.paw.gd.N_c
                kpt.psit_nG = npy.zeros((len(kpt.f_n), a, b, c),float)
            psit_nG = npy.reshape(kpt.psit_nG, (len(kpt.f_n), -1))
            dV = self.paw.gd.h_c[0] * self.paw.gd.h_c[1] * self.paw.gd.h_c[2]
            Porb_n = npy.dot(npy.conjugate(psit_nG), wf) * dV
            
            # Correction to obtain inner product of AE wavefunctions
            for n in range(self.paw.nbands):
                for a, b in zip(self.mol, range(len(self.mol))):
                    atom = self.paw.nuclei[a]
                    p_i = npy.conjugate(atom.P_uni[kpt.u][n])
                    for i in range(len(p_i)):
                        for j in range(len(p_i)):
                            Porb_n[n] += (p_i[i] * atom.setup.O_ii[i][j]
                                          * P_aui[b][kpt.u][j])
            
            Pabs_n = abs(Porb_n)**2
            argsort = npy.argsort(Pabs_n)

            print 'Kpoint', mpi.rank, kpt.u, kpt.k, sum(abs(Porb_n)**2)

            if self.paw.dtype == float:
                ft_m = npy.zeros(len(kpt.f_n), npy.float)
            else:
                ft_m = npy.zeros(len(kpt.f_n), npy.complex)

            for m in argsort[::-1]:
                if self.hole:
                    if Pabs_n[m] < kpt.f_n[m] / kpt.weight:
                        ft_m[m] = npy.conjugate(Porb_n[m])
                else:
                    nosf = 0
                    if (Pabs_n[m] > kpt.f_n[m] / kpt.weight and
                        kpt.eps_n[m] > epsF[kpt.s] + self.Estart and
                        kpt.eps_n[m] < epsF[kpt.s] + self.Eend):
                        ft_m[m] = npy.conjugate(Porb_n[m])
                    nosf += 1
                    if nosf == self.nos:
                        break

            ft_m /= npy.sqrt(sum(abs(ft_m)**2))
            ft_km.append(ft_m)
            
        return ft_km
                    


    


