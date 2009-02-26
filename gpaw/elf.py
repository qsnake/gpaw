# -*- coding: utf-8 -*-

"""This module defines an ELF class."""

from numpy import pi
from ase.units import Bohr, Hartree
from gpaw.operators import Gradient

def _elf(nt, nt_grad2, taut, ncut, spinpol, elf):
    """Pseudo electron localisation function (ELF) as defined in
    Becke and Edgecombe, J. Chem. Phys., vol 92 (1990) 5397

    Arguments:
     =============== =====================================================
     ``nt``          Pseudo valence density.
     ``nt_grad2``    Squared norm of the density gradient.
     ``tau``         Kinetic energy density.
     ``ncut``        Minimum density cutoff parameter.
     ``spinpol``     Boolean indicator for spin polarization.
     ``elf``         Empty grid to storing ELF values in.
     =============== =====================================================
    """

    # Uniform electron gas value of D, Becke eq. (13). TODO! explain 3/10 not 3/5!!!
    if spinpol:
        D0 = 3.0/10.0*(6*pi**2)**(2.0/3.0)*nt**(5.0/3.0)
    else:
        D0 = 3.0/10.0*(3*pi**2)**(2.0/3.0)*nt**(5.0/3.0)

    # Note: The definition of tau in Becke eq. (9) misses the
    # factor 1/2, hence it is twice that of the GPAW implementation.

    #TODO! Weizsacker correction factor - extra 0.5 factor appended!!!
    D = taut - 0.125*nt_grad2/nt*0.5

    elf[:] = 1.0/(1.0+(D/D0)**2.0)

    if ncut is not None:
        elf[nt<ncut] = 0.0

    return None

class ELF:
    """ELF object for calculating the electronic localization function.
    
    Arguments:
     =============== =====================================================
     ``paw``         Instance of ``GPAW`` class.
     ``ncut``        Density cutoff below which the ELF is zero.
     =============== =====================================================
    """

    def __init__(self, paw=None, ncut=1e-6):
        """Create the ELF object."""

        self.gd = paw.gd
        self.finegd = paw.finegd
        self.nspins = paw.occupations.nspins
        self.density = paw.density

        self.ncut = ncut
        self.spinpol = (self.nspins == 2)

        self.initialize(paw)

    def initialize(self, paw):

        if not paw.initialized:
            raise RuntimeError('PAW instance is not initialized')

        if not hasattr(self.density,'taut_sG'):
            self.density.initialize_kinetic(paw.atoms)

        self.nt_grad2_sG = self.gd.empty(self.nspins)
        self.nt_grad2_sg = None

    def interpolate(self):

        self.density.interpolate()
        self.density.interpolate_kinetic()

        if self.nt_grad2_sg is None:
            self.nt_grad2_sg = self.finegd.empty(self.nspins)

        # Transfer the density from the coarse to the fine grid
        for s in range(self.nspins):
            self.density.interpolater.apply(self.nt_grad2_sG[s],
                                            self.nt_grad2_sg[s])

    def update(self, wfs):

        self.density.update_kinetic(wfs)

        self.nt_grad2_sG[:] = 0.0

        ddr = [Gradient(self.gd, c).apply for c in range(3)]      
        d_G = self.gd.empty()

        for s in range(self.nspins):
            for c in range(3):
                ddr[c](self.density.nt_sG[s], d_G)
                self.nt_grad2_sG[s] += d_G**2.0

        #TODO are nct from setups usable for nt_grad2_sG ?

    def get_electronic_localization_function(self, spin=0, gridrefinement=1):

        # Returns dimensionless electronic localization function
        if gridrefinement == 1:
            elf_G = self.gd.empty()
            _elf(self.density.nt_sG[spin], self.nt_grad2_sG[spin],
                 self.density.taut_sG[spin], self.ncut, self.spinpol, elf_G)
            return elf_G
        elif gridrefinement == 2:
            if self.nt_grad2_sg is None:
                self.interpolate()

            elf_g = self.finegd.empty()
            _elf(self.density.nt_sg[spin], self.nt_grad2_sg[spin],
                 self.density.taut_sg[spin], self.ncut, self.spinpol, elf_g)
            return elf_g
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

    def get_kinetic_energy_density(self, spin=0, gridrefinement=1):

        # Returns kinetic energy density in eV / Ang^3
        if gridrefinement == 1:
            return self.density.taut_sG[spin] / (Hartree / Bohr**3.0)
        elif gridrefinement == 2:
            if self.density.taut_sg is None:
                self.density.interpolate_kinetic()

            return self.density.taut_sg[spin] / (Hartree / Bohr**3.0)
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

