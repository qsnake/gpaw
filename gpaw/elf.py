# -*- coding: utf-8 -*-

"""This module defines an ELF class."""

from numpy import pi
from ase.units import Bohr, Hartree
from gpaw.operators import Gradient

def _elf(nt, nt_grad2, taut, ncut, elf):

    D0 = 3.0/5.0*(6*pi**2)**(2.0/3.0)*nt**(5.0/3.0)
    D = 2.0*taut - 0.25*nt_grad2/nt

    elf[:] = 1.0/(1.0+(D/D0)**2.0)

    if ncut is not None:
        elf[nt<ncut] = 0.0

    return None

class ELF:
    """ELF object for calculating the electronic localization function.
    
    Arguments:
     =============== =====================================================
     ``calc``        Instance of ``GPAW`` class.
     ``ncut``        Density cutoff below which the ELF is zero.
     =============== =====================================================
    """

    def __init__(self, calc=None, ncut=1e-6):
        """Create the ELF object."""

        self.ncut = ncut
        self.initialized = False
        self.updated = False

        if calc is not None:
            self.set_calculator(calc)
            self.update(calc.kpt_u)

    def set_calculator(self, calc):
        self.gd = calc.gd
        self.finegd = calc.finegd

        self.nspins = calc.occupation.nspins

        self.density = calc.density

        self.initialized = True
        self.updated = False

        self.initialize_gradient_square()
        self.density.initialize_kinetic()

    def initialize_gradient_square(self):
        assert self.initialized, 'ELF instance is not initialized'

        self.nt_grad2_sG = self.gd.empty(self.nspins)
        self.nt_grad2_sg = self.finegd.empty(self.nspins)

    def update_gradient_square(self):
        assert self.initialized, 'ELF instance is not initialized'

        self.nt_grad2_sG[:] = 0.0

        ddr = [Gradient(self.gd, c).apply for c in range(3)]      
        d_G = self.gd.empty()

        for s in range(0,self.nspins):
            for c in range(0,3):
                ddr[c](self.density.nt_sG[s],d_G)
                self.nt_grad2_sG[s] += d_G**2.0

        # Transfer the density from the coarse to the fine grid
        for s in range(0,self.nspins):
            self.density.interpolate(self.nt_grad2_sG[s], self.nt_grad2_sg[s])

    def update(self, kpt_u):
        assert self.initialized, 'ELF instance is not initialized'

        self.update_gradient_square()

        # The kinetic energy density must be reset before updating
        self.density.taut_sG[:] = 0.0
        self.density.update_kinetic(kpt_u)

        self.updated = True

    def get_electronic_localization_function(self, spin=0, gridrefinement=1):
        assert self.initialized and self.updated, 'ELF instance is not initialized or updated'

        if gridrefinement==1:
            elf_G = self.gd.empty()
            _elf(self.density.nt_sG[spin], self.nt_grad2_sG[spin], 2.0/self.nspins*self.density.taut_sG[spin], self.ncut, elf_G)
            return elf_G
        elif gridrefinement==2:
            elf_g = self.finegd.empty()
            _elf(self.density.nt_sg[spin], self.nt_grad2_sg[spin], 2.0/self.nspins*self.density.taut_sg[spin], self.ncut, elf_g)
            return elf_g
        else:
            raise NotImplementedError, 'Arbitrary grid refinement is not implemented!'

    def get_kinetic_energy_density(self, spin=0, gridrefinement=1):
        assert self.initialized and self.updated, 'ELF instance is not initialized or updated'

        # Returns kinetic energy density in eV / Ang^3
        if gridrefinement==1:
            return self.density.taut_sG[spin] / (Hartree / Bohr**3.0)
        elif gridrefinement==2:
            return self.density.taut_sg[spin] / (Hartree / Bohr**3.0)
        else:
            raise NotImplementedError, 'Arbitrary grid refinement is not implemented!'

