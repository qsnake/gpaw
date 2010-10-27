# -*- coding: utf-8 -*-

"""This module defines an ELF class."""

from numpy import pi
from ase.units import Bohr, Hartree
from gpaw.fd_operators import Gradient
from gpaw.lfc import LocalizedFunctionsCollection as LFC

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

        self.gd = paw.wfs.gd
        self.finegd = paw.density.finegd
        self.nspins = paw.density.nspins
        self.density = paw.density

        self.ncut = ncut
        self.spinpol = (self.nspins == 2)

        self.initialize(paw)

    def initialize(self, paw):

        if not paw.initialized:
            raise RuntimeError('PAW instance is not initialized')

        self.tauct = LFC(self.gd,
                         [[setup.tauct] for setup in self.density.setups],
                         forces=True, cut=True)
        spos_ac = paw.atoms.get_scaled_positions() % 1.0
        self.tauct.set_positions(spos_ac)

        self.taut_sG = self.gd.empty(self.nspins)
        self.taut_sg = None
        self.nt_grad2_sG = self.gd.empty(self.nspins)
        self.nt_grad2_sg = None

    def interpolate(self):

        self.density.interpolate()

        if self.taut_sg is None:
            self.taut_sg = self.finegd.empty(self.nspins)
            self.nt_grad2_sg = self.finegd.empty(self.nspins)

        # Transfer the densities from the coarse to the fine grid
        for s in range(self.nspins):
            self.density.interpolator.apply(self.taut_sG[s],
                                            self.taut_sg[s])
            self.density.interpolator.apply(self.nt_grad2_sG[s],
                                            self.nt_grad2_sg[s])

    def update(self, wfs):
        ddr_v = [Gradient(self.gd, v).apply for v in range(3)]
        assert self.nspins == 1
        self.taut_sG[:] = wfs.calculate_kinetic_energy_density(
            self.taut_sG[:1], ddr_v)

        # Add the pseudo core kinetic array
        self.tauct.add(self.taut_sG[0])

        # For periodic boundary conditions
        if wfs.symmetry is not None:
            wfs.symmetry.symmetrize(self.taut_sG[0], wfs.gd)

        self.nt_grad2_sG[:] = 0.0

        d_G = self.gd.empty()

        for s in range(self.nspins):
            for v in range(3):
                ddr_v[v](self.density.nt_sG[s], d_G)
                self.nt_grad2_sG[s] += d_G**2.0

        #TODO are nct from setups usable for nt_grad2_sG ?

    def get_electronic_localization_function(self, spin=0, gridrefinement=1,
                                             pad=True, broadcast=True):

        # Returns dimensionless electronic localization function
        if gridrefinement == 1:
            elf_G = self.gd.empty()
            _elf(self.density.nt_sG[spin], self.nt_grad2_sG[spin],
                 self.taut_sG[spin], self.ncut, self.spinpol, elf_G)
            elf_G = self.gd.collect(elf_G, broadcast)
            if pad:
                elf_G = self.gd.zero_pad(elf_G)
            return elf_G
        elif gridrefinement == 2:
            if self.nt_grad2_sg is None:
                self.interpolate()

            elf_g = self.finegd.empty()
            _elf(self.density.nt_sg[spin], self.nt_grad2_sg[spin],
                 self.taut_sg[spin], self.ncut, self.spinpol, elf_g)
            elf_g = self.finegd.collect(elf_g, broadcast)
            if pad:
                elf_g = self.finegd.zero_pad(elf_g)
            return elf_g
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

    def get_kinetic_energy_density(self, spin=0, gridrefinement=1):

        # Returns kinetic energy density in eV / Ang^3
        if gridrefinement == 1:
            return self.taut_sG[spin] / (Hartree / Bohr**3.0)
        elif gridrefinement == 2:
            if self.taut_sg is None:
                self.density.interpolator.apply(self.taut_sG[spin],
                                                    self.taut_sg[spin])

            return self.taut_sg[spin] / (Hartree / Bohr**3.0)
        else:
            raise NotImplementedError('Arbitrary refinement not implemented')

