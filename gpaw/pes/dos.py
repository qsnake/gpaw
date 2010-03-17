"""Photoelectron spectra from the (shifted) DOS approach.

"""
from ase import Hartree
from gpaw.pes import BasePES

class DOSPES(BasePES):
    """PES derived from density of states with shifted KS-energies.

    """
    def __init__(self, mother, daughter, shift=True):
        self.c_m = mother
        self.c_d = daughter
        self.f = None
        self.be = None
        self.shift = shift
        self.first_peak_energy = 0.

    def _calculate(self, f_min=0.9):
        """Evaluate energies and spectroscopic factors."""

        self.be = []
        self.f = []
        ex_m = []
        for kpt in self.c_m.wfs.kpt_u:
            for e, f in zip(kpt.eps_n, kpt.f_n):
                if f > f_min:
                    self.be.append(- e * Hartree)
                    self.f.append(f)
                    ex_m.append(- e * Hartree)

        if self.shift is True:
            ex_m.sort()
            e_m = self.c_m.get_potential_energy()
            e_d = self.c_d.get_potential_energy()
            energy_shift = e_d - e_m - ex_m[-1]
        else:
            energy_shift = self.shift

        self.be += energy_shift
