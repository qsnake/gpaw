"""Photoelectron spectra from the (shifted) DOS approach.

"""
from ase import Hartree
from gpaw.pes import BasePES

class DOSPES(BasePES):
    """PES derived from density of states with shifted KS-energies.

    """
    def __init__(self, mother, daughter):
        self.c_m = mother
        self.c_d = daughter
        self.f = None
        self.be = None
        self.first_peak_energy = None

    def _calculate(self):
        """Evaluate energies and spectroscopic factors."""
        e_m = self.c_m.get_potential_energy()
        e_d = self.c_d.get_potential_energy()

        if self.first_peak_energy == None:
            self.first_peak_energy = e_d - e_m

        ex_m = []
        for kpt in self.c_m.wfs.kpt_u:
            for e, f in zip(kpt.eps_n, kpt.f_n):
                # XXX use occupation numbers
                ex_m += [e * Hartree] * int(f + .5)

        ex_m.sort()
        self.be = []
        for j in range(len(ex_m)):
            self.be.append(-ex_m[j] + ex_m[-1] + self.first_peak_energy)

        # XXX use occupation numbers
        self.f = [1] * len(self.be)
