from math import pi, sqrt
from ASE.Utilities.DOS import DOS

import Numeric as num


class LDOS(DOS):
    def __init__(self, calc, width=None, window=None, npts=201):
        """Electronic Local Density Of States object.

        'calc' is a gpaw calculator instance.
        'width' is the width of the "delta-functions". Defaults to the electronic temperature of the calculation.
        'window' is the energy window. Default is from the lowest to the highest eigenvalue.
        'npts' is the number of energy points.
        """
        DOS.__init__(self, calc, width, window, npts)
        self.P_auni = [nucleus.P_uni for nucleus in calc.paw.nuclei]

    def Delta(self, energy):
        """Return a delta-function centered at 'energy'."""
        x = -((self.energies - energy) / self.width)**2
        x = num.clip(x, -100.0, 100.0)
        return num.exp(x) / (sqrt(pi) * self.width)

    def GetLDOS(self, a, spin=None):
        """Get the DOS projected onto the projector functions of atom a."""
        
        if spin is None:
            if self.nspins == 2:
                # Spin-polarized calculation, but no spin specified -
                # return the total DOS:
                return self.GetLDOS(a, spin=0) + self.GetLDOS(a, spin=1)
            else:
                spin = 0

        nk = len(self.w_k)
        P_kni = self.P_auni[a][spin * nk : (spin + 1) * nk]
        ni = len(P_kni[0, 0])
        dos_ie = num.zeros((ni, self.npts), num.Float)
        for w, P_ni, e_n in zip(self.w_k, P_kni, self.e_skn[spin]):
            for P_i, e in zip(P_ni, e_n):
                for i, P in enumerate(P_i):
                    dos_ie[i] += w * abs(P)**2 * self.Delta(e)
        return dos_ie
