from math import pi, sqrt
from ASE.Utilities.DOS import DOS

import Numeric as num

def print_projectors(nucleus):
    n_j = nucleus.setup.n_j
    l_j = nucleus.setup.l_j
    angular = [['1'],
               ['y', 'z', 'x'],
               ['xy', 'yz', '3z^2-r^2', 'xz', 'x^2-y^2'],
               ['3x^2y-y^3', 'xyz', '5yz^2-yr^2', '5z^3-3zr^2',
                '5xz^2-xr^2', 'x^2z-y^2z', 'x^3-3xy^2'],
               ]
    print ' i n l m'
    print '--------'
    i = 0
    for n, l in zip(n_j, l_j):
        for m in range(2*l+1):
            if n == -1:
                n = '*'
            print '%2s %s %s_%s' % (i, n, 'spdf'[l], angular[l][m])
            i += 1


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
        
        if hasattr(a, '__iter__'):
            # a is a list of atom indicies -
            # sum all angular chanels and add indicated atoms:
            dos_e = num.zeros(self.npts, num.Float)
            for i in a:
                dos_e += num.sum(self.GetLDOS(i, spin))
            return dos_e

        nk = len(self.w_k)
        P_kni = self.P_auni[a][spin * nk : (spin + 1) * nk]
        ni = len(P_kni[0, 0])
        dos_ie = num.zeros((ni, self.npts), num.Float)
        for w, P_ni, e_n in zip(self.w_k, P_kni, self.e_skn[spin]):
            for P_i, e in zip(P_ni, e_n):
                for i, P in enumerate(P_i):
                    dos_ie[i] += w * abs(P)**2 * self.Delta(e)
        return dos_ie
