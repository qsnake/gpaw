from bulk_lattice import bulk_lattice
import Numeric as num
import sys


def bulk_calc(symbol, xc='LDA', gpoints=28, kpt=6, crys='fcc', g=10, a=None):
    
    """Calculate the cohesive energies at a varying lattice constants.

    Find the dependence of the cohesive energy with the lattice constant:
    ``symbol`` is the atomic symbol name, ``xc`` is the exchange-correlation
    functional, gpoints is the number of gridpoints in each direction, ``kpt``
    is the number of k points in each direction, ``crys`` is the crystal
    structure, ``g`` is the variation total of the lattice constant ``a``. A
    list of g number of energies and the lattice constants are returned.
    """

    L = 5 + a
    N = int(gpoints / a * L / 4) * 4

    energies = []
    lattice_constants = []
    
    for i in range(g):
        y = a + a * 0.2 * (i - 0.5 * g) / g
        c = bulk_lattice(symbol, xc, gpoints, kpt, y, L, N, crys)
        energies.append(c)
        lattice_constants.append(y)

    return energies, lattice_constants
