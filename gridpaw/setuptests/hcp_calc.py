from gridpaw import Calculator
from ASE.Visualization.RasMol import RasMol
from ASE import ListOfAtoms, Atom
from elements import elements
from math import sqrt
import Numeric as num
import sys


def hcp_calc(symbol, xc='LDA', gpoints=None, kpt=6, g=10, a=None, cov=None):

    """Calculate the cohesive energy.
    
    ``Symbol`` is the atomic symbol name, ``xc`` is the exchange-correlation
    functional, ``gpoints`` is the number of grid points in the cell in each
    direction, ``kpt`` is the number of k-points in each direction and ``g`` is
    the number of variation of the lattice constants ``a`` and ``cov`` (c/a).
    """

    mag = elements[symbol]
    lattice_constants = []
    covera = []
    coh_energy = num.zeros((g, g), num.Float)
    k = 0
    
    for i in range(g):
    
        q = cov + cov * 0.1 * (i - 0.5 * g) / g
        covera.append(q)
        h = 0
    
        for j in range(g):

            y = a + a * 0.1 * (j - 0.5 * g) / g
            lattice_constants.append(y)
            L = 5 + y
            N = int(gpoints[0] / y * L / 4) * 4

            # calculation of the atom
            atom = ListOfAtoms([Atom(symbol, ( L/2, L/2, L/2),
                                     magmom=mag[0])], cell=(L, L, L))

            calca = Calculator(gpts=(N, N, N), xc=xc,
                               out="%s-hcp-atom-%s.out" % (symbol, xc))

            atom.SetCalculator(calca)
            e1 = atom.GetPotentialEnergy()

            # calculation of the bulk cell
            bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0),
                                     magmom=mag[1]),
                                Atom(symbol, ( 1/2., 1/2.,  0.0),
                                     magmom=mag[1]),
                                Atom(symbol, (  0.0, 1/4., 1/2.),
                                     magmom=mag[1]),
                                Atom(symbol, ( 1/2., 3/4., 1/2.),
                                     magmom=mag[1])],
                               periodic=True)

            cell = [ y, sqrt(3.)*y , q*y]
            bulk.SetUnitCell(cell)

            calc = Calculator(gpts=gpoints, kpts=(kpt, kpt-2, kpt-2), xc=xc,
                              out="%s-hcp-bulk-%s.out" % (symbol, xc))

            bulk.SetCalculator(calc)
            e2 = bulk.GetPotentialEnergy()
            coh_energy[k, h] = (e2 - 4*e1)/4

            h += 1
        k += 1
        
    return coh_energy, lattice_constants, covera





