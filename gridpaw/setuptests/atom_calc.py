from gridpaw import Calculator
from ASE import ListOfAtoms, Atom
import Numeric as num
from elements import elements
import sys


def atom_calc(symbol, xc='LDA', gpoints=28, L=5.0):

    """Eggbox test.
    
    Calculate the total energy of the atom, when varying its position
    in the cell with respect to the grid points. ``symbol`` is the atomic
    symbol name, ``xc`` is the exchange-correlation functional, gpoints is the
    number of gridpoints in each direction and ``L`` is the size of the unit
    cell. A list of 101 energies is returned."""

    energies = []

    mag = elements[symbol]

    atom = ListOfAtoms([Atom(symbol, (0, 0, 0), magmom=mag[0])],
                       cell=(L, L, L), periodic=True)

    calc = Calculator(gpts=(gpoints, gpoints, gpoints), xc=xc,
                      out="%s-eggbox-%s.out" % (symbol, xc))
    atom.SetCalculator(calc)

    h = L / gpoints

    for j in range(101):
        k = j * h / 100.0

        atom.SetCartesianPositions([(k, 0, 0)])

        e1 = atom.GetPotentialEnergy()

        energies.append(e1)

    return energies





