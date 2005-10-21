from gridpaw import Calculator
from ASE import ListOfAtoms, Atom
import Numeric as num
from elements import elements


def atom_calc(symbol, xc='LDA', gpoints=28, a=5.0):

    """Atom_calc function
    Calculate the total energy of the atom, when varying its position in the
    cell with respect to the grid points.
    """

    # calculation of the atom

    total_energy = []
    atom_position = []

    m1 = elements[symbol]

    atom = ListOfAtoms([Atom(symbol, (0, 0, 0), magmom=m1[0])],
                       cell=(a, a, a), periodic=True)

    calc = Calculator(gpts=(gpoints, gpoints, gpoints), xc=xc,
                      out="%s-%s.out" % (symbol, xc))
    atom.SetCalculator(calc)

    h = a / gpoints

    for j in range(101):
        k = j * h / 100.0

        atom.SetCartesianPositions([ (k, 0, 0) ])

        e1 = atom.GetPotentialEnergy()

        total_energy.append(e1)

    return total_energy


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
    
    atom_calc(symbol)




