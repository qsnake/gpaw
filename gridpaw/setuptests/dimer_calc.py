from gridpaw import Calculator
from ASE import ListOfAtoms, Atom
from elements import elements
import sys


def dimer_calc(symbol, xc='LDA', gpoints=28, L=None, n=10):

    """Dimer_calc function.
    
    Calculate the bond length of any given dimer for different bond lengths:
    ``Symbol`` is the atomic symbol name, ``xc`` is the exchange correlation
    functional, ``gpoints`` is the number of grid points in the cell in each
    direction, ``L`` is the size of the unit cell and ``n`` is number of the
    variation of the bond length. A list of n number of energies and the
    related bond lengths are returned.
    """

    mag = elements[symbol]

    # calculation of the atom:
    atom = ListOfAtoms([Atom(symbol, (L/2, L/2, L/2), magmom=mag[0])],
                       cell=(L, L, L))

    # gridpaw calculator:
    calc = Calculator(gpts=(gpoints, gpoints, gpoints), xc=xc,
                      out="%s2-atom-%s.out" % (symbol, xc))

    atom.SetCalculator(calc)
    e1 = atom.GetPotentialEnergy()

    # calculation of the dimer:
    molecule = ListOfAtoms([Atom(symbol, (L/2 - mag[2]/2, L/2, L/2),
                                 magmom=mag[1]),
                            Atom(symbol, (L/2 + mag[2]/2, L/2, L/2),
                                 magmom=mag[1])],
                            cell = (L, L, L))

    calc = Calculator(gpts=(gpoints, gpoints, gpoints), xc=xc,
                      out="%s-dimer-%s.out" % (symbol, xc))
    molecule.SetCalculator(calc)
   
    energies = []
    bondlengths = []
   
    for j in range(n):
        k = mag[2] + 0.05 * (j - 0.5 * n) / n

        molecule.SetCartesianPositions([(L/2 - k/2, L/2, L/2),
                                        (L/2 + k/2, L/2, L/2)])
        e2 = molecule.GetPotentialEnergy()
        e3 = (2 * e1 - e2)
        
        bondlengths.append(k)
        energies.append(e3)
       
    return bondlengths, energies
