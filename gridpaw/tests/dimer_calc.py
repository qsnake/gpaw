from gridpaw import Calculator
from ASE import ListOfAtoms, Atom
from elements import elements


def dimer_calc(symbol, xc='LDA', gpoints=28, a = 7.0, n=10):

    """Dimer_calc function
    Calculate the bond length of any given dimer for different bond lengths:
    Symbol is the atomic symbol, xc is the exchange correlation function
    (default is LDA), gpoints is the number of grid points in the cell in each
    direction (default is 28), a is the length of the cubic cell (default is 7
    aangstroem) and n is the number of the variation of the bond length
    (default is 10).
    """

    mag = elements[symbol]

    # calculation of the atom
    atom = ListOfAtoms([Atom(symbol, (a/2, a/2, a/2), magmom=mag[0])],
                       cell=(a, a, a))

    # gridpaw calculator:
    calc = Calculator(gpts=(gpoints, gpoints, gpoints), xc=xc,
                      out="%s-%s.out" % (symbol, xc))

    atom.SetCalculator(calc)
    e1 = atom.GetPotentialEnergy()

    # calculation of the dimer
    molecule = ListOfAtoms([Atom(symbol, (a/2 - mag[2]/2, a/2, a/2),
                                 magmom=mag[1]),
                            Atom(symbol, (a/2 + mag[2]/2, a/2, a/2),
                                 magmom=mag[1])],
                            cell = (a, a, a))

    calc = Calculator(gpts=(gpoints, gpoints, gpoints), xc=xc,
                      out="%s-dimer-%s.out" % (symbol, xc))
    molecule.SetCalculator(calc)
   
    energies = []
    bondlengths = []

    m = ( n - 1 )/2.0
    f = int(m)
  
    if m > f:
        i = 1
    else:
        i = 0
   
    for j in range(-f, f + 1 + i):
        k = mag[2] + 1 * j / 100.0

        molecule.SetCartesianPositions([(a/2 - k/2, a/2, a/2),
                                        (a/2 + k/2, a/2, a/2)])
        e2 = molecule.GetPotentialEnergy()
        e3 = (2 * e1 - e2)
        
        bondlengths.append(k)
        energies.append(e3)
       
    return bondlengths, energies

 
if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
    
    dimer_calc(symbol)




