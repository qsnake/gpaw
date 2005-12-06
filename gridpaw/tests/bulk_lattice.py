from gridpaw import Calculator
from ASE import ListOfAtoms, Atom
from elements import elements
from ASE.Visualization.RasMol import RasMol
import sys
 

def bulk_lattice(symbol, xc='LDA', gpoints=28, kpt=6, a=None, L=None, N=36,
                 crys='fcc'):

    """Calculate the cohesive energy.
    
    ``Symbol`` is the atomic symbol name, ``xc`` is the exchange correlation
    functional, ``gpoints`` is the number of grid points in the cell in each
    direction, ``kpt`` is the number of k-points in each direction, ``a`` is
    the lattice constant, ``L`` is the length of the atomic cell, ``N`` is the
    number of grid points in the atomic cell and ``crys`` is the crystal
    structure type. It returns the cohesive energy.
    """

    mag = elements[symbol]

    # calculation of the single atom:
    atom = ListOfAtoms([Atom(symbol, ( L/2, L/2, L/2), magmom=mag[0])],
                       cell=(L, L, L))

    calc = Calculator(gpts=(N, N, N), xc=xc,
                      out="%s-atom-%s.out" % (symbol, xc))

    atom.SetCalculator(calc)
    e1 = atom.GetPotentialEnergy()

    # calculation of the bulk cell:
    if crys == 'diamond':
        bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0), magmom=mag[1]),
                            Atom(symbol, ( 1/4., 1/4., 1/4.), magmom=mag[1]),
                            Atom(symbol, (  0.0, 1/2., 1/2.), magmom=mag[1]),
                            Atom(symbol, ( 1/4., 3/4., 3/4.), magmom=mag[1]),
                            Atom(symbol, ( 1/2.,  0.0, 1/2.), magmom=mag[1]),
                            Atom(symbol, ( 3/4., 1/4., 3/4.), magmom=mag[1]),
                            Atom(symbol, ( 1/2., 1/2.,  0.0), magmom=mag[1]),
                            Atom(symbol, ( 3/4., 3/4., 1/4.), magmom=mag[1])],
                           periodic=True)
 
    elif crys == 'fcc':
        bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0), magmom=mag[1]),
                            Atom(symbol, (  0.0, 1/2., 1/2.), magmom=mag[1]),
                            Atom(symbol, ( 1/2.,  0.0, 1/2.), magmom=mag[1]),
                            Atom(symbol, ( 1/2., 1/2.,  0.0), magmom=mag[1])],
                           periodic=True)

    elif crys == 'bcc':
        bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0), magmom=mag[1]),
                            Atom(symbol, ( 1/2., 1/2., 1/2.), magmom=mag[1])],
                           periodic=True)
 
    else:
        raise RuntimeError('Unknown crystal type: ' + crys)

    bulk.SetUnitCell([a, a, a])
    calcb = Calculator(gpts=(gpoints, gpoints, gpoints), kpts=(kpt, kpt, kpt),
                       xc=xc, out="%s-bulk-%s.out" % (symbol, xc))

    bulk.SetCalculator(calcb)
    e2 = bulk.GetPotentialEnergy()
    e3 = e2 / len(bulk) - e1

    return e3
