from gridpaw import Calculator
from ASE import ListOfAtoms, Atom
from elements import elements
from ASE.Visualization.RasMol import RasMol


def bulk_lattice(symbol, xc='LDA', gpoints=28, kpt=6, a=5, L=10, N=36,
                 crys='fcc'):

    """bulk_lattice function
    Calculate the cohesive energy:
    Symbol is the atomic symbol, xc is the exchange correlation function
    (default is LDA), gpoints is the number of grid points in the cell in each
    direction (default is 28), a is the lattice constant (the experimential
    value is used from the elements list), L is the length of the atomic cell,
    N is the number of grid points in the atomic cell and crys is the crystal
    structure type (default is fcc).
    """

    m = elements[symbol]

    # calculation of the single atom

    if crys in ('diamond', 'fcc', 'bcc'):
        atom = ListOfAtoms([Atom(symbol, ( L/2, L/2, L/2), magmom=m[0])],
                           cell=(L, L, L))

        calc = Calculator(gpts=(N, N, N), xc=xc, decompose_domain_only=False,
                          out="%s-atom-%s.out" % (symbol, xc))

        atom.SetCalculator(calc)
        e1 = atom.GetPotentialEnergy()
 
    else:
        print 'Unknown crystal type: ', crys

    # calculation of the bulk cell

    if crys == 'diamond':
        
        bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0),    magmom=m[1]),
                            Atom(symbol, ( 1/4., 1/4., 1/4.),    magmom=m[1]),
                            Atom(symbol, (  0.0, 1/2., 1/2.),    magmom=m[1]),
                            Atom(symbol, ( 1/4., 3/4., 3/4.),    magmom=m[1]),
                            Atom(symbol, ( 1/2.,  0.0, 1/2.),    magmom=m[1]),
                            Atom(symbol, ( 3/4., 1/4., 3/4.),    magmom=m[1]),
                            Atom(symbol, ( 1/2., 1/2.,  0.0),    magmom=m[1]),
                            Atom(symbol, ( 3/4., 3/4., 1/4.),    magmom=m[1])],
                           periodic=True)
        cell = [a, a, a]
 
    elif crys == 'fcc':

        bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0),    magmom=m[1]),
                            Atom(symbol, (  0.0, 1/2., 1/2.),    magmom=m[1]),
                            Atom(symbol, ( 1/2.,  0.0, 1/2.),    magmom=m[1]),
                            Atom(symbol, ( 1/2., 1/2.,  0.0),    magmom=m[1])],
                           periodic=True)

        cell = [a, a, a]

    elif crys == 'bcc':

        bulk = ListOfAtoms([Atom(symbol, (  0.0,  0.0,  0.0),    magmom=m[1]),
                            Atom(symbol, ( 1/2., 1/2., 1/2.),    magmom=m[1])],
                           periodic=True)

        cell = [a, a, a]

    bulk.SetUnitCell(cell)
    # plot = RasMol(bulk, (3,3,3))

    calcb = Calculator(gpts=(gpoints, gpoints, gpoints), kpts=(kpt, kpt, kpt),
                       xc=xc, decompose_domain_only=False,
                       out="%s-bulk-%s.out" % (symbol, xc))

    bulk.SetCalculator(calcb)
    e2 = bulk.GetPotentialEnergy()

    if crys == 'diamond':

        e3 = (e2 - 8*e1)/8
        return e3
        
    elif crys == 'fcc':

        e3 = (e2 - 4*e1)/4
        return e3

    elif crys == 'hcp':

        e3 = (e2 - 2*e1)/2
        return e3
    
    elif crys == 'bcc':

        e3 = (e2 - 2*e1)/2
        return e3

  
if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
 
    bulk_lattice(symbol)




