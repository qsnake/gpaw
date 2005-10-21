from gridpaw import Calculator
from ASE.Visualization.RasMol import RasMol
from ASE import ListOfAtoms, Atom
from elements import elements
from math import sqrt
import Numeric as num


def hcp_calc(symbol, xc='LDA', gpoints=(28,28,28), kpt=6, g=10):

    """hcp_calc function
    Calculate the cohesive energy:
    Symbol is the atomic symbol, xc is the exchange correlation function
    (default is LDA), gpoints is the number of grid points in the cell in each
    direction (default is (28,28,28)), kpt is the kpoints in each direction and
    g is the number of variation of the lattice constant a,c.
    """
    
    mag = elements[symbol]
    a = mag[3]
    c = mag[4]
    cov = c/a

    j = []
    covera = []

    m = ( g - 1 ) / 2.0
    f = int(m)
  
    if m > f:
        i = 1
    else:
        i = 0
            
    for n in range( - f, f + 1 + i):
        k = a + n * 0.05
        c = cov + n * 0.05
        j.append(k)
        covera.append(c)

    coh_energy = num.zeros((len(covera), len(j)), num.Float)
    k = 0

    for q in covera:
        i = 0

        for y in j:
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
            # plot = RasMol(bulk, (3,3,3))

            calc = Calculator(gpts=gpoints, kpts=(kpt, kpt-2, kpt-2), xc=xc,
                              out="%s-hcp-bulk-%s.out" % (symbol, xc))

            bulk.SetCalculator(calc)
            e2 = bulk.GetPotentialEnergy()
            e3 = (e2 - 4*e1)/4

            # print k,i,coh_energy.shape
            coh_energy[k, i] = e3

            i += 1
        k += 1

    return coh_energy, j, covera


  
if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
 
    hcp_calc(symbol)




