from hcp_calc import hcp_calc
import Numeric as num
import pickle
from elements import elements
from math import sqrt
import sys


def hcp(symbol, xc='LDA', hmin=0.2, hmax=0.3, kpt=6, g=10):
    
    """hcp function
    Make use of the hcp_calc function and find the dependence of the cohesive
    energy with the lattice constant a,c: symbol is the atomic symbol, xc is
    the exchange correlation functional (default is LDA), it make an array of
    the gridspace resulting from hmin and hmax (default is [0.2,..,0.3]), kpt
    is the number of kpoints in each direction (default is (6, 6, 6)) and g is
    the number of variations of the lattice constant. In the end the function
    make a dictionary with use of pickle, where the results are saved.
    """

    name = "%s-hcp-%s.pickle" % (symbol, xc)

    data = {'Atomic symbol': symbol,
            'exchange-correlation': xc,
            'filename': name,
            'kpoint set': kpt}

    mag = elements[symbol]
    a = mag[3]
    c = mag[4]
    print 'a: ', a, 'c: ', c
    cov = c/a
    p = int(g)
    data['number of calc'] = p
    grid = []
    z = []
    
    nmax = int(a / hmin / 4 + 0.5) * 4
    nmin = int(a / hmax / 4 + 0.5) * 4
    wide = range(nmin, nmax + 1, 4)

    for n in wide:
        nx = n
        ny = 4 * int(sqrt(3) * n / 4 + 0.5)
        nz = 4 * int(cov * n / 4 + 0.5)
        grid.append((nx, ny, nz))

    for x1,x2,x3 in grid:
        z1 = a / x1
        z2 = sqrt(3) * a / x2
        z3 = (cov * a) / x3
        z_res = (z1 * z2 * z3)**(1/3.)
        z.append(z_res)
        
    energies = num.zeros((len(wide), int(g), int(g)), num.Float)
    lattice = num.zeros((int(g)), num.Float)
    cov = num.zeros((int(g)), num.Float)
    
    i=0
    for x in grid:

        coh, gridp, covera = hcp_calc(symbol, xc=xc, gpoints=x, kpt=kpt, g=g)
        lattice = gridp
        cov = covera
        energies[i] = coh

        i += 1
                
    data['Cohesive energies'] = energies
    data['Lattice constants'] = lattice
    data['covera c/a'] = cov
    data['Grid space meanvalue'] = z
    data['Test name'] = 'hcp'

    pickle.dump(data, open(name, 'w'))


if __name__ == '__main__':

    symbol = sys.argv[1]
 
    hcp(symbol)
