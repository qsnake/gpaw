from hcp_calc import hcp_calc
import Numeric as num
import pickle
from elements import elements
from math import sqrt
import sys


def hcp(symbol, xc='LDA', hmin=0.2, hmax=0.3, kpt=6, g=10):
    
    """hcp test.
    
    Make use of the hcp_calc function and find the dependence of the cohesive
    energy with the lattice constant a and cov (c/a): ``symbol`` is the atomic
    symbol name, ``xc`` is the exchange-correlation functional, it make an
    array of the gridspace resulting from ``hmin`` and ``hmax``, ``kpt`` is the
    number of k-points in each direction and ``g`` is the number of variations
    of the lattice constants. A result of g*g energies are saved in the pickle
    file from the hcp_calc function.
    """

    data = {'Atomic symbol': symbol,
            'Exchange-correlation functional': xc,
            'Number of k-points': kpt}

    mag = elements[symbol]

    if mag[5] == 'hcp':
        a = mag[3]
        c = mag[4]
    else:
        a = mag[3]/sqrt(2)
        c = a * 1.63

    cov = c/a
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
        
    energies = num.zeros((len(wide), g, g), num.Float)
    lattice = []
    cova = []    
    i=0
    
    for x in grid:

        coh, lat_cons, covera = hcp_calc(symbol, xc, x, kpt, g, a, cov)
        lattice = lat_cons
        cova = covera
        energies[i] = coh
        
        i += 1
                
    data['Cohesive energies'] = energies
    data['Lattice constants'] = lattice
    data['Covera c/a'] = cova
    data['Grid spacing meanvalues'] = z
    data['Test name'] = 'hcp'
    
    name = "%s-hcp-%s.pickle" % (symbol, xc)
    pickle.dump(data, open(name, 'w'))
