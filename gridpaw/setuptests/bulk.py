from bulk_calc import bulk_calc
import pickle
from math import sqrt
from elements import elements


def bulk(symbol, xc='LDA', hmin=0.2, hmax=0.3, kpt=6, crys='fcc', g=10):
    
    """Bulk test.
    
    Make use of the bulk_calc function and find the dependence of the cohesive
    energy with the lattice constant: ``symbol`` is the atomic symbol name,
    ``xc`` is the exchange-correlation functional, it make an array of the
    gridspace resulting from ``hmin`` and ``hmax``, ``kpt`` is the number of
    kpoints in each direction, ``crys`` is the crystal structure type and ``g``
    is the number of variations of the lattice constant. A result of g energies
    are saved in the pickle file from the bulk_calc function.
    """
    
    data = {'Atomic symbol': symbol,
            'Exchange-correlation functional': xc,
            'Crystal type': crys,
            'Number of k-points': kpt}

    mag = elements[symbol]
    
    if mag[5] == 'hcp':
        a = sqrt(2) * mag[3]
    else:
        a = mag[3]        

    grid = []
    nmax = int(a / hmin / 4 + 0.5) * 4
    nmin = int(a / hmax / 4 + 0.5) * 4

    for n in range(nmin, nmax+1, 4):
        y = n
        grid.append(y)

    energies = []
    lattice = []
    
    for x in grid:
        
        coh, gridp = bulk_calc(symbol, xc, x, kpt, crys, g, a)
        lattice.append(gridp)
        energies.append(coh)

    z=[a / x for x in grid]
    data['Grid spacings'] = z
    data['Cohesive energies'] = energies
    data['Lattice constants'] = lattice
    data['Test name'] = 'bulk'

    name = "%s-%s-%s.pickle" % (symbol, crys, xc)
    pickle.dump(data, open(name, 'w'))
