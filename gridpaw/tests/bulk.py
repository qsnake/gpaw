from bulk_calc import bulk_calc
import Numeric as num
import pickle
from elements import elements


def bulk(symbol, xc='LDA', hmin=0.2, hmax=0.3, kpt=6, crys='fcc', g=10):
    
    """Bulk function
    Make use of the bulk_calc function and find the dependence of the cohesive
    energy with the lattice constant: symbol is the atomic symbol, xc is the
    exchange correlation functional (default is LDA), it make an array of the
    gridspace resulting from hmin and hmax (default is [0.2,..,0.3]), kpt is
    the number of kpoints in each direction (default is (6, 6, 6)), crys is the
    crystal structure type (default is fcc) and g is the number of variations
    of the lattice constant. In the end the function make a dictionary with use
    of pickle, where the results of the heavy calculations are saved.
    """

    name = "%s-bulk-%s.pickle" % (symbol, xc)

    data = {'Atomic symbol': symbol,
            'exchange-correlation': xc,
            'crystal type': crys,
            'filename': name,
            'kpoint set': kpt}

    mag = elements[symbol]
    a = mag[3]
    p = int(g)
    data['number of calc'] = p
    grid = []

    nmax = int(a / hmin / 4 + 0.5) * 4
    nmin = int(a / hmax / 4 + 0.5) * 4

    for n in range(nmin, nmax+1, 4):
        y = n
        grid.append(y)

    energies = num.zeros((len(grid), int(g)), num.Float)
    lattice = num.zeros((len(grid), int(g)), num.Float)
    
    z=[a/x for x in grid]
    i=0
    
    for x in grid:
        
        coh, gridp = bulk_calc(symbol, xc=xc, gpoints=x, kpt=kpt, crys=crys,
                               g=g, a=a)
        lattice[i] = gridp
        energies[i] = coh

        i += 1
                
    data['Cohesive energies'] = energies
    data['Lattice constants'] = lattice
    data['Grid space'] = z
    data['Test name'] = 'bulk'

    pickle.dump(data, open(name, 'w'))


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
 
    bulk(symbol)
