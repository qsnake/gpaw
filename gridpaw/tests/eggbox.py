from atom_calc import atom_calc
import Numeric as num
import pickle
import sys


def eggbox(symbol, xc='LDA', hmin=0.25, hmax=0.3, L = 7.0):
    
    """Eggbox test.

    Make use of the atom_calc function, but with an array of h (here are the
    min and max of the gridspace entered as ``hmin`` and ``hmax``). ``Symbol``
    is the atomic symbol name, ``xc`` is the exchange-correlation functional
    and ``L`` is the cubic cell length. A list of 101 energies is returned at
    varying positions in the grid space from the atom_calc function.
    """

    data = {'Atomic symbol': symbol,
            'Size of unit cell': L,
            'Exchange-correlation functional': xc}
    
    j = []
    nmax = int(L / hmin / 4 + 0.5) * 4
    nmin = int(L / hmax / 4 + 0.5) * 4
              
    for n in range(nmin, nmax + 1, 4):
        k = n
        j.append(k)

    z = [L / x for x in j]
    tot_energy = num.zeros((len(j), 101), num.Float)

    i = 0

    for x in j:
        
        tot = atom_calc(symbol, xc, x, L)
        tot_energy[i] = tot

        i += 1
        
    data['Total energies'] = tot_energy
    data['Grid spacings'] = z
    data['Test name'] = 'eggbox'
    
    name = "%s-eggbox-%s.pickle" % (symbol, xc)
    pickle.dump(data, open(name, 'w'))
