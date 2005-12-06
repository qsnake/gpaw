from dimer_calc import dimer_calc
import Numeric as num
import pickle
import sys


def dimer(symbol, xc='LDA', hmin=0.25, hmax=0.3, L = 10.0, g=10):
    
    """Dimer test.
    
    Make use of the dimer_calc function, with an array of h (here are the min
    and max of the gridspace entered as ``hmin`` and ``hmax``). ``Symbol`` is
    the atomic symbol name, ``xc`` is the exchange correlation functional,
    ``L`` is the length of the cubic cell and ``g`` is number of variation of
    the bond length. A list of n number of energies and the related bond
    lengths are saved in the pickle file.
    """   

    data = {'Atomic symbol': symbol,
            'Size of unit cell': L,
            'Exchange-correlation functional': xc}
    
    nmax = int(L / hmin / 4 + 0.5) * 4
    nmin = int(L / hmax / 4 + 0.5) * 4
    j = []

    for n in range(nmin, nmax+1, 4):
        k = n
        j.append(k)

    energies = num.zeros((len(j), g), num.Float)
    bondlengths = num.zeros((len(j), g), num.Float)

    i = 0
    z=[L / x for x in j]

    for x in j:
        
        b, e = dimer_calc(symbol, xc, x, L, g)

        bondlengths[i] = b
        energies[i] = e
        
        i += 1

    data['Bond lengths'] = bondlengths
    data['Atomization energy'] = energies
    data['Grid spacings'] = z
    data['Test name'] = 'dimer'
    
    name = "%s-dimer-%s.pickle" % (symbol, xc)
    pickle.dump(data, open(name, 'w'))


if __name__ == '__main__':
    
    symbol = sys.argv[1]
 
    dimer(symbol)
