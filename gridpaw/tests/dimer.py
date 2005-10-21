from dimer_calc import dimer_calc
import Numeric as num
import pickle


def dimer(symbol, xc='LDA', hmin=0.25, hmax=0.3, a = 10.0, g=10):
    
    """Dimer function
    Make use of the dimer_calc function, with an array of h (here are the min
    and max of the gridspace entered as hmin and hmax). Symbol is the atomic
    symbol, xc is the exchange correlation functional (default is LDA), a is
    the length of the cubic cell (default is 10 Aangstroem) and g is number of
    variation of the bond length. In the end the function make a dictionary
    with use of pickle, where the results of the heavy calculations are saved.
    """
    
    name = "%s-dimer-%s.pickle" % (symbol, xc)

    data = {'Atomic symbol': symbol,
            'unit-cell': a,
            'exchange-correlation': xc,
            'filename': name}
    
    nmax = int(a / hmin / 4 + 0.5) * 4
    nmin = int(a / hmax / 4 + 0.5) * 4

    j = []
    p = int(g)

    for n in range(nmin, nmax+1, 4):
        k = n
        j.append(k)

    energies = num.zeros((len(j), p), num.Float)
    bondlengths = num.zeros((len(j), p), num.Float)

    i = 0
    z=[a/x for x in j]

    for x in j:
        
        b, e = dimer_calc(symbol, xc, gpoints=x, a=a, n=g)

        bondlengths[i] = b
        energies[i] = e
        
        i += 1

    data['Varying bond-length'] = bondlengths
    data['bonding energy'] = energies
    data['array of grid space'] = z
    data['number of calculations'] = p
    data['Test name'] = 'dimer'
 
    pickle.dump(data, open(name, 'w'))


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
 
    dimer(symbol)
