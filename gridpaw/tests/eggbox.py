from atom_calc import atom_calc
import Numeric as num
import pickle
import sys


def eggbox(symbol, xc='LDA', hmin=0.25, hmax=0.3, a = 7.0):
    
    """Egg box function
    Make use of the atom_calc function, but with an array of h (here are the
    min and max of the gridspace entered as hmin and hmax). Symbol is the
    atomic symbol, xc is the exchange correlation function and a is the cubic
    cell length (default is 7 Aangstroem). In the end the function make a
    dictionary with use of pickle, where the results of are saved.
    """

    name = "%s-eggbox-%s.pickle" % (symbol, xc)

    data = {'Atomic symbol': symbol,
            'unit-cell': a,
            'exchange-correlation': xc,
            'filename': name}
    
    j = []
    nmax = int(a / hmin / 4 + 0.5) * 4
    nmin = int(a / hmax / 4 + 0.5) * 4
              
    for n in range(nmin, nmax + 1, 4):
        k = n
        j.append(k)

    z = [a / x for x in j]
    tot_energy = num.zeros((len(j), 101), num.Float)

    i = 0

    for x in j:
        
        tot = atom_calc(symbol, xc, gpoints=x, a=a)
        tot_energy[i] = tot

        i += 1
        
    data['Total energy of atom'] = tot_energy
    data['array of grid points'] = j
    data['array of grid space'] = z
    data['Test name'] = 'eggbox'

    pickle.dump(data, open(name, 'w'))


if __name__ == '__main__':
    
    symbol = sys.argv[1]
    
    eggbox(symbol)
