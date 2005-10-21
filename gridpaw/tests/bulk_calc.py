from bulk_lattice import bulk_lattice
import Numeric as num


def bulk_calc(symbol, xc='LDA', gpoints=28, kpt=6, crys='fcc', g=10, a=3.0):
    
    """Bulk_calc function
    Make use of the atom_calc function and find the dependence of the cohesive
    energy with the lattice constant: gpoints is the fixed gridpoints for the
    calculation, kpt is the number of kpoints for the bulk calculation, crys
    named the crystal structure type, g is the variation of the lattice
    constant and a is the lattice constant. In the end the function make a
    dictionary with use of pickle, where the results are saved.
    """

    L = 5 + a
    N = int(gpoints / a * L / 4) * 4

    j = []

    m = ( g - 1 ) / 2.0

    f = int(m)
  
    if m > f:
        i = 1
    else:
        i = 0
            
    for n in range( - f, f + 1 + i):
        k = a + n * 0.05
        j.append(k)

    coh_energy = num.zeros((len(j)), num.Float)
    i = 0
    
    for y in j:
        
        c = bulk_lattice(symbol, xc=xc, gpoints=gpoints, kpt=kpt, a=y, L=L,
                         N=N, crys=crys)
        coh_energy[i] = c

        i += 1

    return coh_energy, j


if __name__ == '__main__':
    import sys
    symbol = sys.argv[1]
 
    bulk_calc(symbol)
