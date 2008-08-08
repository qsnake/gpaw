from gpaw import GPAW
from ase.lattice.surface import fcc100

def energy(n, k, a=4.05):
    fcc = fcc100('Al', (1, 1, n), a=a, vacuum=7.5)
    calc = GPAW(nbands=n * 5,
                kpts=(k, k, 1),
                h=0.25,
                txt='slab-%d.txt' % n)
    fcc.set_calculator(calc)
    e = fcc.get_potential_energy()
    calc.write('slab-%d.gpw' % n)
    return e

e4 = energy(4, ?)
