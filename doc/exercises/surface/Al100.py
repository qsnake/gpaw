from gpaw import Calculator
from build_fcc import fcc100

a = 4.05
def energy(n):
    fcc = fcc100('Al', a, n, 20.0)
    calc = Calculator(nbands=n * 5,
                      kpts=(6, 6, 1),
                      h=0.25,
                      txt='slab-%d.txt' % n)
    fcc.set_calculator(calc)
    e = fcc.get_potential_energy()
    calc.write('slab-%d.gpw' % n)
    return e

f = file('e6x6.dat', 'w')
for n in range(1, 7):
    e = energy(n)
    print n, e
    print >> f, n, e
