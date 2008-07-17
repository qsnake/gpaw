from ase import *
from gpaw import Calculator

a = 2.87
m = 2.2
bulk = Atoms('Fe2',
             positions=[(0,   0,   0), (a/2, a/2, a/2)],
             magmoms=[m, -m],
             cell=(a, a, a),
             pbc=True)

calc = Calculator(kpts=(6, 6, 6),
                  h=0.20,
                  nbands=18,
                  eigensolver='dav',
                  txt='anti.txt')

bulk.set_calculator(calc)
print bulk.get_potential_energy()
calc.write('anti.gpw')
