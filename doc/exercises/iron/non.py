from ase import Atoms
from gpaw import GPAW

a = 2.87
bulk = Atoms('Fe2',
             positions=[(0, 0, 0),
                        (a/2, a/2, a/2)],
             cell=(a, a, a),
             pbc=True)

calc = GPAW(kpts=(6, 6, 6),
            h=0.20,
            nbands=18,
            eigensolver='cg',
            txt='non.txt')

bulk.set_calculator(calc)
print bulk.get_potential_energy()
calc.write('non.gpw')
