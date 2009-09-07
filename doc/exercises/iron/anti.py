from ase import *
from gpaw import GPAW
from gpaw.mixer import MixerSum

a = 2.87
m = 2.2
bulk = Atoms('Fe2',
             positions=[(0,   0,   0), (a/2, a/2, a/2)],
             magmoms=[m, -m],
             cell=(a, a, a),
             pbc=True)

calc = GPAW(kpts=(6, 6, 6),
            h=0.20,
            nbands=18,
            mixer=MixerSum(beta=0.2, nmaxold=5),
            eigensolver='dav',
            txt='anti.txt')

bulk.set_calculator(calc)
print bulk.get_potential_energy()
calc.write('anti.gpw')
