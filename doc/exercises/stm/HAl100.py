from ase import Atoms, Atom
from gpaw import GPAW

a = 4.0
b = a / 2**.5
L = 11.0

# Set up 2-layer 2x2 (100) Al-slab:
slab = Atoms('Al2',
             positions=[(0, 0, 0),
                        (b / 2, b/ 2, -a / 2)],
             cell=(b, b, L),
             pbc=True)
slab *= (2, 2, 1)

if True:
    # Add adsorbate:
    slab += Atom('H', (b, b, 1.55))
    
calc = GPAW(kpts=(4, 4, 1))
slab.set_calculator(calc)
slab.get_potential_energy()
calc.write('HAl100.gpw', 'all')
