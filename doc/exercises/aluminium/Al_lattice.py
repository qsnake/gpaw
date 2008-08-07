"""Lattice parameter convergence test"""

from ase import *
from gpaw import *

bulk = Atoms(symbols='4Al', pbc=True,
             positions=[(.0, .0, .0),
                        (.5, .5, .0),
                        (.0, .5, .5),
                        (.5, .0, .5)])

calc = GPAW(nbands=16,  gpts=(16, 16, 16), kpts=(6, 6, 6))
bulk.set_calculator(calc)
for a in [3.9, 4.0, 4.1, 4.2]:
    calc.set(txt='bulk-fcc-a%.1f.txt' % a)
    bulk.set_cell((a, a, a), scale_atoms=True)
    print a, bulk.get_potential_energy()

# run: ag bulk-fcc*.txt
# Choose 'Tools -> Bulk Modulus' to get

# B = 85.823 GPa, and
# V = 63.270 A^3 <=> a = 3.985

