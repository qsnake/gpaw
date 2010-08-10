"""Bulk Al(fcc) test"""

from ase import Atoms
from ase.visualize import view
from gpaw import GPAW

k = 4
g = 12
for a in [3.9, 4.0, 4.1, 4.2]:
    name = 'wbulk-fcc-%.1f-%d' % (a, k)
    b = a / 2 

    bulk = Atoms('Al',
                 cell=[[0, b, b],
                       [b, 0, b],
                       [b, b, 0]],
                 pbc=True)

    calc = GPAW(gpts=(g, g, g),     # grid points
                kpts=(k, k, k),     # k-points
                txt=name + '.txt')  # output file

    bulk.set_calculator(calc)

    energy = bulk.get_potential_energy()
    print 'Energy:', energy, 'eV'

# run: ag bulk-fcc*.txt
# Choose 'Tools -> Bulk Modulus' to get

# B = 87.599 GPa, and
# V = 15.901 A^3 <=> a = 3.992
# 15.830, 87.154
# 15.777, 85.998
# 15.805, 84.032
#(15.812, 83.875)
# 15.831, 84.439
# 15.843, 84.529
#(15.845,)
