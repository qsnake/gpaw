"""Bulk Al(fcc) test"""

from ase import Atoms
from ase.visualize import view
from gpaw import GPAW

k = 8
g = 16
for a in [3.9, 4.0, 4.1, 4.2]:
    name = 'bulk-fcc-%.1f-%d' % (a, k)
    b = a / 2 

    bulk = Atoms('Al',
                 cell=[[0, b, b],
                       [b, 0, b],
                       [b, b, 0]],
                 pbc=True)

    calc = GPAW(gpts=(g, g, g),           # grid points
                kpts=(k, k, k),           # k-points
                txt=name + '.txt',        # output file
                parallel=dict(domain=1))  # force parallelization over k-points

    bulk.set_calculator(calc)

    energy = bulk.get_potential_energy()
    print 'Energy:', energy, 'eV'

# run: ag bulk-fcc*.txt
# Choose 'Tools -> Bulk Modulus' to get

# B = 85.355 GPa, and
# V = 15.783 A^3 <=> a = 3.982
