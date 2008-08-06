"""Bulk Al(bcc) test"""

from ase import *
from gpaw import *

afcc = 3.985          # Theoretical fcc lattice parameter
a = afcc * 2**(-1/3.) # Assuming the same volume per atom
a = afcc * sqrt(2/3.) # Assuming the same nearest neighbor distance

bulk = Atoms(symbols='2Al', pbc=True,
             positions=[(0, 0, 0),
                        (.5, .5, .5)])
bulk.set_cell((a, a, a), scale_atoms=True)

# View 3x3x3 repeated structure
view(bulk * [3, 3, 3])

calc = Calculator(nbands=8)
bulk.set_calculator(calc)

# Convergence with respect to k-points:
calc.set(h=.25, txt='Al-fcc-k.txt')
for k in [4, 6, 8, 10]: 
    calc.set(kpts=(k, k, k))
    print k, bulk.get_potential_energy() 

# Convergence with respect to grid spacing:
calc.set(kpts=(8, 8, 8), txt='Al-bcc-h.txt')
for g in [12, 16, 20]:
    h = a / g
    calc.set(h=h)
    print h, bulk.get_potential_energy() 

# Set parameters to reasonably converged values
calc.set(h=.28, kpts=(8, 8, 8))
for a in [3.0, 3.1, 3.2, 3.3]:
    calc.set(txt='bulk-bcc-a%.1f.txt' % a)
    bulk.set_cell((a, a, a), scale_atoms=True)
    print a, bulk.get_potential_energy()

# run: ag bulk-bcc*.txt
# Choose 'Tools -> Bulk Modulus' to get

# B = 74.633 GPa, and
# V = 32.472 A^3 <=> a = 3.190

# To be compared to the fcc values:
# B = 85.823 GPa, and
# V = 63.270 A^3 <=> a = 3.985
