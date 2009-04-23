from ase import *
# note that we overwite numpy.equal in the next line
from gpaw.utilities import equal
from gpaw import GPAW

# Values from revision 3775.
ref_3775 = { # Values from revision 3775.
    # A         Energy
    2.6  : -1.9857831386443334,
    2.65 : -1.9878176493286341,
    2.7  : -1.984691908434324,
    2.75 : -1.9773280974295946,
    2.8  : -1.9662827649879555,
    }

bulk = Atoms([Atom('Li')], pbc=True)
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2)
bulk.set_calculator(calc)
e = []
for a in ref_3775.keys():
    bulk.set_cell((a, a, a))
    e_a = bulk.get_potential_energy()
    equal(ref_3775[a], e_a, 2e-5)
    e.append(e_a)
print e

import numpy as np
a = np.roots(np.polyder(np.polyfit( ref_3775.keys(), e, 2), 1))[0]
print 'a =', a
assert abs(a - 2.6430) < 0.0001
