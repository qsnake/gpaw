from ase import *
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker
from gpaw.test import equal

bm = BasisMaker('Li', run=False)
bm.generator.N = 300
bm.generator.run(write_xml=False)
basis = bm.generate(2, 1, energysplit=0.3, tailnorm=0.03**0.5)

bulk = Atoms('Li', pbc=True)
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k),
                  mode='lcao', basis={'Li' : basis})
bulk.set_calculator(calc)
e = []
niter = []
A = [2.6, 2.65, 2.7, 2.75, 2.8]
for a in A:
    bulk.set_cell((a, a, a))
    e.append(bulk.get_potential_energy())
    niter.append(calc.get_number_of_iterations())

import numpy as np
a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
print 'a =', a
assert abs(a - 2.6562) < 0.0001

e_ref = [-1.7881215825065007, -1.791655508469838, -1.7894800958944614, -1.7816813613079183, -1.7677598265562726]
niter_ref = [6, 6, 6, 6, 6] # svnversion 5252

print e
energy_tolerance = 0.000005
niter_tolerance = 0
for i in range(len(A)):
    equal(e[i], e_ref[i], energy_tolerance)
    equal(niter[i], niter_ref[i], niter_tolerance)
