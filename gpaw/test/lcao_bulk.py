import numpy as np
from ase import Atoms
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

a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
print 'a =', a
equal(a, 2.65573, 0.0001)

e_ref = [-1.7884213491957757, -1.7919042298804813, -1.7896686029340458,
         -1.7817996470037494, -1.7677986622459712]
niter_ref = [6, 6, 6, 6, 6]

print e
energy_tolerance = 0.00003
niter_tolerance = 0
for i in range(len(A)):
    equal(e[i], e_ref[i], energy_tolerance)
    equal(niter[i], niter_ref[i], niter_tolerance)
