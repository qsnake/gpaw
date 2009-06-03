from ase import *
from gpaw import GPAW, extra_parameters
from gpaw.atom.basis import BasisMaker

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
A = [2.6, 2.65, 2.7, 2.75, 2.8]
for a in A:
    bulk.set_cell((a, a, a))
    e.append(bulk.get_potential_energy())
print e


import numpy as np
a = np.roots(np.polyder(np.polyfit(A, e, 2), 1))[0]
print 'a =', a
if extra_parameters.get('usenewlfc'):
    assert abs(a - 2.6591) < 0.0001
else:
    assert abs(a - 2.6567) < 0.0001
