from ase import *
from gpaw import GPAW

bulk = Atoms([Atom('Li')], pbc=True)
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None)
bulk.set_calculator(calc)
a = np.linspace(2.6, 2.8, 5)
e = []
for x in a:
    bulk.set_cell((x, x, x))
    e.append(bulk.get_potential_energy())

fit = np.polyfit(a, e, 2)
a0 = np.roots(np.polyder(fit, 1))[0]
e0 = np.polyval(fit, a0)
print 'a,e =', a0, e0
assert abs(a0 - 2.6418) < 0.0001
assert abs(e0 - -1.98323) < 0.00002
