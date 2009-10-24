from ase import *
from gpaw import GPAW
from gpaw.test import equal

bulk = Atoms([Atom('Li')], pbc=True)
k = 4
g = 8
calc = GPAW(gpts=(g, g, g), kpts=(k, k, k), nbands=2, txt=None)
bulk.set_calculator(calc)
a = np.linspace(2.6, 2.8, 5)
e = []
for x in a:
    bulk.set_cell((x, x, x))
    e1 = bulk.get_potential_energy()
    niter1 = calc.get_number_of_iterations()
    e.append(e1)

fit = np.polyfit(a, e, 2)
a0 = np.roots(np.polyder(fit, 1))[0]
e0 = np.polyval(fit, a0)
print 'a,e =', a0, e0
assert abs(a0 - 2.6418) < 0.0001
assert abs(e0 - -1.98323) < 0.00002

energy_tolerance = 0.00001
niter_tolerance = 0
equal(e1, -1.96150207243, energy_tolerance) # svnversion 5252
equal(niter1, 14, niter_tolerance) # svnversion 5252
