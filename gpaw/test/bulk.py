from ase import *
from gpaw import GPAW

# Values from revision 3775.
eref = np.array([ # Values from revision 3775.
    -1.9857831386443334,
    -1.9878176493286341,
    -1.9846919084343240,
    -1.9773280974295946,
    -1.9662827649879555])

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

print e, e - eref
assert abs(e - eref).max() < 2e-5

a0 = np.roots(np.polyder(np.polyfit(a, e, 2), 1))[0]
print 'a =', a0
assert abs(a0 - 2.6430) < 0.0001
