from ase import *
from gpaw import GPAW, Mixer
from gpaw.test import equal

a = 4.0
n = 20
d = 1.0
x = d / 3**0.5
atoms = Atoms([Atom('C', (0.0, 0.0, 0.0)),
               Atom('H', (x, x, x)),
               Atom('H', (-x, -x, x)),
               Atom('H', (x, -x, -x)),
               Atom('H', (-x, x, -x))],
              cell=(a, a, a), pbc=True)
calc = GPAW(gpts=(n, n, n), nbands=4, txt=None,
            mixer=Mixer(0.25, 3, 1))
atoms.set_calculator(calc)
e0 = atoms.get_potential_energy()
niter0 = calc.get_number_of_iterations()

D = [1, 1.05, 1.1, 1.15]
E = []
for d in D:
    x = d / 3**0.5
    atoms.positions[1] = (x, x, x)
    e = atoms.get_potential_energy()
    print d, e - e0
    E.append(e)

fit = np.polyfit(D, E, 2)
d0 = np.roots(np.polyder(fit, 1))[0]
e0 = np.polyval(fit, d0)
print 'd,e =', d0, e0
equal(d0, 1.093204, 0.0001)
equal(e0, -23.2076, 0.0002)
assert 32 <= niter0 <= 34, niter0
