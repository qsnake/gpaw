from ase import *
# note that we overwite numpy.equal in the next line
from gpaw.utilities import equal
from gpaw import GPAW, extra_parameters

ref_3775 = [ # Values from revision 3775.
    # d         Energy
    (1.00, -23.4215573953),
    (1.05, -23.541808956),
    (1.10, -23.5629210099),
    (1.15, -23.50919649),
    ]

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
atoms.set_calculator(GPAW(gpts=(n, n, n), nbands=4, txt=None))
e0 = atoms.get_potential_energy()

for d, eref in ref_3775:
    x = d / 3**0.5
    atoms.positions[1] = (x, x, x)
    e = atoms.get_potential_energy()
    print d, e - e0, e-eref
    if extra_parameters.get('usenewlfc'):
        eref += 0.00036
    equal(eref, e, 7e-5)
