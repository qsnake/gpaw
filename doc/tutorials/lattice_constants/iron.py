import numpy as np
from ase.structure import bulk
from gpaw import FermiDirac
from gpaw.utilities.bulk2 import GPAWRunner

strains = np.linspace(0.98, 1.02, 9)
a0 = 2.84
atoms = bulk('Fe', 'bcc', a0, orthorhombic=True)
atoms.set_initial_magnetic_moments([2.3, 2.3])
def f(width, k, g):
    tag = 'B-%.2f-%02d-%2d' % (width, k, g)
    r = GPAWRunner('Fe', atoms, strains, tag=tag)
    r.set_parameters(xc='PBE',
                     occupations=FermiDirac(width),
                     kpts=(k, k, k),
                     gpts=(g, g, g))
    r.run()

for width in [0.05, 0.1, 0.15, 0.2]:
    for k in [2, 4, 6, 8, 10, 12]:
        f(width, k, 16)
for g in range(12, 40, 4):
    f(0.1, 6, g)
