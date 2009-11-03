import numpy as np
from gpaw import FermiDirac
from gpaw.utilities.bulk2 import bulk, GPAWRunner

strains = np.linspace(0.98, 1.02, 5)
a = 2.84
atoms = bulk('Fe', 'bcc', a, orthorhombic=True)
for width in [0.05, 0.1, 0.2]:
    for k in [4, 6, 8, 10]:
        for g in [12, 16, 20, 24]:
            tag = '%.2f-%02d-%2d' % (width, k, g)
            r = GPAWRunner('Fe', atoms, strains, tag=tag)
            r.set_parameters(xc='PBE',
                             occupations=FermiDirac(width),
                             kpts=(k, k, k),
                             gpts=(g, g, g))
            r.run()
            r.summary(a0=a)
