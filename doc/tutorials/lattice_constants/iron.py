import numpy as np
from ase.structure import bulk
from gpaw import FermiDirac, MethfesselPaxton, MixerSum
from gpaw.utilities.bulk2 import GPAWRunner

strains = np.linspace(0.98, 1.02, 9)
a0 = 2.84
atoms = bulk('Fe', 'bcc', a0)
atoms.set_initial_magnetic_moments([2.3])
def f(name, dist, k, g):
    tag = '%s-%02d-%2d' % (name, k, g)
    r = GPAWRunner('Fe', atoms, strains, tag=tag)
    r.set_parameters(xc='PBE',
                     occupations=dist,
                     basis='dzp',
                     nbands=9,
                     kpts=(k, k, k),
                     gpts=(g, g, g))
    r.run()

for width in [0.05, 0.1, 0.15, 0.2]:
    for k in [2, 4, 6, 8, 10, 12]:
        f('FD-%.2f' % width, FermiDirac(width), k, 12)
        f('MP-%.2f' % width, MethfesselPaxton(width), k, 12)
for g in range(8, 32, 4):
    f('FD-%.2f' % 0.1, FermiDirac(0.1), 8, g)
