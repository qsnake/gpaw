""" Compare Al(fcc) and Al(bcc) at two different grid spacings"""

from ase import *
from gpaw import *

afcc = 3.985
abcc = 3.190

for h in [0.25, 0.20]:
    #fcc
    bulk =  Atoms(symbols='4Al', pbc=True,
                  positions=[(.0, .0, .0),
                             (.5, .5, .0),
                             (.0, .5, .5),
                             (.5, .0, .5)])
    bulk.set_cell([afcc,afcc,afcc],scale_atoms=True)
    calc = GPAW(nbands=16,
                txt='bulk-fcc-h%.2f.txt' % h,
                h=h ,
                kpts=(6, 6, 6))
    bulk.set_calculator(calc)
    Efcc=bulk.get_potential_energy()
    #bcc
    bulk = Atoms(symbols='2Al', pbc=True,
                 positions=[(0, 0, 0),
                            (.5, .5, .5)])
    bulk.set_cell((abcc, abcc, abcc), scale_atoms=True)
    calc = GPAW(nbands=8,
                txt='bulk-bcc-h%.2f.txt' % h,
                h=h,
                kpts=(8, 8, 8))
    bulk.set_calculator(calc)
    Ebcc=bulk.get_potential_energy()
    print h, Efcc/4., Ebcc/2., 0.25*Efcc-0.5*Ebcc

# 0.25 -4.17322125175 -4.08445636201 -0.0887648897448
# 0.20 -4.17337174875 -4.0846250688  -0.088746679949
# The total energy is converged within ca. 1e-4 eV/atom
# The energy difference is convereged within ca. 1e-5 eV/atom
