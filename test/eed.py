import os

from ase import *
from gpaw import GPAW
from gpaw.utilities import equal
from gpaw.analyse.eed import ExteriorElectronDensity

sc = 1.5
R=0.7 # approx. experimental bond length
a = 2 * sc
c = 3 * sc
H2 = Atoms([Atom('H', (a/2, a/2, (c-R)/2)),
            Atom('H', (a/2, a/2, (c+R)/2))],
           cell=(a,a,c), pbc=False)
calc = GPAW(h=0.3, nbands=2,
                  convergence={'eigenstates':1.e-6})
H2.set_calculator(calc)
H2.get_potential_energy()

eed = ExteriorElectronDensity(calc.gd, calc.wfs.setups, H2)
eed.write_mies_weights(calc.wfs)
