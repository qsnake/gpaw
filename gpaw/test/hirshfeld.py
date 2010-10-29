from ase import *
import numpy as np
from ase.io import write
from ase.data.molecules import molecule
from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning
from gpaw.utilities.tools import coordinates
from gpaw.test import equal

h=.3
gpwname = 'H2O' + str(h) + '.gpw'
try:
    calc = GPAW(gpwname, txt=None)
    mol = calc.get_atoms()
except:
    mol = Cluster(molecule('H2O'))
    mol.minimal_box(3, h=h)
    calc = GPAW(nbands=6,
                h = h)
    calc.calculate(mol)
    calc.write(gpwname)

hd = HirshfeldDensity(calc)

# check for the number of electrons
expected = [[None, 10],
            [[0, 1, 2], 10],
            [[1, 2], 2],
            [[0], 8],
            ]
for result in expected:
    indicees, result = result
    full, gd = hd.get_density(indicees)
    print 'indicees', indicees, 
    print 'result, expected:', gd.integrate(full[0]), result
    assert(abs(gd.integrate(full[0]) - result) < 1e-10)

hp = HirshfeldPartitioning(calc)
vr = hp.get_effective_volume_ratios()
print vr
equal(vr[1], vr[2], 1.e-12)
