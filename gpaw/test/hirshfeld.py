from ase.data.molecules import molecule
from ase.parallel import parprint

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.hirshfeld import HirshfeldDensity, HirshfeldPartitioning
from gpaw.test import equal

h = 0.3
gpwname = 'H2O' + str(h) + '.gpw'
try:
    calc = GPAW(gpwname + 'notfound', txt=None)
    mol = calc.get_atoms()
except:
    mol = Cluster(molecule('H2O'))
    mol.minimal_box(3, h=h)
    calc = GPAW(nbands=6,
                h = h, 
                txt=None)
    calc.calculate(mol)
    calc.write(gpwname)

hd = HirshfeldDensity(calc)

# check for the number of electrons
expected = [[None, 10],
            [[0, 1, 2], 10],
            [[1, 2], 2],
            [[0], 8],
            ]
#expected = [[[0, 2], 9], ]
#expected = [[None, 10], ]
for result in expected:
    indicees, result = result
    full, gd = hd.get_density(indicees)
    parprint('indicees', indicees, end=': ') 
    parprint('result, expected:', gd.integrate(full), result)
    equal(gd.integrate(full), result, 1.e-8)


if 1:
    hp = HirshfeldPartitioning(calc)
    vr = hp.get_effective_volume_ratios()
    parprint(vr)
#    equal(vr[1], vr[2], 1.e-3)
