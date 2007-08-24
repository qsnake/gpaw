from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.wannier import Wannier
from ASE.Utilities.MonkhorstPack import MonkhorstPack
from gpaw.utilities import equal

natoms = 1
hhbondlength = 0.9
atoms = ListOfAtoms([Atom('H', (0, 4.0, 4.0))],
                    cell=(hhbondlength, 8., 8.),
                    periodic=True).Repeat((natoms, 1, 1))

# Displace kpoints sligthly, so that the symmetry program does
# not use inversion symmetry to reduce kpoints.
kpts = [21, 11, 7][natoms - 1]
occupationenergy = [30., 0., 0.][natoms - 1]
kpts = MonkhorstPack((kpts, 1, 1)) + 2e-5

if 1:
    # GPAW calculator:
    calc = Calculator(nbands=natoms // 2 + 4,
                      kpts=kpts,
                      width=.1,
                      spinpol=False,
                      tolerance=1e-7)
    atoms.SetCalculator(calc)
    atoms.GetPotentialEnergy()
    calc.write('hwire.gpw', 'all')
else:
    calc = Calculator('hwire.gpw', txt=None)

wannier = Wannier(numberofwannier=natoms, 
                  calculator=calc,
                  occupationenergy=occupationenergy)

wannier.Localize()
wannier.TranslateAllWannierFunctionsToCell([1, 0, 0])

centers = wannier.GetCenters()
for i in wannier.GetSortedIndices():
    center = centers[i]['pos']
    print center
    equal(center[0] % hhbondlength, 0., 2e-3)
    equal(center[1], 4., 2e-3)
    equal(center[2], 4., 2e-3)

for i in range(natoms):
    wannier.WriteCube(i, 'hwire%s.cube' % i)
