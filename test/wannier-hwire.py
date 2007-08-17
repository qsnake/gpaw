from gpaw import Calculator
from ASE import Atom, ListOfAtoms
from gpaw.wannier import Wannier
from ASE.Utilities.MonkhorstPack import MonkhorstPack

natoms = 1
hhbondlength = 0.9
atoms1 = ListOfAtoms([Atom('H', (0, 4.0, 4.0))],
                    cell=(hhbondlength, 8., 8.), periodic=True)
atomsN = atoms1.Repeat((natoms, 1, 1))

# Displace kpoints sligthly, so that the symmetry program does
# not use inversion symmetry to reduce kpoints.
kpts = MonkhorstPack((21, 1, 1)) + 2e-5

if 1:
    # GPAW calculator:
    calc = Calculator(nbands=natoms // 2 + 4,
                      kpts=kpts,
                      width=.08,
                      tolerance=1e-6)
    atomsN.SetCalculator(calc)
    atomsN.GetPotentialEnergy()
    calc.write('wire.gpw', 'all')

calc = Calculator('wire.gpw')
wannier = Wannier(numberofwannier=natoms,
                  calculator=calc,
                  numberoffixedstates=[natoms] * len(kpts))

wannier.Localize()
wannier.TranslateAllWannierFunctionsToCell([10,0,0])

centers = wannier.GetCenters()
for i in wannier.GetSortedIndices():
   print centers[i]

wannier.WriteCube(0, 'hwire.cube')
