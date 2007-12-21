import Numeric as num

from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.utilities.dos import raw_orbital_LDOS, raw_wignerseitz_LDOS
from gpaw.utilities import center, equal

Hnospin = ListOfAtoms([Atom('H')], cell=[5, 5, 5], periodic=False)
Hspin = ListOfAtoms([Atom('H', magmom=1)], cell=[5, 5, 5], periodic=False)
LiH = ListOfAtoms([Atom('Li', [.0, .0, .41]),
                   Atom('H', [.0, .0, -1.23]),
                   ], cell=[5, 5, 6.5], periodic=False)
center(Hnospin)
center(Hspin)
center(LiH)

calc = Calculator(fixmom=True, hund=True)
Hnospin.SetCalculator(calc)
Hnospin.GetPotentialEnergy()
energies, sweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')
energies, pdfweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='pdf')

calc = Calculator(fixmom=True, hund=True)
Hspin.SetCalculator(calc)
Hspin.GetPotentialEnergy()
energies,sweight_spin = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')

calc = Calculator(fixmom=True, nbands=2, eigensolver='dav')
LiH.SetCalculator(calc)
LiH.GetPotentialEnergy()
energies, Li_orbitalweight = raw_orbital_LDOS(calc, a=0, spin=0, angular=None)
energies, H_orbitalweight = raw_orbital_LDOS(calc, a=1, spin=0, angular=None)
energies, Li_wzweight = raw_wignerseitz_LDOS(calc, a=0, spin=0)
energies, H_wzweight = raw_wignerseitz_LDOS(calc, a=1, spin=0)
n_a = calc.GetWignerSeitzDensities(spin=0)

print sweight, pdfweight
print sweight_spin

print Li_orbitalweight
print Li_wzweight
print H_wzweight
print n_a

equal(sweight[0], 1., .06) 
equal(pdfweight[0], 0., .0001) 
equal(sweight_spin[0], 1.14, .06) 
assert num.alltrue(num.around(Li_wzweight - [.12, .94], 2) == 0)
assert num.alltrue(num.around(H_wzweight  - [.88, .06], 2) == 0)
equal(num.sum(n_a), 0., 1e-5)
equal(n_a[1], .766, .001)

#               HOMO    s   px  pz  py  *s
Li_orbitalweight[0] -= [.4, .0, .6, .0, .0]
H_orbitalweight[0]  -= [.7, .0, .0, .0, .0]

#              LUMO       s  px   pz  py  *s
Li_orbitalweight[1] -= [1.6, .0, 1.1, .0, .0]
H_orbitalweight[1]  -= [0.1, .0, 0.0, .0, .0]

assert num.alltrue(num.around(Li_orbitalweight, 1) == 0)
assert num.alltrue(num.around(H_orbitalweight, 1) == 0)
