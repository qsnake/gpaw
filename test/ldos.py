import os
import numpy as npy
from ase import *
from gpaw import Calculator
from gpaw.utilities.dos import raw_orbital_LDOS, raw_wignerseitz_LDOS, RawLDOS
from gpaw.utilities import equal

Hnospin = Atoms([Atom('H')], cell=[5, 5, 5], pbc=False)
Hspin = Atoms([Atom('H', magmom=1)], cell=[5, 5, 5], pbc=False)
LiH = Atoms([Atom('Li', [.0, .0, .41]),
             Atom('H', [.0, .0, -1.23]),
             ], cell=[5, 5, 6.5], pbc=False)
Hnospin.center()
Hspin.center()
LiH.center()

# This is needed for the Wigner-Zeitz test to give
# architecture-independent results:
LiH.translate(0.003234)

calc = Calculator(fixmom=True, hund=True)
Hnospin.set_calculator(calc)
Hnospin.get_potential_energy()
energies, sweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')
energies, pdfweight = raw_orbital_LDOS(calc, a=0, spin=0, angular='pdf')

calc = Calculator(fixmom=True, hund=True)
Hspin.set_calculator(calc)
Hspin.get_potential_energy()
energies,sweight_spin = raw_orbital_LDOS(calc, a=0, spin=0, angular='s')

calc = Calculator(fixmom=True, nbands=2, eigensolver='dav')
LiH.set_calculator(calc)
LiH.get_potential_energy()
energies, Li_orbitalweight = raw_orbital_LDOS(calc, a=0, spin=0, angular=None)
energies, H_orbitalweight = raw_orbital_LDOS(calc, a=1, spin=0, angular=None)
energies, Li_wzweight = raw_wignerseitz_LDOS(calc, a=0, spin=0)
energies, H_wzweight = raw_wignerseitz_LDOS(calc, a=1, spin=0)
n_a = calc.get_wigner_seitz_densities(spin=0)

ldos = RawLDOS(calc)
fname = 'ldbe.dat'
ldos.by_element_to_file(fname)
ldos.by_element_to_file(fname, 2.)
os.remove(fname)

## print sweight, pdfweight
## print sweight_spin
## print Li_orbitalweight
## print Li_wzweight
## print H_wzweight
## print n_a

equal(sweight[0], 1., .06) 
equal(pdfweight[0], 0., .0001) 
equal(sweight_spin[0], 1.14, .06) 
assert ((Li_wzweight - [.12, .94]).round(2) == 0).all()
assert ((H_wzweight - [.88, .06]).round(2) == 0).all()
assert ((Li_wzweight + H_wzweight).round(5) == 1).all()
equal(n_a.sum(), 0., 1e-5)
equal(n_a[1], .764, .001)

#               HOMO    s   py  pz  px  *s
Li_orbitalweight[0] -= [.4, .0, .6, .0, .0]
H_orbitalweight[0]  -= [.7, .0, .0, .0, .0]

#              LUMO       s  py   pz  px  *s
Li_orbitalweight[1] -= [1.6, .0, 1.1, .0, .0]
H_orbitalweight[1]  -= [0.1, .0, 0.0, .0, .0]

assert not Li_orbitalweight.round(1).any()
assert not H_orbitalweight.round(1).any()
