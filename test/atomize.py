from ASE import Atom, ListOfAtoms
from gpaw import Calculator

a = 4.  # Size of unit cell (Angstrom)
c = a / 2
# Hydrogen atom:
atom = ListOfAtoms([Atom('H', (c, c, c), magmom=1)],
                   cell=(a, a, a), periodic=False)

# gpaw calculator:
calc = Calculator(h=0.2, nbands=1, xc='PBE', txt='H.txt')
atom.SetCalculator(calc)

e1 = atom.GetPotentialEnergy()

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = ListOfAtoms([Atom('H', (c - d / 2, c, c)),
                        Atom('H', (c + d / 2, c, c))],
                       cell=(a, a, a), periodic=False)

calc.set(txt='H2.txt')
molecule.SetCalculator(calc)
e2 = molecule.GetPotentialEnergy()

print 'hydrogen atom energy:     %5.2f eV' % e1
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'atomization energy:       %5.2f eV' % (2 * e1 - e2)

import os
os.remove('H.txt')
os.remove('H2.txt')
