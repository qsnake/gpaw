from ase import *
from gpaw import GPAW

a = 6.  # Size of unit cell (Angstrom)
c = a / 2
# Hydrogen atom:
atom = Atoms([Atom('H', (c, c, c), magmom=1)],
                   cell=(a, a, a), pbc=False)

# gpaw calculator:
calc = GPAW(h=0.2, nbands=1, xc='PBE', txt='H.txt')
atom.set_calculator(calc)

e1 = atom.get_potential_energy()
e1tpss = e1 + calc.get_xc_difference('TPSS')

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = Atoms([Atom('H', (c - d / 2, c, c)),
                        Atom('H', (c + d / 2, c, c))],
                       cell=(a, a, a), pbc=False)

calc.set(txt='H2.txt')
molecule.set_calculator(calc)
e2 = molecule.get_potential_energy()
e2tpss = e2 + calc.get_xc_difference('TPSS')

print 'hydrogen atom energy:     %5.2f eV' % e1
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'atomization energy:       %5.2f eV' % (2 * e1 - e2)
print 'atomization energy:       %5.2f eV' % (2 * e1tpss - e2tpss)
