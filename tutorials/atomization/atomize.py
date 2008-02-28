from ase import *
from gpaw import Calculator

a = 4.  # Size of unit cell (Angstrom)
c = a / 2
# Hydrogen atom:
atom = Atoms([Atom('H', (c, c, c), magmom=1)],
                   cell=(a, a, a), pbc=False)

# gpaw calculator:
calc = Calculator(h=0.18, nbands=1, xc='PBE', txt='H.out')
atom.set_calculator(calc)

e1 = atom.get_potential_energy()
calc.write('H.gpw')

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = Atoms([Atom('H', (c - d / 2, c, c)),
                        Atom('H', (c + d / 2, c, c))],
                       cell=(a, a, a), pbc=False)

calc.set(txt='H2.out')
molecule.set_calculator(calc)
e2 = molecule.get_potential_energy()
calc.write('H2.gpw')

print 'hydrogen atom energy:     %5.2f eV' % e1
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'atomization energy:       %5.2f eV' % (2 * e1 - e2)
