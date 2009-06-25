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
de12t = calc.get_xc_difference('TPSS')
de12m = calc.get_xc_difference('M06L')

# Hydrogen molecule:
d = 0.74  # Experimental bond length
molecule = Atoms([Atom('H', (c - d / 2, c, c)),
                        Atom('H', (c + d / 2, c, c))],
                       cell=(a, a, a), pbc=False)

calc.set(txt='H2.txt')
molecule.set_calculator(calc)
e2 = molecule.get_potential_energy()
de22t = calc.get_xc_difference('TPSS')
de22m = calc.get_xc_difference('M06L')

print 'hydrogen atom energy:     %5.2f eV' % e1
print 'hydrogen molecule energy: %5.2f eV' % e2
print 'atomization energy:       %5.2f eV' % (2 * e1 - e2)
print 'atomization energy  tpss: %5.2f eV' % (2 * (e1+de12t) - (e2+de22t))
print 'atomization energy  m06l: %5.2f eV' % (2 * (e1+de12m) - (e2+de22m))
PBETPSSdifference = (2 * e1 - e2)-(2 * (e1+de12t) - (e2+de22t))
PBEM06Ldifference = (2 * e1 - e2)-(2 * (e1+de12m) - (e2+de22m))
print PBETPSSdifference 
print PBEM06Ldifference 
# TPSS value is from JCP 120 (15) 6898, 2004
assert abs(PBETPSSdifference + 0.3599)  < 0.002
assert abs(PBEM06Ldifference + 0.169)  < 0.002
