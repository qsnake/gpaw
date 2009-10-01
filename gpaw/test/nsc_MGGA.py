import os
from ase import *
from gpaw.test import equal
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw.xc_functional import XCFunctional
from gpaw import setup_paths
from gpaw.test import equal

# test with revision 4432

symbol = 'H'
g = Generator(symbol, 'TPSS', scalarrel=True, nofiles=True)

a = 6 * Bohr
n = 12
atoms = Atoms([Atom('H', (0.0, 0.0, 0.0),magmom=1)], cell=(a, a, a), pbc=True)
atoms.center(vacuum=3)
calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE', txt='Hnsc.txt')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()
e1ref = calc.get_reference_energy()
de12t = calc.get_xc_difference('TPSS')
de12m = calc.get_xc_difference('M06L')

print '================'
print 'e1 = ', e1
print 'de12t = ', de12t
print 'de12m = ', de12m
print 'tpss = ', e1+de12t
print 'm06l = ', e1+de12m
print '================'

equal(e1+de12t, 15.70495612949, 0.005)
equal(e1+de12m, 15.775609818, 0.005)

symbol = 'He'
g = Generator(symbol, 'TPSS', scalarrel=True, nofiles=True)

a = 6 * Bohr
n = 12
atomsHe = Atoms([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a), pbc=True)
atomsHe.center(vacuum=3)
calc = GPAW(gpts=(n, n, n), nbands=1, xc='PBE', txt='Hensc.txt')
atomsHe.set_calculator(calc)
e1He = atomsHe.get_potential_energy()
e1refHe = calc.get_reference_energy()
de12tHe = calc.get_xc_difference('TPSS')
de12mHe = calc.get_xc_difference('M06L')

print '================'
print 'e1He = ', e1He
print 'de12tHe = ', de12tHe
print 'de12mHe = ', de12mHe
print 'tpss = ', e1He+de12tHe
print 'm06l = ', e1He+de12mHe
print '================'

equal(e1He+de12tHe, 3.21494509843, 0.005)
equal(e1He+de12mHe, 2.60645239522, 0.005)
