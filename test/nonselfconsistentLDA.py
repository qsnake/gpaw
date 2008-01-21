import os
from ase import *
from gpaw.utilities import equal
from gpaw import Calculator

a = 7.5 * Bohr
n = 16
atoms = Atoms([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a), pbc=True)
calc = Calculator(gpts=(n, n, n), nbands=1, xc='LDA')
atoms.set_calculator(calc)
e1 = atoms.get_potential_energy()
e1ref = calc.get_reference_energy()
de12 = calc.get_xc_difference('PBE')
calc.set(xc='PBE')
e2 = atoms.get_potential_energy()
e2ref = calc.get_reference_energy()
de21 = calc.get_xc_difference('LDA')
print e1ref + e1 + de12, e2ref + e2
print e1ref + e1, e2ref + e2 + de21
print de12, de21
equal(e1ref + e1 + de12, e2ref + e2, 56e-5 * 27)
equal(e1ref + e1, e2ref + e2 + de21, 93e-5 * 27)

calc.write('PBE.gpw')

de21b = Calculator('PBE.gpw').get_xc_difference('LDA')
equal(de21, de21b, 9e-8)

os.remove('PBE.gpw')
