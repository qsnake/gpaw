import os
from ASE import Crystal, Atom
from ASE.Units import units
from gpaw.utilities import equal
from gpaw import Calculator


units.SetUnits('Bohr', 'Hartree')

a = 7.5
n = 16
atoms = Crystal([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a))
calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
atoms.SetCalculator(calc)
e1 = atoms.GetPotentialEnergy()
e1ref = calc.get_reference_energy()
de12 = calc.get_xc_difference('revPBE')
calc.set(xc='revPBE')
e2 = atoms.GetPotentialEnergy()
e2ref = calc.get_reference_energy()
de21 = calc.get_xc_difference('PBE')
print e1ref + e1 + de12, e2ref + e2
print e1ref + e1, e2ref + e2 + de21
print de12, de21
equal(e1ref + e1 + de12, e2ref + e2, 18e-5)
equal(e1ref + e1, e2ref + e2 + de21, 18e-5)

calc.write('revPBE.gpw')

de21b = Calculator('revPBE.gpw').get_xc_difference('PBE')
equal(de21, de21b, 9e-8)

os.remove('revPBE.gpw')
