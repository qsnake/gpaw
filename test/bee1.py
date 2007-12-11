from ASE import Atom, ListOfAtoms
from gpaw import Calculator
from gpaw.xc_functional import XCFunctional

a = 4.0
atoms = ListOfAtoms([Atom('H')], cell=(a, a, a), periodic=True)
calc = Calculator(txt=None)
atoms.SetCalculator(calc)
atoms.GetPotentialEnergy()
e1 = calc.get_xc_difference(XCFunctional('BEE1', parameters=[-100.0]))
e2 = calc.get_xc_difference('X-C_PBE')
print e1, e2
assert abs(e1 - e2) < 2e-5
e1 = calc.get_xc_difference(XCFunctional('BEE1', parameters=[0.0]))
e2 = calc.get_xc_difference('PBE')
print e1, e2
assert abs(e1 - e2) < 3e-6
