from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.xc_functional import XCFunctional

a = 4.0
atoms = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
calc = GPAW(txt=None)
atoms.set_calculator(calc)
atoms.get_potential_energy()
e1 = calc.get_xc_difference(XCFunctional('BEE1', parameters=[-100.0]))
e2 = calc.get_xc_difference('X-C_PBE')
print e1, e2
assert abs(e1 - e2) < 3e-5
e1 = calc.get_xc_difference(XCFunctional('BEE1', parameters=[0.0]))
e2 = calc.get_xc_difference('PBE')
print e1, e2
assert abs(e1 - e2) < 3e-6
e1 = calc.get_xc_difference(XCFunctional('BEE1', parameters=[[42], [0.0]]))
e2 = calc.get_xc_difference('None-C_PBE')
print e1, e2
assert abs(e1 - e2) < 3e-6
