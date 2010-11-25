from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.xc import XC
from gpaw.xc.bee import BEEVDWFunctional

a = 4.0
atoms = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
calc = GPAW(txt=None)
atoms.set_calculator(calc)
atoms.get_potential_energy()
e1 = calc.get_xc_difference(XC('BEE1', [-100.0, 1.0]))
e2 = calc.get_xc_difference('LDA_X+GGA_C_PBE')
print e1, e2
assert abs(e1 - e2) < 3e-5

e1 = calc.get_xc_difference(XC('BEE1', [0.0, 1.0]))
e2 = calc.get_xc_difference('PBE')
e3 = calc.get_xc_difference(BEEVDWFunctional())
print e1, e2, e3
assert abs(e1 - e2) < 3e-6
assert abs(e1 - e3) < 3e-6

e1 = calc.get_xc_difference(XC('BEE1', [42, 0.0]))
e2 = calc.get_xc_difference('GGA_C_PBE')
print e1, e2
assert abs(e1 - e2) < 3e-6

e1 = calc.get_xc_difference('LDA')
e2 = calc.get_xc_difference(BEEVDWFunctional([-1000, 1], [1, 0, 0, 0]))
print e1, e2
assert abs(e1 - e2) < 3e-6
