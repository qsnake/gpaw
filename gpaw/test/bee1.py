from ase import Atom, Atoms
from gpaw import GPAW
from gpaw.xc import XC

a = 4.0
atoms = Atoms([Atom('H')], cell=(a, a, a), pbc=True)
calc = GPAW(txt=None)
atoms.set_calculator(calc)
atoms.get_potential_energy()
e1 = calc.get_xc_difference(XC('BEE1', [-100.0]))
e2 = calc.get_xc_difference('LDA_X+GGA_C_PBE')
print e1, e2
assert abs(e1 - e2) < 3e-5
e1 = calc.get_xc_difference(XC('BEE1', [0.0]))
e2 = calc.get_xc_difference('PBE')
print e1, e2
assert abs(e1 - e2) < 3e-6
e1 = calc.get_xc_difference(XC('BEE1', [[42], [0.0]]))
e2 = calc.get_xc_difference('GGA_C_PBE')
print e1, e2
assert abs(e1 - e2) < 3e-6
