from ASE import Crystal, Atom
from ASE.Units import units
from gpaw.utilities import equal
from gpaw import Calculator


units.SetEnergyUnit('Hartree')
a = 5.0
n = 24
li = Crystal([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a))

calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
li.SetCalculator(calc)
e = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(e, -7.462, 0.056)

calc.set(xc='revPBE')
erev = li.GetPotentialEnergy() + calc.get_reference_energy()
equal(erev, -7.487, 0.057)
equal(e - erev, 0.025, 0.002)
