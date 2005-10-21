from ASE import Crystal, Atom
from ASE.Units import units
from gridpaw.utilities import equal
from gridpaw import Calculator


units.SetEnergyUnit('Hartree')
a = 5.0
n = 24
li = Crystal([Atom('Li', (0.0, 0.0, 0.0), magmom=1.0)], cell=(a, a, a))

calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
li.SetCalculator(calc)
e = li.GetPotentialEnergy() + calc.GetReferenceEnergy()
equal(e, -7.462, 0.056)

calc.Set(xc='revPBE')
erev = li.GetPotentialEnergy() + calc.GetReferenceEnergy()
equal(erev, -7.487, 0.057)
equal(e - erev, 0.025, 0.002)
