from ASE import Crystal, Atom
from ASE.Units import units
from gridpaw.utilities import equal
from gridpaw import Calculator


units.SetUnits('Bohr', 'Hartree')

a = 7.5
n = 16
atoms = Crystal([Atom('He', (0.0, 0.0, 0.0))], cell=(a, a, a))
calc = Calculator(gpts=(n, n, n), nbands=1, xc='PBE')
atoms.SetCalculator(calc)
e1 = atoms.GetPotentialEnergy()
e1a = calc.GetReferenceEnergy()
calc.Set(xc='revPBE')
e2 = atoms.GetPotentialEnergy()
e2a = calc.GetReferenceEnergy()

equal(e1a, -2.893, 27e-5)
equal(e2a, -2.908, 40e-5)
equal(e1, e2, 29e-5)
