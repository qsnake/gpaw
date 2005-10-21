from gridpaw import Calculator
from ASE import Crystal, Atom
from gridpaw.utilities import equal
from ASE.Units import Convert, units


def f(length, energy):
    # Don't do this at home:
    units.length_used = False 
    units.energy_used = False
    units.SetUnits(length, energy)
    a = Convert(8.0, 'Bohr', length)
    n = 16
    atom = Crystal([Atom('H', (0.0, 0.0, 0.0))], cell=(a, a, a))
    atom.SetCalculator(Calculator(gpts=(n, n, n), nbands=1, out=None))
    return Convert(atom.GetPotentialEnergy(), energy, 'Hartree')

e1 = f('Bohr', 'Hartree')
e2 = f('nm', 'eV')
e3 = f('Ang', 'Hartree')

equal(e1, e2, 5.0e-7)
equal(e1, e3, 5.0e-7)
