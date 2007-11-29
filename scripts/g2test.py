from gpaw import Calculator
from gpaw.testing.g2 import get_g2, atoms
from gpaw.testing.atomization_data import atomization_vasp
from ASE.Utilities.Parallel import paropen

cell = [12., 13., 14.]
data = paropen('g2data.txt', 'w')
systems = atoms.keys() + atomization_vasp.keys()

for formula in systems:
    loa = get_g2(formula, cell)
    calc = Calculator(h=.18,
                      nbands=-5,
                      xc='PBE',
                      fixmom=True,
                      txt=formula + '.txt')
    if len(loa) == 1:
        calc.set(hund=True)
    loa.SetCalculator(calc)
    try:
        energy = loa.GetPotentialEnergy()
    except:
        print >>data, formula, 'Error'
    else:
        print >>data, formula, repr(energy)
        calc.write(formula + '.gpw', mode='all')
