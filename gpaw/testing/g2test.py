import traceback

from gpaw import Calculator
from gpaw.mpi import rank, MASTER
from gpaw.testing.g2 import get_g2, atoms
from gpaw.testing.atomization_data import atomization_vasp
from ase.parallel import paropen
from sys import stderr

cell = [12., 13., 14.]
data = paropen('g2data.txt', 'w')
systems = atoms.keys() + atomization_vasp.keys()

for formula in systems:
    loa = get_g2(formula, cell)
    calc = Calculator(h=.18,
                      nbands=-5,
                      xc='PBE',
                      fixmom=True,
                     #setups={'C': 'new', 'S': 'new', 'Li': 'new', 'P': 'new'},
                      txt=formula + '.txt')
    if len(loa) == 1:
        calc.set(hund=True)
    loa.SetCalculator(calc)
    if formula == 'BeH':
        calc.initialize()
        calc.nuclei[0].f_si = [(1, 0, 0.5, 0),
                               (0.5, 0, 0, 0)]
    try:
        energy = loa.GetPotentialEnergy()
    except:
        print >>data, formula, 'Error'
        if rank == MASTER:
            print >>stderr, 'Error in', formula
            traceback.print_exc(file=stderr)
    else:
        print >>data, formula, repr(energy)
        calc.write(formula + '.gpw', mode='all')
        data.flush()
