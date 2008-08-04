import traceback

from gpaw import Calculator
from gpaw.mpi import rank, MASTER
from ase.parallel import paropen
from ase.data.molecules import atoms, g1, molecule
from sys import stderr

cell = [12., 13., 14.]
data = paropen('g1data.txt', 'w')
systems = atoms + g1

for formula in systems:
    loa = molecule(formula)
    loa.set_cell(cell)
    loa.center()
    calc = Calculator(h=.18,
                      nbands=-5,
                      xc='PBE',
                      fixmom=True,
                      txt=formula + '.txt')
    if len(loa) == 1:
        calc.set(hund=True)
    loa.set_calculator(calc)
    if formula == 'BeH':
        calc.initialize(loa)
        calc.nuclei[0].f_si = [(1, 0, 0.5, 0),
                               (0.5, 0, 0, 0)]
    try:
        energy = loa.get_potential_energy()
    except:
        print >>data, formula, 'Error'
        if rank == MASTER:
            print >>stderr, 'Error in', formula
            traceback.print_exc(file=stderr)
    else:
        print >>data, formula, repr(energy)
        calc.write(formula + '.gpw')#, mode='all')
        data.flush()
