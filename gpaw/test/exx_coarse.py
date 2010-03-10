import sys

from ase import Atoms, Atom

from gpaw import GPAW
from gpaw.test import equal
from gpaw.utilities.timing import Timer

timer = Timer()

loa = Atoms([Atom('Be', (0, 0, 0)), Atom('Be', (2.45, 0, 0))],
            cell= [5.9, 4.8, 5.0])
loa.center()

fgl = [False, True]
#fgl = [True, False]

txt='-'
txt='/dev/null'

E = {}
niter = {}
for fg in fgl:
    if fg:
        tstr = 'Exx on fine grid'
    else:
        tstr = 'Exx on coarse grid'
    timer.start(tstr)
    calc = GPAW(h = .3, xc='PBE',
                      nbands=4,
                      convergence={'eigenstates': 1e-4},
                      charge=-1)
    loa.set_calculator(calc)
    E[fg] = loa.get_potential_energy()
    calc.set(xc={'name':'PBE0', 'finegrid': fg})
    E[fg] = loa.get_potential_energy()
    niter[fg] = calc.get_number_of_iterations()
    timer.stop(tstr)

timer.write(sys.stdout)

print 'Total energy on the fine grid   =', E[True]
print 'Total energy on the coarse grid =', E[False]
equal(E[True], E[False], 0.02)

energy_tolerance = 0.0003
niter_tolerance = 0
equal(E[False], -788.67538, energy_tolerance)
assert 15 <= niter[False] <= 20, niter[False]
equal(E[True], -788.68187, energy_tolerance)
assert 16 <= niter[True] <= 22, niter[True]
