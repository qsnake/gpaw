from ase import *
from gpaw import GPAW
from gpaw.utilities import equal

a = 4.8 # => N = 4.8 / 0.2 = 24
loa = Atoms([Atom('C', [a / 2 + .3, a / 2 -.1, a / 2], magmom=2)],
                  pbc=False,
                  cell=(a, a, a))
p = []
exx = []
i = 0
for hosts in [1, 4]:
    calc = GPAW(convergence={'eigenstates': 1e-6}, hosts=hosts,
                      txt='exx_parallel.txt')

    loa.set_calculator(calc)
    p.append(loa.get_potential_energy())
    exx.append(calc.get_exact_exchange())

    print 'number of CPUs   :', hosts
    print 'Potential energy :', p[i]
    print 'Exchange energy  :', exx[i]
    print ''
    
    i += 1

for i in range(1, len(exx)):
    equal(p[i], p[0], 1e-2)
    equal(exx[i], exx[0], 1e-2)

## number of CPUs   : 1
## Potential energy : -1.07206007502
## Exchange energy  : -137.443595686

## number of CPUs   : 2
## Potential energy : -1.07206007502
## Exchange energy  : -137.443595686

## number of CPUs   : 3
## Potential energy : -1.0721486372
## Exchange energy  : -137.405085235

## number of CPUs   : 4
## Potential energy : -1.07194681834
## Exchange energy  : -137.441377715

