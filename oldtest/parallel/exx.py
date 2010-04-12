from ase import *
from gpaw import Calculator
from gpaw.utilities import equal

a = 4.8 # => N = 4.8 / 0.2 = 24
loa = Atoms([Atom('C', [a / 2 + .3, a / 2 -.1, a / 2], magmom=2)],
            pbc=False,
            cell=(a, a, a))
p = []
exx = []
i = 0
calc = Calculator(convergence={'eigenstates': 1e-6}, hund=1,
                  txt='exx_parallel.txt')

loa.set_calculator(calc)
p.append(loa.get_potential_energy())
exx.append(calc.get_exact_exchange())

print 'Potential energy :', p[i]
print 'Exchange energy  :', exx[i]
print ''



## number of CPUs   : 1
## Potential energy : -1.04946572514
## Exchange energy  : -5.05252232093

## number of CPUs   : 2
## Potential energy : -1.04946572514
## Exchange energy  : -5.05252232093

## number of CPUs   : 3
## Potential energy : -1.04948210572
## Exchange energy  : -5.0525195586

## number of CPUs   : 4
## Potential energy : -1.04946572514
## Exchange energy  : -5.05252232093

