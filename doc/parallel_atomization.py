"""This script calculates the atomization energy of nitrogen using two
processes, each process working on a separate system."""

from gpaw import Calculator, mpi
import Numeric as num
from ASE import ListOfAtoms, Atom

cell = (8., 8., 8.)
pos = 4.
separation = 1.103

rank = mpi.world.rank

# Master process calculates energy of N, while the other one takes N2
if rank == 0:
    system = ListOfAtoms([Atom('N', (pos, pos, pos), magmom=3)], cell=cell)
elif rank == 1:
    system = ListOfAtoms([Atom('N', (pos, pos, pos + separation/2.)),
                          Atom('N', (pos, pos, pos - separation/2.))],
                         cell=cell)
else:
    raise Exception('This example uses only two processes')

# Open different files depending on rank
output = '%d.txt' % rank

calc = Calculator(communicator=[rank], txt=output, xc='PBE')
system.SetCalculator(calc)
energy = system.GetPotentialEnergy()

# Now send the energy from the second process to the first process,
if rank == 1:
    # Communicators work with arrays from Numeric only:
    mpi.world.send(num.array([energy]), 0)
else:
    # The first process receives the number and prints the atomization energy
    container = num.array([0.])
    mpi.world.receive(container, 1)

    # Ea = E[molecule] - 2 * E[atom]
    atomization_energy = container[0] - 2 * energy
    print 'Energy:',atomization_energy
