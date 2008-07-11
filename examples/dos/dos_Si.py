import pylab as p
from ase import *
from gpaw import Calculator

# Do the ground-state calculation
a = 5.475
atoms = Atoms(symbols='Si4', pbc=True,
              positions=[(.0, .0, .0),
                         (.5, .5, .5),
                         (.0, .5, .75),
                         (.5, .0, .25),])
atoms.set_cell([a / sqrt(2), a / sqrt(2), a], scale_atoms=True)
calc = Calculator(kpts=(4, 4, 2), txt='si.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('si.gpw')

# Plot the band structure
p.plot(*calc.get_dos(width=.2))
p.axis('tight')
p.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
p.ylabel('Density of States (1/eV)')
p.show()
