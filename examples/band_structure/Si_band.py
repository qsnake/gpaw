from gpaw import Calculator
from ase import *

a = 5.475
atoms = Atoms(symbols='Si4', pbc=True,
              positions=[(.0, .0, .0),
                         (.5, .5, .5),
                         (.0, .5, .75),
                         (.5, .0, .25),])
atoms.set_cell([a / sqrt(2), a / sqrt(2), a], scale_atoms=True)

# Make self-consistent calculation and save results
calc = Calculator(h=0.25, kpts=(6, 6, 6), width=0.05,
                  nbands=10, txt='Si_sc.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('Si_sc.gpw')

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 50
kpts = [(0.5 * k / nkpt, 0, 0) for k in range(nkpt + 1)]
calc = Calculator('Si_sc.gpw', txt='Si_harris.txt',
                  kpts=kpts, fixdensity=True, nbands=16,
                  eigensolver='cg', convergence={'bands': 'all'})
calc.get_potential_energy()
calc.write('Si_harris.gpw')
