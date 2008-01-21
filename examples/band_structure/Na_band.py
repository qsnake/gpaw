from gpaw import Calculator
from ase import *

a = 4.23
atoms = Atoms([Atom('Na', (0, 0, 0)),
               Atom('Na', (0.5, 0.5, 0.5))],
              pbc=True)
atoms.set_cell((a, a, a), fix=False)

if 1:
    # Make self-consistent calculation and save results
    calc = Calculator(h=0.25, kpts=(8, 8, 8), width=0.05,
                      nbands=3, txt='Na_sc.txt')
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('Na_sc.gpw')

# Calculate band structure along Gamma-X i.e. from 0 to 0.5
nkpt = 50
kpts = [(0.5 * k / nkpt, 0, 0) for k in range(nkpt + 1)]
calc = Calculator('Na_sc.gpw', txt='Na_harris.txt',
                  kpts=kpts, fixdensity=True, nbands=7,
                  eigensolver='cg',
                  convergence={'bands': 'all'})
calc.get_potential_energy()
calc.write('Na_harris.gpw')
