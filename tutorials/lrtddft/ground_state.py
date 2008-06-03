from gpaw import *
from ase import *
from gpaw.tddft import *


# Sodium dimer, Na2
d = 1.55
atoms = Atoms( symbols='Na2',
               positions=[( 0, 0, d),
                          ( 0, 0,-d)],
               pbc=False)


atoms.center(vacuum=4.0) # excited states might need more vacuum
calc = Calculator(nbands=1, h=0.30, txt='Na2_gs.txt')
atoms.set_calculator(calc)
e = atoms.get_potential_energy()

# Calculate also unoccupied states with the fixed density
calc.set(nbands=20, convergence={'bands': 'all'}, 
         eigensolver='cg', # unoccupied states converge often better with cg
         fixdensity=True)
e = atoms.get_potential_energy()
# write the wave functions to a file
calc.write('na2_gs.gpw', 'all')

