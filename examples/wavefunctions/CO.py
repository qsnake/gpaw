from ase import *
from gpaw import *

# Gpaw calculator with 4 unonccupied bands (CO has 10 valence electrons)
calc = Calculator(h=0.2, txt='CO.txt', nbands=-4)

# Make the CO molecule and relax the structure
CO = molecule('CO')
CO.center(vacuum=3.)
CO.set_calculator(calc)
QuasiNewton(CO, trajectory='CO.traj').run(fmax=0.05)

# Write wave functions to gpw file
calc.write('CO.gpw', mode='all')

# Generate cube-files of the orbitals.
for n in range(calc.get_number_of_bands()):
  wf = calc.get_pseudo_wave_function(band=n)
  write('CO%d.cube' % n, CO, data=wf)
