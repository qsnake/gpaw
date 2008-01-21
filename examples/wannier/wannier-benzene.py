from ase import *
from gpaw import Calculator
from gpaw.wannier import Wannier

calc = Calculator('benzene.gpw')
atoms = calc.get_atoms()

homo = calc.get_pseudo_wave_function(band=14)
write('homo.cube', atoms, data=homo)
write('homo.plt', atoms, data=homo)

# Initialize the Wannier class
w = Wannier(calc)
w.localize()
centers = w.get_centers()
view(atoms + Atoms(symbols='X15', positions=centers))

# Find the index of the center with the lowest y-coordinate:
nsigma = centers[:, 1].argmin()
sigma = w.get_function(calc, nsigma)

write('benzene.xyz', atoms)
write('sigma.cube', atoms, data=sigma)
write('sigma.plt', atoms, data=sigma)
