from ase import *
from gpaw import Calculator
from gpaw.wannier import Wannier

calc = Calculator('h2o.gpw')
atoms = calc.get_atoms()

# Initialize the Wannier class
w = Wannier(calc)
w.localize()
centers = w.get_centers()
view(atoms + Atoms(symbols='X4', positions=centers))
