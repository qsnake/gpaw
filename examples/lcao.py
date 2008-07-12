from ase import *
from gpaw import *

# First generate basis functions, and put them in your setup directory
# $ gpaw-basis --type dzp C H

calc = Calculator(eigensolver='lcao', basis='dzp', nbands=6)
atoms = molecule('CH4')
atoms.center(vacuum=3.5)
atoms.set_calculator(calc)
atoms.get_potential_energy()
