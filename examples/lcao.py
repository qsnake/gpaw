#!/usr/bin/env python
from ase.data.molecules import molecule
from gpaw import GPAW

# First generate basis functions, and put them in your setup directory
# $ gpaw-basis --type dzp C H

calc = GPAW(eigensolver='lcao', basis='dzp', nbands=6)
atoms = molecule('CH4')
atoms.center(vacuum=3.5)
atoms.set_calculator(calc)
atoms.get_potential_energy()
