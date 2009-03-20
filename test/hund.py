from ase import *
from gpaw import *

atoms = Atoms('H')
atoms.center(vacuum=4.)

# No magmom, no hund
atoms.set_calculator(GPAW())
E_nom_noh = atoms.get_potential_energy()

# No magmom, hund
atoms.set_calculator(GPAW(hund=True))
E_nom_h = atoms.get_potential_energy()

atoms.set_initial_magnetic_moments([1.])

# magmom, no hund
atoms.set_calculator(GPAW())
E_m_noh = atoms.get_potential_energy()

# magmom, hund
atoms.set_calculator(GPAW(hund=True))
E_m_h = atoms.get_potential_energy()
