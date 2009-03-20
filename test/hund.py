from ase import *
from gpaw import *

atoms = Atoms('H')
atoms.center(vacuum=4.)

params = dict(h=.3, convergence=dict(density=.005, eigenstates=1e-6))

# No magmom, no hund
atoms.set_calculator(GPAW(**params))
E_nom_noh = atoms.get_potential_energy()
assert np.all(atoms.get_magnetic_moments() == 0.)
assert atoms.calc.get_number_of_spins() == 1

# No magmom, hund
atoms.set_calculator(GPAW(hund=True, **params))
E_nom_h = atoms.get_potential_energy()
assert np.all(atoms.get_magnetic_moments() == 1.)
assert atoms.calc.get_number_of_spins() == 2

atoms.set_initial_magnetic_moments([1.])

# magmom, no hund
atoms.set_calculator(GPAW(fixmom=True, **params))
E_m_noh = atoms.get_potential_energy()
assert np.all(atoms.get_magnetic_moments() == 1.)
assert atoms.calc.get_number_of_spins() == 2

# magmom, hund
atoms.set_calculator(GPAW(hund=True, **params))
E_m_h = atoms.get_potential_energy()
assert np.all(atoms.get_magnetic_moments() == 1.)
assert atoms.calc.get_number_of_spins() == 2

print E_nom_noh
print E_nom_h
print E_m_noh
print E_m_h

assert E_m_h == E_m_noh == E_nom_h
