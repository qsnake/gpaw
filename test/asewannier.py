from ase import *
from gpaw import GPAW
from gpaw.utilities import equal
from ase.dft import Wannier

# Test of ase wannier using gpaw

calc = GPAW(nbands=4)
atoms = molecule('H2', calculator=calc)
atoms.center(vacuum=3.)
atoms.get_potential_energy()

pos = atoms.positions + np.array([[0, 0, .2339], [0, 0, -.2339]])
com = atoms.get_center_of_mass()

wan = Wannier(nwannier=2, calc=calc, initialwannier='bloch')
equal(wan.get_functional_value(), 2.964, 1e-3)
equal(np.linalg.norm(wan.get_centers() - [com, com]), 0, 1e-4)

wan = Wannier(nwannier=2, calc=calc, initialwannier='projectors')
equal(wan.get_functional_value(), 3.099, 1e-3)
equal(np.linalg.norm(wan.get_centers() - pos), 0, 1e-4)

wan = Wannier(nwannier=2, calc=calc, initialwannier=[[0, 0, .5], [1, 0, .5]])
equal(wan.get_functional_value(), 3.099, 1e-3)
equal(np.linalg.norm(wan.get_centers() - pos), 0, 1e-4)

wan.localize()
equal(wan.get_functional_value(), 3.099, 1e-3)
equal(np.linalg.norm(wan.get_centers() - pos), 0, 1e-4)
equal(np.linalg.norm(wan.get_radii() - 1.2396), 0, 1e-4)
eig = np.sort(np.linalg.eigvals(wan.get_hamiltonian().real))
equal(np.linalg.norm(eig - calc.get_eigenvalues()[:2]), 0, 1e-4)
