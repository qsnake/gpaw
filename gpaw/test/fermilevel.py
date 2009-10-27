import numpy as np

from ase import Atoms
from gpaw import GPAW
from gpaw.test import equal

calc = GPAW(nbands=3, txt=None)
atoms = Atoms('H', pbc=True, calculator=calc)
atoms.center(vacuum=3)

calc.set(width=0)
e0 = atoms.get_potential_energy()
niter0 = calc.get_number_of_iterations()
try:
    calc.get_fermi_level()
except NotImplementedError:
    pass # It *should* raise an error
else:
    raise RuntimeError, 'get_fermi_level should not be possible for width=0'
homo, lumo = calc.get_homo_lumo()
equal(homo, -6.34849, .01)
equal(lumo, -0.39322, .01)
calc.write('test.gpw')
assert np.all(GPAW('test.gpw', txt=None).get_homo_lumo() == (homo, lumo))

calc.set(width=0.1)
e1 = atoms.get_potential_energy()
niter1 = calc.get_number_of_iterations()
ef = calc.get_fermi_level()
equal(ef, -6.34843, .01)
calc.write('test.gpw')
equal(GPAW('test.gpw', txt=None).get_fermi_level(), ef, 1e-8)

energy_tolerance = 0.00005
niter_tolerance = 0
equal(e0, -0.0280845579468, energy_tolerance) # svnversion 5252
#equal(niter0, 22, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 17 <= niter0 <= 22, niter0
equal(e1, -0.0973921063256, energy_tolerance) # svnversion 5252
equal(niter1, 4, niter_tolerance) # svnversion 5252
