import numpy as np

from ase import Atoms
from gpaw import GPAW
from gpaw.utilities import equal

calc = GPAW(nbands=3, txt=None)
atoms = Atoms('H', pbc=True, calculator=calc)
atoms.center(vacuum=3)

calc.set(width=0)
atoms.get_potential_energy()
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
atoms.get_potential_energy()
ef = calc.get_fermi_level()
equal(ef, -6.34843, .01)
calc.write('test.gpw')
equal(GPAW('test.gpw', txt=None).get_fermi_level(), ef, 1e-8)
