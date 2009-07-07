import sys
sys.argv.append('--gpaw=usenewlfc=True')

from ase import *
from gpaw import *
from gpaw.utilities import equal

a = 5.475
atoms = Atoms(symbols='Si2', pbc=True,
              cell=.5 * a * np.array([(1, 1, 0),
                                      (1, 0, 1),
                                      (0, 1, 1)]),
              scaled_positions=[(.00, .00, .00),
                                (.25, .25, .25)],
              calculator=GPAW(h=.25, kpts=(4, 4, 4), width=.1, nbands=5))
E = atoms.get_potential_energy()
equal(E, -12.03, 0.01)
