from ase import *
from gpaw import *
from gpaw.test import equal

a = 5.475
atoms = Atoms(symbols='Si2', pbc=True,
              cell=.5 * a * np.array([(1, 1, 0),
                                      (1, 0, 1),
                                      (0, 1, 1)]),
              scaled_positions=[(.00, .00, .00),
                                (.25, .25, .25)],
              calculator=GPAW(h=.24, kpts=(4, 4, 4), width=.1, nbands=5))
E = atoms.get_potential_energy()

equal(E, -12.0736, 0.005)
equal(atoms.calc.get_fermi_level(), 5.16301, 0.005)
assert 22 <= atoms.calc.iter <= 26
