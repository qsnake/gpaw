from ase import *
from gpaw import *
from gpaw.test import equal

a = 5.475
calc = GPAW(h=0.24,
            kpts=(4, 4, 4),
            occupations=FermiDirac(width=0.1),
            nbands=5)
atoms = Atoms(symbols='Si2', pbc=True,
              cell=0.5 * a * np.array([(1, 1, 0),
                                      (1, 0, 1),
                                      (0, 1, 1)]),
              scaled_positions=[(0.0, 0.0, 0.0),
                                (0.25, 0.25, 0.25)],
              calculator=calc)
E = atoms.get_potential_energy()
niter = calc.get_number_of_iterations()

equal(E, -12.0736, 0.005)
equal(atoms.calc.get_fermi_level(), 5.16301, 0.005)

energy_tolerance = 0.0001
niter_tolerance = 0
equal(E, -12.0735247139, energy_tolerance) # svnversion 5252
#equal(niter, 29, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 26 <= niter <= 28, niter
