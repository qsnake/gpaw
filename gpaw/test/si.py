from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.utilities import equal

a = 5.404
bulk = Atoms(symbols='Si8',
             scaled_positions=[(0, 0, 0),
                               (0, 0.5, 0.5),
                               (0.5, 0, 0.5),
                               (0.5, 0.5, 0),
                               (0.25, 0.25, 0.25),
                               (0.25, 0.75, 0.75),
                               (0.75, 0.25, 0.75),
                               (0.75, 0.75, 0.25)],
             pbc=True, cell=(a, a, a))
n = 20
calc = GPAW(gpts=(n, n, n),
            nbands=8*3,
            width=0.01,
            kpts=(1, 1, 1))
bulk.set_calculator(calc)
bulk.get_potential_energy()
eigs = calc.get_eigenvalues(kpt=0)
calc.write('temp.gpw')
del bulk
del calc

bulk, calc = restart('temp.gpw', fixdensity=True)
#calc.scf.reset()
bulk.get_potential_energy()
eigs2 = calc.get_eigenvalues(kpt=0)
print 'Orginal', eigs
print 'Fixdensity', eigs2
print 'Difference', eigs2-eigs

assert np.fabs(eigs2 - eigs)[:-1].max() < 3e-5
