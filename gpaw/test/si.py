from gpaw import GPAW, restart
from ase import *
from ase.calculators import numeric_force
from gpaw.test import equal

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
e1 = bulk.get_potential_energy()
niter1 = calc.get_number_of_iterations()
eigs = calc.get_eigenvalues(kpt=0)
calc.write('temp.gpw')
del bulk
del calc

bulk, calc = restart('temp.gpw', fixdensity=True)
#calc.scf.reset()
e2 = bulk.get_potential_energy()
try: # number of iterations needed in restart
    niter2 = calc.get_number_of_iterations()
except: pass
eigs2 = calc.get_eigenvalues(kpt=0)
print 'Orginal', eigs
print 'Fixdensity', eigs2
print 'Difference', eigs2-eigs

assert np.fabs(eigs2 - eigs)[:-1].max() < 3e-5

energy_tolerance = 0.0005
niter_tolerance = 0
equal(e1, -36.7229861549, energy_tolerance) # svnversion 5252
#equal(niter1, 23, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 23 <= niter1 <= 25, niter1
equal(e2, -36.7229861549, energy_tolerance) # svnversion 5252
