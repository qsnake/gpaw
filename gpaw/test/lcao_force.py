# This tests calculates the force on the atoms of a small molecule.
#
# If the test fails, set the fd boolean below to enable a (costly) finite
# difference check.

import numpy as np
from ase.data.molecules import molecule
from gpaw import GPAW
from gpaw.atom.basis import BasisMaker

obasis = BasisMaker('O').generate(2, 1, energysplit=0.3, tailnorm=0.03**.5)
hbasis = BasisMaker('H').generate(2, 1, energysplit=0.3, tailnorm=0.03**.5)
basis = {'O' : obasis, 'H' : hbasis}

system = molecule('H2O')
system.center(vacuum=1.5)
system.rattle(stdev=.2, seed=42)
system.set_pbc(1)

calc = GPAW(h=0.2,
            mode='lcao',
            basis=basis,
            kpts=[(0., 0., 0.), (.3, .1, .4)],
            convergence={'density':1e-5, 'energy': 1e-6}
            )

system.set_calculator(calc)

F_ac = system.get_forces()


# Previous FD result, generated by disabled code below
F_ac_ref = np.array([[ 1.03815055,  1.64363717, -4.82334017],
                     [-0.70735722, -0.89238151,  3.0422243 ],
                     [-0.32680291, -0.72907592,  1.75320264]])

err_ac = np.abs(F_ac - F_ac_ref)
err = err_ac.max()

print 'Force'
print F_ac
print
print 'Reference result'
print F_ac_ref
print
print 'Error'
print err_ac
print
print 'Max error'
print err

# ASE uses dx = [+|-] 0.001 by default,
# error should be around 2e-3.  In fact 4e-3 would probably be acceptable
assert err < 3e-3

# Set boolean to run new FD check
fd = not not False

if fd:
    from ase.calculators.test import numeric_forces
    F_ac_fd = numeric_forces(system)
    print 'Self-consistent forces'
    print F_ac
    print 'FD'
    print F_ac_fd
    print repr(F_ac_fd)
    print F_ac - F_ac_fd, np.abs(F_ac - F_ac_fd).max()

    assert np.abs(F_ac - F_ac_fd).max() < 4e-3
