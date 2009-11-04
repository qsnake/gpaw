from ase import *
from gpaw import GPAW
from gpaw.test import equal

a = 5.0
d = 1.0
x = d / 3**0.5
atoms = Atoms([Atom('C', (0.0, 0.0, 0.0)),
                     Atom('H', (x, x, x)),
                     Atom('H', (-x, -x, x)),
                     Atom('H', (x, -x, -x)),
                     Atom('H', (-x, x, -x))],
                    cell=(a, a, a),
                    pbc=False)

atoms.positions[:] += a / 2
calc = GPAW(h=0.25, nbands=4, convergence={'eigenstates': 1e-11})
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
niter = calc.get_number_of_iterations()

# The three eigenvalues e[1], e[2], and e[3] must be degenerate:
e = atoms.get_calculator().wfs.kpt_u[0].eps_n
print e[1] - e[3]
equal(e[1], e[3], 9.3e-8)

energy_tolerance = 0.0003
niter_tolerance = 0
equal(energy, -23.76976642, energy_tolerance) # svnversion 5252
#equal(niter, 42, niter_tolerance) # svnversion 5252 # niter depends on the number of processes
assert 39 <= niter <= 42, niter
