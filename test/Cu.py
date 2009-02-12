import os
from ase import *
from gpaw import GPAW
from gpaw.atom.generator import Generator, parameters
from gpaw import setup_paths
import gpaw.mpi as mpi

if mpi.rank == 0:
    # Generate non-scalar-relativistic setup for Cu:
    g = Generator('Cu', scalarrel=False, nofiles=True)
    g.run(logderiv=True, **parameters['Cu'])

mpi.world.barrier()

setup_paths.insert(0, '.')

a = 8.0
c = a / 2
Cu = Atoms('Cu', [(c, c, c)], magmoms=[1],
           cell=(a, a, a), pbc=0)

calc = GPAW(h=0.2, lmax=0)# basis='sz')
Cu.set_calculator(calc)
Cu.get_potential_energy()

e_4s_major = calc.get_eigenvalues(spin=0)[5] / Hartree
e_3d_minor = calc.get_eigenvalues(spin=1)[4] / Hartree
print mpi.rank, e_4s_major, e_3d_minor

#
# The reference values are from:
#
#   http://physics.nist.gov/PhysRefData/DFTdata/Tables/29Cu.html
#
if mpi.rank == 0:
    print e_4s_major - e_3d_minor, -0.184013 - -0.197109
    assert abs(e_4s_major - e_3d_minor - (-0.184013 - -0.197109)) < 0.001

del setup_paths[0]
