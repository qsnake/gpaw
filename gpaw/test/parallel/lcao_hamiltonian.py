import cPickle as pickle
from ase import *
from gpaw import GPAW, restart, setup_paths
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import world
from gpaw.atom.basis import BasisMaker

if world.rank == 0:
    basis = BasisMaker('Li', 'szp').generate(1, 1)
    basis.write_xml()
world.barrier()
if '.' not in setup_paths:
    setup_paths.append('.')
    
if 1:
    a = 2.7
    bulk = Atoms('Li', pbc=True, cell=[a, a, a])
    calc = GPAW(gpts=(8, 8, 8), kpts=(4, 4, 4), mode='lcao', basis='szp')
    bulk.set_calculator(calc)
    bulk.get_potential_energy()
    calc.write('temp.gpw')

atoms, calc = restart('temp.gpw')
H_skMM, S_kMM = get_lcao_hamiltonian(calc)
eigs = calc.get_eigenvalues(kpt=2)

if world.rank == 0:
    eigs2 = np.linalg.eigvals(np.linalg.solve(S_kMM[2], H_skMM[0, 2])).real
    eigs2.sort()
    assert abs(sum(eigs - eigs2)) < 1e-8
