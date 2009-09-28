from ase import *
from gpaw import *
from gpaw.lcao.tools import get_lcao_hamiltonian
from gpaw.mpi import rank, MASTER
import cPickle as pickle

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

if rank == MASTER:
    eigs2 = np.linalg.eigvals(np.linalg.solve(S_kMM[2], H_skMM[0, 2])).real
    eigs2.sort()
    assert abs(sum(eigs - eigs2)) < 1e-8
