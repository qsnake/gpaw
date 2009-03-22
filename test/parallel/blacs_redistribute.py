import numpy as np
from gpaw.mpi import world, rank, size
import _gpaw
A = np.arange(9, dtype=float).reshape(3, 3)
d12 = _gpaw.blacs_create(world.new_communicator(np.array([0, 1])),
                         3, 3, 1, 2, 3, 2)
d22 = _gpaw.blacs_create(world.new_communicator(np.array([0, 1, 2, 3])),
                         3, 3, 2, 2, 2, 2)
if rank == 0:
    C = _gpaw.scalapack_redist(A[:2], d12, d22)
elif rank == 1:
    C = _gpaw.scalapack_redist(A[2:], d12, d22)
elif rank < 4:
    C = _gpaw.scalapack_redist(None, d12, d22)
print rank, C
