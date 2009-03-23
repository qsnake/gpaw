import numpy as np
from gpaw.mpi import world, rank, size
import _gpaw
A = np.arange(16, dtype=float).reshape(4, 4)
c1 = world.new_communicator(np.array([0, 1]))
d12 = _gpaw.blacs_create(c1, 4, 4, 1, 2, 2, 2)
c2 = world.new_communicator(np.array([0, 1, 2, 3]))
d22 = _gpaw.blacs_create(c2, 4, 4, 2, 2, 2, 2)
C = None
if rank == 0:
    print A
    C = _gpaw.scalapack_redist(A[:2], d12, d22, d22)
elif rank == 1:
    C = _gpaw.scalapack_redist(A[2:], d12, d22, d22)
elif rank < 4:
    C = _gpaw.scalapack_redist(None, d12, d22, d22)
print rank, C
