"""Test of pblas_pdgemm.  This test requires 4 processors.

This is a test of the GPAW interface to the parallel
matrix multiplication routine, pblas_pdgemm.

The test generates random matrices A and B and their product C on master.

Then A and B are distributed, and pblas_dgemm is invoked to get C in
distributed form.  This is then collected to master and compared to
the serially calculated reference.
"""

import numpy as np
from gpaw.blacs import BlacsGrid, Redistributor, parallelprint
from gpaw.utilities.blacs import pblas_simple_gemm
from gpaw.mpi import world, rank
import _gpaw


def main():
    seed = 42
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, 2, 2)

    M = 16
    N = 12
    K = 14

    # Create matrices on master:
    globA = grid.new_descriptor(M, K, M, K)
    globB = grid.new_descriptor(K, N, K, N)
    globC = grid.new_descriptor(M, N, M, N)

    # Random matrices local to master:
    A0 = gen.rand(*globA.shape)
    B0 = gen.rand(*globB.shape)
    C0 = globC.empty()

    # Local matrix product:
    if rank == 0:
        C0[:] = np.dot(A0, B0)

    assert globA.check(A0) and globB.check(B0) and globC.check(C0)

    # Create distributed destriptors with various block sizes:
    distA = grid.new_descriptor(M, K, 2, 3)
    distB = grid.new_descriptor(K, N, 2, 4)
    distC = grid.new_descriptor(M, N, 3, 2)

    # Distributed matrices:
    A = distA.empty()
    B = distB.empty()
    C = distC.empty()

    Redistributor(world, globA, distA).redistribute(A0, A)
    Redistributor(world, globB, distB).redistribute(B0, B)

    pblas_simple_gemm(distA, distB, distC, A, B, C)

    # Collect result back on master
    C1 = globC.empty()
    Redistributor(world, distC, globC).redistribute(C, C1)

    if rank == 0:
        err = abs(C0 - C1).max()
        print 'Err', err
    else:
        err = 0.0
    err = world.sum(err) # We don't like exceptions on only one process
    assert err < 1e-14
    

if __name__ == '__main__':
    main()
