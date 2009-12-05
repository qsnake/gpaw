"""Test of pblas_pdgemm.  This test requires 4 processors unless other
values of mprocs and nprocs are specified to main().

This is a test of the GPAW interface to the parallel
matrix multiplication routine, pblas_pdgemm.

The test generates random matrices A and B and their product C on master.

Then A and B are distributed, and pblas_dgemm is invoked to get C in
distributed form.  This is then collected to master and compared to
the serially calculated reference.
"""

import sys

import numpy as np

from gpaw.blacs import BlacsGrid, Redistributor, parallelprint
from gpaw.utilities.blas import gemm, gemv
from gpaw.utilities.blacs import pblas_simple_gemm, pblas_simple_gemv
from gpaw.mpi import world, rank
import _gpaw


def main(M=160, N=120, K=140, seed=42, mprocs=2, nprocs=2):
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, mprocs, nprocs)

    # Create descriptors for matrices on master:
    globA = grid.new_descriptor(M, K, M, K)
    globB = grid.new_descriptor(K, N, K, N)
    globC = grid.new_descriptor(M, N, M, N)
    globX = grid.new_descriptor(K, 1, K, 1)
    globY = grid.new_descriptor(M, 1, M, 1)

    # Populate matrices local to master:
    A0 = gen.rand(*globA.shape)
    B0 = gen.rand(*globB.shape)
    X0 = gen.rand(*globX.shape)

    # Local result matrices
    Y0 = globY.empty()
    C0 = globC.empty()

    # Local reference matrix product:
    if rank == 0:
        # C0[:] = np.dot(A0, B0)
        gemm(1.0, B0, A0, 0.0, C0)
        # Y0[:] = np.dot(A0, X0)
        gemv(1.0, A0, X0, 0.0, Y0)

    assert globA.check(A0) and globB.check(B0) and globC.check(C0)
    assert globX.check(X0) and globY.check(Y0)

    # Create distributed destriptors with various block sizes:
    distA = grid.new_descriptor(M, K, 2, 2)
    distB = grid.new_descriptor(K, N, 2, 4)
    distC = grid.new_descriptor(M, N, 3, 2)
    distX = grid.new_descriptor(K, 1, 4, 1)
    distY = grid.new_descriptor(M, 1, 3, 1)

    # Distributed matrices:
    A = distA.empty()
    B = distB.empty()
    C = distC.empty()
    X = distX.empty()
    Y = distY.empty()

    Redistributor(world, globA, distA).redistribute(A0, A)
    Redistributor(world, globB, distB).redistribute(B0, B)
    Redistributor(world, globX, distX).redistribute(X0, X)

    pblas_simple_gemm(distA, distB, distC, A, B, C)
    pblas_simple_gemv(distA, distX, distY, A, X, Y)

    # Collect result back on master
    C1 = globC.empty()
    Y1 = globY.empty()
    Redistributor(world, distC, globC).redistribute(C, C1)
    Redistributor(world, distY, globY).redistribute(Y, Y1)

    if rank == 0:
        gemm_err = abs(C1 - C0).max()
        gemv_err = abs(Y1 - Y0).max()
        print 'gemm err', gemm_err
        print 'gemv err', gemv_err
    else:
        gemm_err = 0.0
        gemv_err = 0.0
    gemm_err = world.sum(gemm_err) # We don't like exceptions on only one cpu
    gemv_err = world.sum(gemv_err)
    
    assert gemm_err < 1e-13
    assert gemv_err < 1e-13
    

if __name__ == '__main__':
    main()
