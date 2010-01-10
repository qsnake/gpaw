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
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import rk, gemm
from gpaw.utilities.blacs import scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_ex, scalapack_diagonalize_dc
from gpaw.mpi import world, rank

tol = 2.0e-8

def main(N=1000, seed=42, mprocs=2, nprocs=2, dtype=float):
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, mprocs, nprocs)
    
    if (dtype==complex):
        epsilon = 1.0j
    else:
        epsilon = 0.0

    # Create descriptors for matrices on master:
    globH = grid.new_descriptor(N, N, N, N)
    globS = grid.new_descriptor(N, N, N, N)

    # print globA.asarray()
    # Populate matrices local to master:
    H0 = globH.zeros(dtype=dtype) + gen.rand(*globH.shape)
    S0 = globS.zeros(dtype=dtype) + gen.rand(*globS.shape)
    if rank == 0:
        # Complex case must have real numbers on the diagonal.
        # We make a simple complex Hermitian matrix below.
        H0 = H0 + epsilon * (0.1*np.tri(N, N, k= -N // nprocs) + 0.3*np.tri(N, N, k=-1))
        S0 = S0 + epsilon * (0.2*np.tri(N, N, k= -N // nprocs) + 0.4*np.tri(N, N, k=-1))
        # Make matrices symmetric
        rk(1.0, H0.copy(), 0.0, H0)
        rk(1.0, S0.copy(), 0.0, S0)
        # Overlap matrix must be semi-positive definite
        S0 = S0 + 50.0*np.eye(N, N, 0)
        # Hamiltonian is usually diagonally dominant
        H0 = H0 + 75.0*np.eye(N, N, 0)

    # Local result matrices
    W0 = np.zeros((N),dtype=float)
    W0_g = np.zeros((N),dtype=float)

    # Calculate eigenvalues
    if rank == 0:
        info = diagonalize(H0.copy(), W0)
        if info > 0:
            raise RuntimeError('LAPACK diagonalized failed.')
        info = diagonalize(H0.copy(), W0_g, S0.copy())
        if info > 0:
            raise RuntimeError('LAPACK general diagonalize failed.')
        
    assert globH.check(H0) and globS.check(S0)

    # Create distributed destriptors with various block sizes:
    distH = grid.new_descriptor(N, N, 64, 64)
    distS = grid.new_descriptor(N, N, 64, 64)
    distZ = grid.new_descriptor(N, N, 64, 64)

    # Distributed matrices:
    H = distH.zeros(dtype=dtype)
    S = distS.zeros(dtype=dtype)
    Z = distZ.zeros(dtype=dtype)  
    W = np.zeros((N), dtype=float)
    W_dc = np.zeros((N), dtype=float)
    W_g = np.zeros((N), dtype=float)
    
    Redistributor(world, globH, distH).redistribute(H0, H)
    Redistributor(world, globS, distS).redistribute(S0, S)

    scalapack_diagonalize_ex(distH, H.copy(), Z, W, 'U')
    scalapack_diagonalize_dc(distH, H.copy(), Z, W_dc, 'U')
    scalapack_general_diagonalize_ex(distH, H.copy(), S.copy(), Z, W_g, 'U')

    if rank == 0:
        diag_ex_err = abs(W - W0).max()
        diag_dc_err = abs(W_dc - W0).max()
        general_diag_ex_err = abs(W_g - W0_g).max()
        print 'diagonalize ex err', diag_ex_err
        print 'diagonalize dc err', diag_dc_err
        print 'general diagonalize ex err', general_diag_ex_err

    else:
        diag_ex_err = 0.0
        diag_dc_err = 0.0
        general_diag_ex_err = 0.0

    # We don't like exceptions on only one cpu
    diag_ex_err = world.sum(diag_ex_err)
    diag_dc_err = world.sum(diag_dc_err)
    general_diag_ex_err = world.sum(general_diag_ex_err) 
    assert diag_ex_err < tol
    assert diag_dc_err < tol
    assert general_diag_ex_err < tol

if __name__ == '__main__':
    main(dtype=float)
    main(dtype=complex)


                   
