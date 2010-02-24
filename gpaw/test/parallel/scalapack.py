"""Test of ScaLAPACK diagonalize and inverse cholesky.

The test generates a random symmetric matrix H0 and 
positive definite matrix S0 on a 1-by-1 BLACS grid. They
are redistributed to a mprocs-by-nprocs BLACS grid, 
diagonalized in parallel, and eigenvalues are compared
against LAPACK. Eigenvectors are not compared.
"""

import sys

import numpy as np

from gpaw.mpi import world, rank
from gpaw.blacs import BlacsGrid, Redistributor, parallelprint
from gpaw.utilities import scalapack
from gpaw.utilities.lapack import diagonalize, general_diagonalize, \
    inverse_cholesky
from gpaw.utilities.blas import rk, gemm
from gpaw.utilities.blacs import scalapack_general_diagonalize_dc, \
    scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_dc, scalapack_diagonalize_ex, \
    scalapack_inverse_cholesky ## , \
    ## scalapack_diagonalize_mr3, scalapack_general_diagonalize_mr3

tol = 1.0e-8

def main(N=73, seed=42, mprocs=2, nprocs=2, dtype=float):
    gen = np.random.RandomState(seed)
    grid = BlacsGrid(world, mprocs, nprocs)
    
    if (dtype==complex):
        epsilon = 1.0j
    else:
        epsilon = 0.0

    # Create descriptors for matrices on master:
    glob = grid.new_descriptor(N, N, N, N)

    # print globA.asarray()
    # Populate matrices local to master:
    H0 = glob.zeros(dtype=dtype) + gen.rand(*glob.shape)
    S0 = glob.zeros(dtype=dtype) + gen.rand(*glob.shape)
    C0 = glob.empty(dtype=dtype)
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
        C0 = S0.copy()

    # Local result matrices
    W0 = np.empty((N),dtype=float)
    W0_g = np.empty((N),dtype=float)

    # Calculate eigenvalues
    if rank == 0:
        info = diagonalize(H0.copy(), W0)
        if info > 0:
            raise RuntimeError('LAPACK diagonalized failed.')
        info = general_diagonalize(H0.copy(), W0_g, S0.copy())
        if info > 0:
            raise RuntimeError('LAPACK general diagonalize failed.')
        info = inverse_cholesky(C0) # result returned in lower triangle
        # tri2full(C0) # symmetrize
        if info > 0:
            raise RuntimeError('LAPACK inverse cholesky failed.')
        
    assert glob.check(H0) and glob.check(S0) and glob.check(C0)

    # Create distributed destriptors with various block sizes:
    dist = grid.new_descriptor(N, N, 8, 8)

    # Distributed matrices:
    # We can use empty here, but end up with garbage on
    # on the other half of the triangle when we redistribute.
    # This is fine because ScaLAPACK does not care.

    H = dist.empty(dtype=dtype)
    S = dist.empty(dtype=dtype)
    Z = dist.empty(dtype=dtype)
    C = dist.empty(dtype=dtype)

    # Eigenvalues are non-BLACS matrices
    W = np.empty((N), dtype=float)
    W_dc = np.empty((N), dtype=float)
    W_mr3 = np.empty((N), dtype=float)
    W_g = np.empty((N), dtype=float)
    W_g_dc = np.empty((N), dtype=float)
    W_g_mr3 = np.empty((N), dtype=float)

    Glob2dist = Redistributor(world, glob, dist, uplo='L')
    Glob2dist.redistribute(H0, H)
    Glob2dist.redistribute(S0, S)
    Glob2dist.redistribute(S0, C) # C0 was previously \
        # overwritten

    scalapack_diagonalize_ex(dist, H.copy(), Z, W, 'L')
    scalapack_diagonalize_dc(dist, H.copy(), Z, W_dc, 'L')
    ## scalapack_diagonalize_mr3(dist, H.copy(), Z, W_mr3, 'L')
    scalapack_general_diagonalize_ex(dist, H.copy(), S.copy(), Z, W_g, 'L')
    scalapack_general_diagonalize_dc(dist, H.copy(), S.copy(), Z, W_g_dc, 'L')
    ## scalapack_general_diagonalize_mr3(dist, H.copy(), S.copy(), Z, W_g_mr3, 'L')
    scalapack_inverse_cholesky(dist, C, 'L')

    # Undo redistribute
    C_test = glob.empty(dtype=dtype)
    Dist2glob = Redistributor(world, dist, glob)
    Dist2glob.redistribute(C, C_test)

    if rank == 0:
        diag_ex_err = abs(W - W0).max()
        diag_dc_err = abs(W_dc - W0).max()
        ## diag_mr3_err = abs(W_mr3 - W0).max()
        general_diag_ex_err = abs(W_g - W0_g).max()
        general_diag_dc_err = abs(W_g_dc - W0_g).max()
        ## general_diag_mr3_err = abs(W_g_mr3 - W0_g).max()
        inverse_chol_err = abs(C_test-C0).max()
        print 'diagonalize ex err', diag_ex_err
        print 'diagonalize dc err', diag_dc_err
        ## print 'diagonalize mr3 err', diag_mr3_err
        print 'general diagonalize ex err', general_diag_ex_err
        print 'general diagonalize dc err', general_diag_dc_err
        ## print 'general diagonalize mr3 err', general_diag_mr3_err
        print 'inverse chol err', inverse_chol_err 
    else:
        diag_ex_err = 0.0
        diag_dc_err = 0.0
        ## diag_mr3_err = 0.0
        general_diag_ex_err = 0.0
        general_diag_dc_err = 0.0
        ## general_diag_mr3_err = 0.0
        inverse_chol_err = 0.0

    # We don't like exceptions on only one cpu
    diag_ex_err = world.sum(diag_ex_err)
    diag_dc_err = world.sum(diag_dc_err)
    ## diag_mr3_err = world.sum(diag_mr3_err)
    general_diag_ex_err = world.sum(general_diag_ex_err)
    general_diag_dc_err = world.sum(general_diag_dc_err)
    ## general_diag_mr3_err = world.sum(general_diag_mr3_err) 
    inverse_chol_err = world.sum(inverse_chol_err)
    assert diag_ex_err < tol
    assert diag_dc_err < tol
    ## assert diag_mr3_err < tol
    assert general_diag_ex_err < tol
    assert general_diag_dc_err < tol
    ## assert general_diag_mr3_err < tol
    assert inverse_chol_err < tol

if __name__ in ['__main__', '__builtin__']:
    if not scalapack(True):
        print('Not built with ScaLAPACK. Test does not apply.')
    else:
        main(dtype=float)
        main(dtype=complex)


                   
