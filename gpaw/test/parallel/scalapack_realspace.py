"""Test of ScaLAPACK diagonalize and inverse cholesky.

The test generates a random symmetric matrix H0 and 
positive definite matrix S0 on a 1-by-1 BLACS grid. They
are redistributed to a mprocs-by-nprocs BLACS grid, 
diagonalized in parallel, and eigenvalues are compared
against LAPACK. Eigenvectors are not compared.
"""

import sys

import numpy as np

from gpaw import parsize, parsize_bands
from gpaw.grid_descriptor import GridDescriptor
from gpaw.band_descriptor import BandDescriptor

# do not import sl_diagonalize, will cause problems in
# this example
from gpaw.mpi import world 

from gpaw.blacs import BlacsGrid, Redistributor, parallelprint, \
    BlacsBandDescriptor
from gpaw.utilities import scalapack
from gpaw.utilities.lapack import diagonalize, inverse_cholesky
from gpaw.utilities.blas import rk, gemm
from gpaw.utilities.blacs import scalapack_diagonalize_dc, \
    scalapack_inverse_cholesky

from gpaw.utilities.timing import nulltimer

B = parsize_bands   # number of blocks
    
G = 30  # number of grid points (G x G x G)
N = 100  # number of bands

h = 0.2        # grid spacing
a = h * G      # side length of box
n = N // B     # number of bands per block
assert N % B == 0 # state-parallelization must be integer divisible

D = parsize[0] * parsize[1] * parsize[2] # number of domains

# Set up communicators:
size = world.size
rank = world.rank

r0 = (rank // D) * D
ranks = np.arange(r0, r0 + D)
domain_comm = world.new_communicator(ranks)

r0 = rank % (D * B)
ranks = np.arange(r0, r0 + size, D * B)
kpt_comm = world.new_communicator(ranks)

r0 = rank % D + kpt_comm.rank * (D * B)
ranks = np.arange(r0, r0 + (D * B), D)
band_comm = world.new_communicator(ranks)

assert size == D*B*kpt_comm.size # Check to make sure everything adds up

# Set up grid descriptor:
gd = GridDescriptor((G, G, G), (a, a, a), True, domain_comm, parsize)
bd = BandDescriptor(N, band_comm)

# Set up column communicator for use with 1D BLACS grid
bcommsize = bd.comm.size
gcommsize = gd.comm.size

# Tolerance for diagonalize and inverse_cholesky
tol = 1.0e-8

# ScaLAPACK parameters
mprocs, nprocs, blocksize = 2, 2, 64
assert mprocs*nprocs <= gcommsize*bcommsize

def main(seed=42, dtype=float):
    gen = np.random.RandomState(seed)

    # horrible acronym
    bbd = BlacsBandDescriptor(world, gd, bd, kpt_comm, mprocs, nprocs, blocksize)

    assert N % B == 0 # B must be integer divisible for state-parallelzation
    n = N / B
    if (dtype==complex):
        epsilon = 1.0j
    else:
        epsilon = 0.0

    # Create descriptors for matrices on master.
    # We can do this on the 1D or 2D grid
    glob = bbd.columngrid.new_descriptor(N, N, N, N) # this cannot be blockgrid

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
    eps0_N = np.empty(N,dtype=float)

    # Calculate eigenvalues
    if rank == 0:
        info = diagonalize(H0.copy(), eps0_N)
        if info > 0:
            raise RuntimeError('LAPACK diagonalized failed.')
        info = inverse_cholesky(C0) # result returned in lower triangle
        # tri2full(C0) # symmetrize
        if info > 0:
            raise RuntimeError('LAPACK inverse cholesky failed.')

    assert glob.check(H0) and glob.check(S0) and glob.check(C0)

    # This descriptor matches the 1D layout that arises
    # from the parallel matrix multiply in hs_operators
    # dist1D = bbd.columngrid.new_descriptor(N, N, N, N/B)
    dist1D = bbd.Nndescriptor

    # Distributed matrices:
    # input
    H_Nn = dist1D.empty(dtype=dtype)
    S_Nn = dist1D.empty(dtype=dtype)
    # output
    Z_Nn = dist1D.empty(dtype=dtype)
    C_Nn = dist1D.zeros(dtype=dtype)

    # Eigenvalues
    eps_n = np.empty(n, dtype=float)
    
    # Glob2dist1D
    Glob2dist1D = Redistributor(bbd.blockgrid.comm, glob, dist1D, 'U')
    Glob2dist1D.redistribute(H0, H_Nn)
    Glob2dist1D.redistribute(S0, S_Nn)
    
    diagonalizer = bbd.get_diagonalizer()
    diagonalizer.diagonalize(H_Nn, C_Nn, eps_n)

if __name__ == '__main__':
    if not scalapack():
        print('Not built with ScaLAPACK. Test does not apply.')
    else:
        main(dtype=float)
        main(dtype=complex)


                   
