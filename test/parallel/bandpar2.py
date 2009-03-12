from time import time
import sys
import numpy as np
from gpaw import parsize, parsize_bands
from gpaw.grid_descriptor import GridDescriptor
from gpaw.operators import Laplace
from gpaw.mpi import world
from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import inverse_cholesky

B = parsize_bands   # number of blocks
    
G = 120  # number of grid points (G x G x G)
N = 2000  # number of bands

h = 0.2        # grid spacing
a = h * G      # side length of box
M = N // B     # number of bands per block
assert M * B == N

D = world.size // B  # number of domains
assert D * B == world.size

# Set up communicators:
r = world.rank // D * D
domain_comm = world.new_communicator(np.arange(r, r + D))
band_comm = world.new_communicator(np.arange(world.rank % D, world.size, D))

# Set up grid descriptor:
gd = GridDescriptor((G, G, G), (a, a, a), True, domain_comm, parsize)

# Random wave functions:
np.random.seed(world.rank)
psit_mG = np.random.uniform(-0.5, 0.5, size=(M,) + tuple(gd.n_c))
if world.rank == 0:
    print 'Size of wave function array:', psit_mG.shape

# Send and receive buffers:
send_mG = gd.empty(M)
recv_mG = gd.empty(M)

def run():
    S_nn = overlap(psit_mG, send_mG, recv_mG)

    t1 = time()
    if world.rank == 0:
        inverse_cholesky(S_nn)
        C_nn = S_nn
    else:
        C_nn = np.empty((N, N))
    t2 = time()

    if world.rank == 0:
        print 'Cholesky Time %f' % (t2-t1)
        
    # Distribute matrix:
    world.broadcast(C_nn, 0)

    psit_mG[:] = matrix_multiply(C_nn, psit_mG, send_mG, recv_mG)

    if world.rank == 0:
        print 'Made it past matrix multiply'

    # Check:
    S_nn = overlap(psit_mG, send_mG, recv_mG)

#    Assert below requires more memory.
    if world.rank == 0:
        # Fill in upper part:
        for n in range(N - 1):
            S_nn[n, n + 1:] = S_nn[n + 1:, n]
#      assert (S_nn.round(7) == np.eye(N)).all()

def overlap(psit_mG, send_mG, recv_mG):
    """Calculate overlap matrix.

    The master (rank=0) will return the matrix with only the lower
    part filled in."""
    
    Q = B // 2 + 1
    rank = band_comm.rank
    S_imm = np.empty((Q, M, M))
    send_mG[:] = psit_mG

    # Shift wave functions:
    for i in range(Q - 1):
        rrequest = band_comm.receive(recv_mG, (rank + 1) % B, 42, False)
        srequest = band_comm.send(send_mG, (rank - 1) % B, 42, False)
        gemm(gd.dv, psit_mG, send_mG, 0.0, S_imm[i], 'c')
        band_comm.wait(rrequest)
        band_comm.wait(srequest)
        send_mG, recv_mG = recv_mG, send_mG
    gemm(gd.dv, psit_mG, send_mG, 0.0, S_imm[Q - 1], 'c')

    domain_comm.sum(S_imm)

    t1 = time()
    if domain_comm.rank == 0:
        if band_comm.rank == 0:
            # Master collects submatrices:
            S_nn = np.empty((N,N))
            S_nn[:] = 11111.77777
            S_imim = S_nn.reshape((B, M, B, M))
            S_imim[:Q, :, 0] = S_imm
            for i1 in range(1, B):
                band_comm.receive(S_imm, i1, i1)
                for i2 in range(Q):
                    if i1 + i2 < B:
                        S_imim[i1 + i2, :, i1] = S_imm[i2]
                    else:
                        S_imim[i1, :, i1 + i2 - B] = S_imm[i2].T
            t2 = time()
            print 'Collect submatrices time %f' % (t2-t1)
            return S_nn
        else:
            # Slaves send their submatrices:
            band_comm.send(S_imm, 0, band_comm.rank)

def matrix_multiply(C_nn, psit_mG, send_mG, recv_mG):
    """Calculate new linear compination of wave functions."""
    rank = band_comm.rank
    C_imim = C_nn.reshape((B, M, B, M))
    send_mG[:] = psit_mG
    psit_mG[:] = 0.0
    beta = 0.0
    for i in range(B - 1):
        rrequest = band_comm.receive(recv_mG, (rank + 1) % B, 117, False)
        srequest = band_comm.send(send_mG, (rank - 1) % B, 117, False)
        gemm(1.0, send_mG, C_imim[rank, :, (rank + i) % B], beta, psit_mG)
        beta = 1.0
        band_comm.wait(rrequest)
        band_comm.wait(srequest)
        send_mG, recv_mG = recv_mG, send_mG
    gemm(1.0, send_mG, C_imim[rank, :, rank - 1], beta, psit_mG)

    return psit_mG

ta = time()

# Do twenty iterations
for x in range(20):
    run()

tb = time()

if world.rank == 0:
    print 'Total Time %f' % (tb -ta)
    
