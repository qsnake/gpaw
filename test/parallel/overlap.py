from time import time
import sys
import numpy as np
from gpaw import parsize, parsize_bands
from gpaw.band_descriptor import BandDescriptor
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.hs_operators import Operator

B = parsize_bands or 1   # number of groups
    
G = 120  # number of grid points (G x G x G)
N = 2000  # number of bands
repeats = 20

try:
    N = int(sys.argv[1])
    K = int(sys.argv[2])
except (IndexError, ValueError):
    N = 6
    K = 3
    repeats = 3

h = 0.2        # grid spacing
a = h * G      # side length of box
M = N // B     # number of bands per group
assert M * B == N

D = world.size // B  # number of domains
assert D * B == world.size

# Set up communicators:
r = world.rank // D * D
domain_comm = world.new_communicator(np.arange(r, r + D))
band_comm = world.new_communicator(np.arange(world.rank % D, world.size, D))

# Set up band and grid descriptors:
bd = BandDescriptor(N, band_comm, False)
gd = GridDescriptor((G, G, G), (a, a, a), True, domain_comm, parsize)

# Random wave functions:
psit_mG = gd.empty(M)
for m in range(M):
    np.random.seed(world.rank * M + m)
    psit_mG[m] = np.random.uniform(-0.5, 0.5, tuple(gd.n_c))
if world.rank == 0:
    print 'Size of wave function array:', psit_mG.shape
P_ani = {0: psit_mG[:, :2, 0, 0].copy(),
         1: psit_mG[:, -1, -1, -3:].copy()}
if 0:
    X = M // K
    assert K * X == M
    if G**3 // D // K * K < G**3 // D:
        X += 1
    print X
    work1_xG = gd.empty(X)
    work2_xG = gd.empty(X)

def run(psit_mG):
    overlap = Operator(bd, gd, K)
    if 0:
        overlap.work1_xG = work1_xG
        overlap.work2_xG = work2_xG
    #S_nn = np.empty((N, N))
    def S(x):
        return x
    dS_aii = {0: np.ones((2, 2)) * 0.123, 1: np.ones((3, 3)) * 0.321}
    S_nn = overlap.calculate_matrix_elements(psit_mG, P_ani, S, dS_aii)

    t1 = time()
    if world.rank == 0:
        print S_nn.round(5)
        inverse_cholesky(S_nn)
    C_nn = S_nn
    t2 = time()

    if world.rank == 0:
        print 'Cholesky Time %f' % (t2-t1)
        
    # Distribute matrix:
    world.broadcast(C_nn, 0)

    psit_mG = overlap.matrix_multiply(C_nn, psit_mG, P_ani)

    if world.rank == 0:
        print 'Made it past matrix multiply'

    # Check:
    S_nn = overlap.calculate_matrix_elements(psit_mG, P_ani, S, dS_aii)

    assert not(P_ani[0] - psit_mG[:, :2, 0, 0]).round(10).any()
    assert not(P_ani[1] - psit_mG[:, -1, -1, -3:]).round(10).any()

    if world.rank == 0:
        for n in range(N):
            assert abs(S_nn[n, n] - 1.0) < 1e-10
            assert not S_nn[n + 1:, n].round(10).any()

    return psit_mG

ta = time()

# Do twenty iterations
for x in range(repeats):
    psit_mG = run(psit_mG)

tb = time()

if world.rank == 0:
    print 'Total Time %f' % (tb -ta)
    
