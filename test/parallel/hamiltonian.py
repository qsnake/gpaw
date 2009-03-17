from time import time
import sys
import numpy as np
from gpaw import parsize, parsize_bands
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world
from gpaw.utilities.lapack import inverse_cholesky
from gpaw.hs_operators import Operator
from gpaw.operators import Laplace

B = parsize_bands or 1  # number of groups
    
G = 120  # number of grid points (G x G x G)
N = 2000  # number of bands
repeats = 20

try:
    N = int(sys.argv[1])
    J = int(sys.argv[2])
except (IndexError, ValueError):
    N = 6
    J = 3
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

# Set up grid descriptor:
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

kin = Laplace(gd, -0.5, 2).apply
vt_G = gd.empty()
vt_G.fill(0.567)

def run(psit_mG):
    overlap = Operator(band_comm, gd, J)
    def H(psit_xG):
        kin(psit_xG, overlap.work1_xG[:M // J])
        for psit_G, y_G in zip(psit_xG, overlap.work1_xG):
            y_G += vt_G * psit_G
        return overlap.work1_xG[:M // J]
    dH_aii = {0: np.ones((2, 2)) * 0.123, 1: np.ones((3, 3)) * 0.321}
    H_nn = overlap.calculate_matrix_elements(psit_mG, P_ani, H, dH_aii)

    t1 = time()
    if world.rank == 0:
        eps_n, H_nn = np.linalg.eigh(H_nn)
        H_nn = H_nn.T
    t2 = time()

    if world.rank == 0:
        print 'Diagonalization Time %f' % (t2-t1)
        print eps_n

    # Distribute matrix:
    world.broadcast(H_nn, 0)

    psit_mG = overlap.matrix_multiply(H_nn, psit_mG, P_ani)

    if world.rank == 0:
        print 'Made it past matrix multiply'

    # Check:
    assert not(P_ani[0] - psit_mG[:, :2, 0, 0]).round(10).any()
    assert not(P_ani[1] - psit_mG[:, -1, -1, -3:]).round(10).any()

    H_nn = overlap.calculate_matrix_elements(psit_mG, P_ani, H, dH_aii)
    if world.rank == 0:
        for n in range(N):
            assert abs(H_nn[n, n] - eps_n[n]) < 1e-8
            assert not H_nn[n + 1:, n].round(8).any()

    return psit_mG

ta = time()

for x in range(repeats):
    psit_mG = run(psit_mG)

tb = time()

if world.rank == 0:
    print 'Total Time %f' % (tb -ta)
    
