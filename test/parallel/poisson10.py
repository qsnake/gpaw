import numpy as np
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.operators import Laplace
from gpaw.mpi import rank, size, world


G = 40
N = 2 * 3 * 2 * 5 * 7 * 8 * 3 * 11
B = 2
h = 0.2
a = h * R
M = N // B
assert M * B == N
D = size // B
assert D * B == size
r = rank // B * B
domain_comm = mpi.world.new_communicator(np.arange(r, r + D))
band_comm = mpi.world.new_communicator(np.arange(rank, size, D))
domain = Domain((a, a, a))
domain.set_decomposition(domain_comm, (G, G, G))
gd = GridDescriptor(domain, (G, G, G))

np.random.seed(rank)
psit_mG = np.random.uniform(size=(M,) + tuple(gd.n_c))
send_mG = gd.empty(M)
recv_mG = gd.empty(M)

laplace = [Laplace(g, n=1).apply for g in gd]

for i in range(B // 2):
    rrequest = band_comm.reveive(recv_mG, (rank + D) % size, 42, 0)
    srequest = band_comm.send(send_mG, (rank - D) % size, 17, 0)
    
