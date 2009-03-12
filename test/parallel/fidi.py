from time import time
from gpaw.grid_descriptor import GridDescriptor
from gpaw.operators import Laplace
import gpaw.mpi as mpi

n = 96
h = 0.1
L = n * h
gd = GridDescriptor((n, n, n), [L, L, L])

# Allocate arrays:
a = gd.zeros(100) + 1.2
b = gd.empty(100)

o = Laplace(gd, 2).apply

t0 = time()
for r in range(10):
    o(a, b)

if mpi.rank == 0:
    print time() - t0, a.shape
    
