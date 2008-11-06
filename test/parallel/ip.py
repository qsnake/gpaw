from time import time
from gpaw.transformers import Transformer
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.mpi import world

ngpts = 80
N_c = (ngpts, ngpts, ngpts)
a = 10.0
domain = Domain((a, a, a))
domain.set_decomposition(world, N_c=N_c)
gd = GridDescriptor(domain, N_c)
gdfine = gd.refine()
interpolate = Transformer(gd, gdfine, 3).apply
a1 = gd.empty()
a1[:] = 1.0
f = gdfine.empty()
ta = time()
r = 600
for i in range(r):
    interpolate(a1, f)
tb = time()
n = 8 * (1 + 2 + 4) * ngpts**3
print '%.3f GFlops' % (r * n / (tb - ta) * 1e-9)

"""
python: 0.330 GFlops
mpirun -np 2 gpaw-python: 0.500 GFlops
gpaw-python + OMP: 0.432 GFlops
"""
