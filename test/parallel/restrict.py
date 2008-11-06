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
gdcoarse = gd.coarsen()
restrict = Transformer(gd, gdcoarse, 2).apply
a1 = gd.empty()
a1[:] = 1.0
f = gdcoarse.empty()
ta = time()
r = 600
print a1.shape, f.shape
for i in range(r):
    restrict(a1, f)
tb = time()
print tb - ta
#n = 8 * (1 + 2 + 4) * ngpts**3
#print '%.3f GFlops' % (r * n / (tb - ta) * 1e-9)
