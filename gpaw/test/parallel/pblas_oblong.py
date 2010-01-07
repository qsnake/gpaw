import numpy as np

from gpaw.blacs import BlacsGrid, parallelprint
from gpaw.mpi import world, rank, size
from gpaw.utilities.blacs import pblas_simple_gemm

gen = np.random.RandomState(42)

grid = BlacsGrid(world, 2, 32)

nbands = 10
nG = 1300000

nGdesc = grid.new_descriptor(nbands, nG, 8, 80)
nndesc = grid.new_descriptor(nbands, nbands, 8, 8)

psit_nG = gen.rand(*nGdesc.shape)
A_nn = gen.rand(*nndesc.shape)

assert nGdesc.check(psit_nG)
assert nndesc.check(A_nn)

parallelprint(world, (A_nn.shape, nndesc.shape, nndesc.lld))

pblas_simple_gemm(nGdesc, nGdesc, nndesc, psit_nG, psit_nG, A_nn,
                  transa='N', transb='T')
