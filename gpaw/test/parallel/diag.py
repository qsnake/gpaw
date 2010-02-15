import numpy as np

from time import time

from gpaw.blacs import BlacsGrid, parallelprint
from gpaw.mpi import world, rank, size
from gpaw.utilities.blacs import scalapack_general_diagonalize_dc, \
    scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_dc, scalapack_diagonalize_ex, \
    scalapack_inverse_cholesky, \
    scalapack_diagonalize_mr3, scalapack_general_diagonalize_mr3

from gpaw.utilities.blacs import scalapack_set, scalapack_zero

rank = world.rank

mproc = 64
grid = BlacsGrid(world, mproc, mproc)
nbands = 8000

nndesc = grid.new_descriptor(nbands, nbands, 32, 32)

H_nn = nndesc.empty(dtype=float)
U_nn = nndesc.empty(dtype=float)
eps_N  = np.empty((nbands), dtype=float)

scalapack_set(nndesc, H_nn, 0.1, 75.0, 'L') 

t1 = time()
scalapack_diagonalize_dc(nndesc, H_nn.copy(), U_nn, eps_N, 'L')
t2 = time()

if world.rank == 0:
    print 'diagonalize_dc', t2-t1
    
t1 = time()
scalapack_diagonalize_ex(nndesc, H_nn.copy(), U_nn, eps_N, 'L')
t2 = time()

if world.rank == 0:
    print 'diagonalize_ex', t2-t1
    
t1 = time()
scalapack_diagonalize_mr3(nndesc, H_nn.copy(), U_nn, eps_N, 'L')
t2 = time()

if world.rank == 0:
    print 'diagonalize_mr3', t2-t1
    
