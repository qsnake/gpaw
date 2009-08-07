from time import time
import sys
import numpy as np
from gpaw.utilities.blas import gemm
from _gpaw import hpm_start, hpm_stop

G = 20  # number of grid points (G x G x G)
N = 2000  # total number of bands
repeats = 5

J = 1 # number of blocks

# Random "Hamiltonian" matrix
A_nn = np.random.uniform(-0.5, 0.5, (N,N))

for B in (1,2,4,8,16):

    M = N // B     # number of bands per group
    assert M * B == N

    A_bnbn = A_nn.reshape((B, M, B, M))

    # Random wave functions:
    shape = (M, G, G, G)
    np.random.seed(M)
    psit_mG = np.random.uniform(-0.5, 0.5, shape)
    tmp_mG = psit_mG.copy()

    ttot = 0.0
    ta = time()
    reg = "gemm" + str(B)
    hpm_start(reg)
    for n in range(B):
        A_mm = A_bnbn[0, :, n % B] 
        gemm(1.0, tmp_mG, A_mm, 0.0, psit_mG)
    hpm_stop(reg)
    ttot = time() - ta

    print "B:", B, "Time:", ttot
