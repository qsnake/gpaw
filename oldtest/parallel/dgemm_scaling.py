from time import time
import sys
import numpy as np
from gpaw.utilities.blas import gemm, scal
from _gpaw import hpm_start, hpm_stop

G = 15  # number of grid points (G x G x G)
N = 2000 # total number of bands

beta = 1.0
dv = 0.1

# Random matrix
C_nn = np.random.uniform(-0.5, 0.5, (N,N))

for B in (1,2,4,8,16):

    M = N // B     # number of bands per group
    assert M * B == N

    Q = B // 2 + 1
    C_bnbn = C_nn.reshape((B, M, B, M))
    S_mm = np.empty((Q,M,M))

    # Random wave functions:
    shape = (M, G, G, G)
    np.random.seed(M)
    psit_mG = np.random.uniform(-0.5, 0.5, shape)
    tmp_mG = psit_mG.copy()

    # Overlap
    ttot = 0.0
    ta = time()
    reg = "Overlap" + str(B)
    hpm_start(reg)
    for n in range(Q - 1):
        gemm(dv, tmp_mG, psit_mG, 0.0, S_mm[n], 'c')
        tmp_mG, psit_mG = psit_mG, tmp_mG
    gemm(dv, tmp_mG, psit_mG, 0.0, S_mm[Q - 1], 'c')
    hpm_stop(reg)
    ttot = time() - ta

    # print "Overlap, B:", B, "Time:", ttot

    # Matrix multiply
    ttot = 0.0
    ta = time()
    reg = "Matrix_Mutiply" + str(B)
    hpm_start(reg)
    beta = 0.0
    for n in range(B - 1):
        C_mm = C_bnbn[0, :, n % B] 
        gemm(1.0, tmp_mG, C_mm, beta, psit_mG)
        beta = 1.0
        tmp_mG, psit_mG = psit_mG, tmp_mG
    C_mm = C_bnbn[0, :, -1] 
    gemm(1.0, tmp_mG, C_mm, beta, psit_mG)      
    hpm_stop(reg)
    ttot = time() - ta

    # print "Matrix_Multiply, B:", B, "Time:", ttot
