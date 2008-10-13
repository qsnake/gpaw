import numpy as np
from gpaw.eigensolvers.eigensolver import blocked_matrix_multiply as bmm
for r in [9, 18]:
    psit_nG = np.ones((16, 18, 18, r))
    work_nG = np.ones((4, 18, 18, r))
    U_nn = np.zeros((16, 16))
    bmm(psit_nG, U_nn, work_nG)
