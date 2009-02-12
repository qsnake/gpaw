import numpy as np
import numpy.random as ra
from gpaw.setup import Setup
from gpaw.xc_functional import XCFunctional


x = 0.000001
ra.seed(8)
nspins = 1
xcfunc = XCFunctional('LDA', nspins)
s = Setup('H', xcfunc)
ni = s.ni
nii = ni * (ni + 1) / 2
D_p = 0.1 * ra.random((1, nii)) + 0.2
H_p = np.zeros(nii)

def f(x):
    return x

J_pp = s.xc_correction.four_phi_integrals(D_p, f)

# Check integrals using two_phi_integrals function and finite differences:
pass
