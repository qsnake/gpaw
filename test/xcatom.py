import numpy as np
import numpy.random as ra
from gpaw.setup import Setup
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import equal


x = 0.000001
ra.seed(8)
nspins_1 = 1
nspins_2 = 2
for xc in ['LDA', 'PBE']:
    xcfunc = XCFunctional(xc, nspins_1)
    s = Setup('N', xcfunc)
    ni = s.ni
    nii = ni * (ni + 1) / 2
    D_p = 0.1 * ra.random(nii) + 0.2
    H_p = np.zeros(nii)

    E1 = s.xc_correction.calculate_energy_and_derivatives([D_p], [H_p])
    dD_p = x * ra.random(nii)
    D_p += dD_p
    dE = np.dot(H_p, dD_p) / x
    E2 = s.xc_correction.calculate_energy_and_derivatives([D_p], [H_p])
    equal(dE, (E2 - E1) / x, 0.003)

    xcfunc = XCFunctional(xc, nspins_2)
    d = Setup('N', xcfunc, nspins=2)
    E2s = d.xc_correction.calculate_energy_and_derivatives([0.5 * D_p,
                                                            0.5 * D_p],
                                                           [H_p, H_p])
    equal(E2, E2s, 1.0e-12)

    D_sp = 0.1 * ra.random((2, nii)) + 0.2
    H_sp = np.zeros((2, nii))

    E1 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    dD_sp = x * ra.random((2, nii))
    D_sp += dD_sp
    dE = np.dot(H_sp.ravel(), dD_sp.ravel()) / x
    E2 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    equal(dE, (E2 - E1) / x, 0.005)
