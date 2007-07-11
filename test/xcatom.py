import Numeric as num
import RandomArray as ra
from gpaw.setup import Setup
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import equal


x = 0.000001
ra.seed(1, 2)
nspins_1 = 1
nspins_2 = 2
for xc in ['LDA', 'PBE']:
    xcfunc = XCFunctional(xc, nspins_1)
    s = Setup('N', xcfunc)
    ni = s.ni
    np = ni * (ni + 1) / 2
    D_p = 0.1 * ra.random(np) + 0.2
    H_p = num.zeros(np, num.Float)

    E1 = s.xc_correction.calculate_energy_and_derivatives([D_p], [H_p])
    dD_p = x * ra.random(np)
    D_p += dD_p
    dE = num.dot(H_p, dD_p) / x
    E2 = s.xc_correction.calculate_energy_and_derivatives([D_p], [H_p])
    equal(dE, (E2 - E1) / x, 0.003)

    xcfunc = XCFunctional(xc, nspins_2)
    d = Setup('N', xcfunc, nspins=2)
    E2s = d.xc_correction.calculate_energy_and_derivatives([0.5 * D_p,
                                                            0.5 * D_p],
                                                           [H_p, H_p])
    equal(E2, E2s, 1.0e-12)

    D_sp = 0.1 * ra.random((2, np)) + 0.2
    H_sp = num.zeros((2, np), num.Float)

    E1 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    dD_sp = x * ra.random((2, np))
    D_sp += dD_sp
    dE = num.dot(H_sp.flat, dD_sp.flat) / x
    E2 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    equal(dE, (E2 - E1) / x, 0.005)
