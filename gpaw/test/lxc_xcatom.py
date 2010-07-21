import numpy as np
import numpy.random as ra
from gpaw.setup import create_setup
from gpaw.xc_functional import XCFunctional
from gpaw.test import equal, gen

for functional in [
    'X-None', 'X-C_PW', 'X-C_VWN', 'X-C_PZ',
    'X_PBE-C_PBE', 'X_PBE_R-C_PBE',
    'X_B88-C_P86', 'X_B88-C_LYP',
    'X_FT97_A-C_LYP'
    ]:
    gen('N', xcname=functional)

tolerance = 0.000005 # libxc must reproduce old gpaw energies
# zero Kelvin: in Hartree

reference = { # svnversion 5252
    'X-C_PW': 2.23194461135, # 'LDA'
    'X_PBE-C_PBE': 2.28208665019, # 'PBE'
    'X_PBE_R-C_PBE': 2.29201920843, # 'revPBE'
    }

tolerance_libxc = 0.000001 # libxc must reproduce reference libxc energies

reference_libxc = { # svnversion 5252
    'X-None': 1.95030600807,
    'X-C_PW': 2.23194461135,
    'X-C_VWN': 2.23297429824,
    'X-C_PZ': 2.23146045547,
    'X_PBE-C_PBE': 2.28208665019,
    'X_PBE_R-C_PBE': 2.29201920843,
    'X_B88-C_P86': 2.30508027546,
    'X_B88-C_LYP': 2.28183010548,
    'X_FT97_A-C_LYP': 2.26846048873,
    }

libxc_set = [
    'X-None', 'X-C_PW', 'X-C_VWN', 'X-C_PZ',
    'X_PBE-C_PBE', 'X_PBE_R-C_PBE',
    'X_B88-C_P86', 'X_B88-C_LYP',
    'X_FT97_A-C_LYP'
    ]

x = 0.000001
for xc in libxc_set:
    ra.seed(8)
    xcfunc = XCFunctional(xc, 1)
    s = create_setup('N', xcfunc)
    ni = s.ni
    nii = ni * (ni + 1) // 2
    D_p = 0.1 * ra.random(nii) + 0.4
    H_p = np.zeros(nii)

    E1 = s.xc_correction.calculate_energy_and_derivatives(D_p.reshape(1, -1),
                                                          H_p.reshape(1, -1))
    dD_p = x * ra.random(nii)
    D_p += dD_p
    dE = np.dot(H_p, dD_p) / x
    E2 = s.xc_correction.calculate_energy_and_derivatives(D_p.reshape(1, -1),
                                                          H_p.reshape(1, -1))
    print xc, dE, (E2 - E1) / x
    equal(dE, (E2 - E1) / x, 0.003)

    xcfunc = XCFunctional(xc, 2)
    d = create_setup('N', xcfunc)
    E2s = d.xc_correction.calculate_energy_and_derivatives(
        np.array([0.5 * D_p, 0.5 * D_p]), np.array([H_p, H_p]))
    print E2, E2s
    equal(E2, E2s, 1.0e-12)

    if xc in reference: # compare with old gpaw
        print 'A:', E2, reference[xc]
        equal(E2, reference[xc], tolerance)

    if xc in reference_libxc: # compare with reference libxc
        print 'B:', E2, reference_libxc[xc]
        equal(E2, reference_libxc[xc], tolerance)

    D_sp = 0.1 * ra.random((2, nii)) + 0.2
    H_sp = np.zeros((2, nii))

    E1 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    dD_sp = x * ra.random((2, nii))
    D_sp += dD_sp
    dE = np.dot(H_sp.ravel(), dD_sp.ravel()) / x
    E2 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    print dE, (E2 - E1) / x
    equal(dE, (E2 - E1) / x, 0.005)
