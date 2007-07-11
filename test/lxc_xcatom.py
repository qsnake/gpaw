import Numeric as num
import RandomArray as ra
from gpaw.setup import Setup
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import equal
from gpaw import setup_paths
from lxc_testsetups import Lxc_testsetups

setup_paths.insert(0, '.')

setups = Lxc_testsetups()
setups.create()

tolerance = 0.000005 # libxc must reproduce old gpaw energies
# zero Kelvin: in Hartree
reference_886 = { # version 886
    'lxcX-C_PW': 2.3306776296, # 'LDA'
    'lxcX_PBE-C_PBE': 2.36833876588, # 'PBE'
    'lxcX_PBE_R-C_PBE': 2.37142515318 # 'revPBE'
    }

tolerance_libxc = 0.000001 # libxc must reproduce old libxc energies
reference_libxc_886 = { # version 886
    'lxcX-None': 2.04220165015,
    'lxcX-C_PW': 2.3306776296,
    'lxcX-C_VWN': 2.33175973998,
    'lxcX-C_PZ': 2.33011279593,
    'lxcX_PBE-C_PBE': 2.36833735076,
    'lxcX_PBE_R-C_PBE': 2.37142425035,
    'lxcX_B88-C_P86': 2.38801013406,
    'lxcX_B88-C_LYP': 2.3719969122,
    'lxcX_FT97_A-C_LYP': 2.34666425237
    }

libxc_set = [
    'lxcX-None', 'lxcX-C_PW', 'lxcX-C_VWN', 'lxcX-C_PZ',
    'lxcX_PBE-C_PBE', 'lxcX_PBE_R-C_PBE',
    'lxcX_B88-C_P86', 'lxcX_B88-C_LYP',
    'lxcX_FT97_A-C_LYP'
    ]

x = 0.000001
for xc in libxc_set:
    ra.seed(1, 2)
    xcfunc = XCFunctional(xc, 1)
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

    xcfunc = XCFunctional(xc, 2)
    d = Setup('N', xcfunc, nspins=2)
    E2s = d.xc_correction.calculate_energy_and_derivatives([0.5 * D_p,
                                                            0.5 * D_p],
                                                           [H_p, H_p])
    equal(E2, E2s, 1.0e-12)

    if reference_886.has_key(xc): # compare with old gpaw
        equal(E2, reference_886[xc], tolerance)

    if reference_libxc_886.has_key(xc): # compare with old libxc
        equal(E2, reference_libxc_886[xc], tolerance)

    D_sp = 0.1 * ra.random((2, np)) + 0.2
    H_sp = num.zeros((2, np), num.Float)

    E1 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    dD_sp = x * ra.random((2, np))
    D_sp += dD_sp
    dE = num.dot(H_sp.flat, dD_sp.flat) / x
    E2 = d.xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
    equal(dE, (E2 - E1) / x, 0.005)

setups.clean()
