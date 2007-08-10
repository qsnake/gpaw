from math import sqrt, pi
import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!
from gpaw.domain import Domain
from gpaw.setup import Setup
from gpaw.grid_descriptor import GridDescriptor
from gpaw.localized_functions import create_localized_functions
from gpaw.xc_functional import XCFunctional

n = 40 /8 * 10
a = 10.0
domain = Domain((a, a, a))
gd = GridDescriptor(domain, (n, n, n))
c_LL = num.identity(9, num.Float)
a_Lg = gd.new_array(9)
nspins = 2
xcfunc = XCFunctional('LDA', nspins)
for soft in [False]:
    s = Setup('Cu', xcfunc, lmax=2)
    ghat_l = s.ghat_l
    ghat_Lg = create_localized_functions(ghat_l, gd, (0.54321, 0.5432, 0.543))
    a_Lg[:] = 0.0
    ghat_Lg.add(a_Lg, c_LL)
    for l in range(3):
        for m in range(2 * l + 1):
            print soft, l, m
            L = l**2 + m
            a_g = a_Lg[L]
            Q0 = gd.integrate(a_g) / sqrt(4 * pi)
            Q1_m = -gd.calculate_dipole_moment(a_g) / sqrt(4 * pi / 3)
            if l == 0:
                Q0 -= 1.0
                Q1_m[:] = 0.0
            elif l == 1:
                Q1_m[(m + 1) % 3] -= 1.0
            print Q0, Q1_m, m
            assert abs(Q0) < 2e-6
            assert num.alltrue(abs(Q1_m) < 3e-5)
    b_Lg = num.reshape(a_Lg, (9, n**3))
    S_LL = inner(b_Lg, b_Lg)
    S_LL.flat[::10] = 0.0
    print max(abs(S_LL).flat)
    assert num.alltrue(abs(S_LL) < 1e-4)
