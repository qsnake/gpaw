from math import sqrt, pi
import Numeric as num
from gridpaw.domain import Domain
from gridpaw.setup import Setup
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.localized_functions import create_localized_functions
from gridpaw.xc_functional import XCFunctional

n = 40 /8 * 10
a = 10.0
domain = Domain((a, a, a))
gd = GridDescriptor(domain, (n, n, n))
c_LL = num.identity(9, num.Float)
a_Lg = gd.new_array(9)
xcfunc = XCFunctional('LDA')
for soft in [False, True]:
    s = Setup('Cu', xcfunc, lmax=2, softgauss=soft)
    gt_l = s.get_shape_functions()
    gt_Lg = create_localized_functions(gt_l, gd, (0.54321, 0.5432, 0.543))
    a_Lg[:] = 0.0
    gt_Lg.add(a_Lg, c_LL)
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
                Q1_m[m] -= 1.0
            print Q0, Q1_m
            assert abs(Q0) < 2e-6
            assert num.alltrue(abs(Q1_m) < 3e-5)
    b_Lg = num.reshape(a_Lg, (9, n**3))
    S_LL = num.innerproduct(b_Lg, b_Lg)
    S_LL.flat[::10] = 0.0
    print max(abs(S_LL).flat)
    assert num.alltrue(abs(S_LL) < 1e-4)
