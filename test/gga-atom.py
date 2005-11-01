import Numeric as num
import RandomArray as ra
from gridpaw.utilities import equal
from gridpaw.setup import Setup
from gridpaw.domain import Domain
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.localized_functions import create_localized_functions
from gridpaw.spline import Spline
from gridpaw.xc_functional import XCFunctional, XCOperator
from gridpaw.utilities import pack


ra.seed(1, 3)
for name in ['LDA', 'PBE']:
    xcfunc = XCFunctional(name)
    d = Setup('N', xcfunc)
    ni = d.get_number_of_partial_waves()
    niAO = d.get_number_of_atomic_orbitals()
    wt0_j = d.get_atomic_orbitals()

    wt_j = []
    for wt0 in wt0_j:
        data = [wt0(r) for r in num.arange(121) * d.rcut / 100]
        data[-1] = 0.0
        l = wt0.get_angular_momentum_number()
        wt_j.append(Spline(l, 1.2 * d.rcut, data))

    a = d.rcut * 1.2 * 2 + 1.0
##    n = 120
    n = 70
    n = 90
    domain = Domain((a, a, a))
    gd = GridDescriptor(domain, (n, n, n))
    pr = create_localized_functions(wt_j, gd, (0.5, 0.5, 0.5), onohirose=1)

    coefs = num.identity(niAO, num.Float)
    psit_ig = num.zeros((niAO, n, n, n), num.Float)
    pr.add(psit_ig, coefs)

    np = ni * (ni + 1) / 2
    npAO = niAO * (niAO + 1) / 2
    D_p = num.zeros(np, num.Float)
    H_p = num.zeros(np, num.Float)


    e_g = num.zeros((n, n, n), num.Float)
    n_g = num.zeros((n, n, n), num.Float)
    v_g = num.zeros((n, n, n), num.Float)

    P_ni = 0.2 * ra.random((20, ni))
    P_ni[:, niAO:] = 0.0
    D_i1i2 = num.dot(num.transpose(P_ni), P_ni)
    D_p = pack(D_i1i2)

    p = 0
    for i1 in range(niAO):
        for i2 in range(i1, niAO):
            n_g += D_p[p] * psit_ig[i1] * psit_ig[i2]
            p += 1
        p += ni - niAO



    p = create_localized_functions([d.get_smooth_core_density()],
                                   gd, (0.5, 0.5, 0.5), onohirose=1)
    p.add(n_g, num.ones(1, num.Float))
    xc = XCOperator(xcfunc, gd, nspins=1)
    xc.get_energy_and_potential(n_g, v_g)

    r2_g = num.sum((num.indices((n, n, n)) - n / 2)**2)
    dv_g = gd.dv * num.less(r2_g, (d.rcut / a * n)**2)

    E2 = -num.dot(xc.e_g.flat, dv_g.flat)

    d.xc.n_qg[:] = 0.0
    d.xc.nc_g[:] = 0.0
    E1 = d.xc.calculate_energy_and_derivatives([D_p], [H_p]) + d.xc.Exc0

    equal(E1, E2, 0.0032)
