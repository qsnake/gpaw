import numpy as np
import numpy.random as ra
from gpaw.utilities import equal
from gpaw.setup import Setup
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.localized_functions import create_localized_functions
from gpaw.spline import Spline
from gpaw.xc_functional import XCFunctional, XC3DGrid
from gpaw.utilities import pack


ra.seed(8)
nspins=1
for name in ['LDA', 'PBE']:
    xcfunc = XCFunctional(name, nspins)
    s = Setup('N', xcfunc)
    ni = s.ni
    niAO = s.niAO
    wt0_j = s.phit_j

    rcut = s.xc_correction.rgd.r_g[-1]

    wt_j = []
    for wt0 in wt0_j:
        data = [wt0(r) for r in np.arange(121) * rcut / 100]
        data[-1] = 0.0
        l = wt0.get_angular_momentum_number()
        wt_j.append(Spline(l, 1.2 * rcut, data))

    a = rcut * 1.2 * 2 + 1.0
##    n = 120
    n = 70
    n = 90
    domain = Domain((a, a, a))
    gd = GridDescriptor(domain, (n, n, n))
    pr = create_localized_functions(wt_j, gd, (0.5, 0.5, 0.5))

    coefs = np.identity(niAO, float)
    psit_ig = np.zeros((niAO, n, n, n))
    pr.add(psit_ig, coefs)

    nii = ni * (ni + 1) / 2
    D_p = np.zeros(nii)
    H_p = np.zeros(nii)


    e_g = np.zeros((n, n, n))
    n_g = np.zeros((n, n, n))
    v_g = np.zeros((n, n, n))

    P_ni = 0.2 * ra.random((20, ni))
    P_ni[:, niAO:] = 0.0
    D_ii = np.dot(np.transpose(P_ni), P_ni)
    D_p = pack(D_ii)
    p = 0
    for i1 in range(niAO):
        for i2 in range(i1, niAO):
            n_g += D_p[p] * psit_ig[i1] * psit_ig[i2]
            p += 1
        p += ni - niAO

    p = create_localized_functions([s.nct], gd, (0.5, 0.5, 0.5))
    p.add(n_g, np.ones(1))
    xc = XC3DGrid(xcfunc, gd, nspins=1)
    xc.get_energy_and_potential(n_g, v_g)

    r2_g = np.sum((np.indices((n, n, n)) - n / 2)**2, axis=0)
    dv_g = gd.dv * np.less(r2_g, (rcut / a * n)**2)

    E2 = -np.dot(xc.e_g.ravel(), dv_g.ravel())

    s.xc_correction.n_qg[:] = 0.0
    s.xc_correction.nc_g[:] = 0.0
    E1 = (s.xc_correction.calculate_energy_and_derivatives([D_p], [H_p]) +
          s.xc_correction.Exc0)

    print name, E1, E2, E1 - E2
    equal(E1, E2, 0.0013)
