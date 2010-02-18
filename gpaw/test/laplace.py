from math import sin, cos, pi
import numpy as np
from gpaw.fd_operators import GUCLaplace as Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import size

cells = [# distorted hexagonal
    (4, [(1, 0, 0),
         (1.02*cos(pi/3-0.02), 1.02*sin(pi/3-0.02), 0),
         (0, 0, 1.0)]),
    # hexagonal
    (4, [(1, 0, 0),
         (0.5, 3**0.5 / 2, 0),
         (0, 0, 1.1)]),
    # fcc
    (6, [(1, 0, 0),
         (0.5, 3**0.5 / 2, 0),
         (0.5, 3**0.5 / 6, (2.0 / 3)**0.5)]),
    # fcc
    (6, [(0, 1, 1), (1, 0, 1), (1, 1, 0)]),
    # bcc
    (4, [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]),
    # sc
    (3, [1, 1, 1]),
    # distorted sc
    (6, [(1, 0, 0), (0.01, 1, 0), (0, 0.02, 1)]),
    # rocksalt
    (6, [(2.*np.sqrt(1./3.), np.sqrt(1./8.), -np.sqrt(1./24.)), (2.*np.sqrt(1./3.), -np.sqrt(1./8.), -np.sqrt(1./24.)), (2.*np.sqrt(1./3.), 0., np.sqrt(1./6.))])]

if size == 1:
    for D, cell in cells:
        print cell
        for n in range(1, 6):
            N = 2 * n + 2
            gd = GridDescriptor((N, N, N), cell)
            b_g = gd.zeros()
            r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
            c_v = gd.cell_cv.sum(0) / 2
            r_gv -= c_v
            lap = Laplace(gd, n=n)
            assert lap.npoints == D * 2 * n + 1
            for m in range(0, 2 * n + 1):
                for ix in range(m + 1):
                    for iy in range(m - ix + 1):
                        iz = m - ix - iy
                        a_g = (r_gv**(ix, iy, iz)).prod(3)
                        if ix + iy + iz == 2 and max(ix, iy, iz) == 2:
                            r = 2.0
                        else:
                            r = 0.0
                        lap.apply(a_g, b_g)
                        e = b_g[n + 1, n + 1, n + 1] - r
                        assert abs(e) < 1e-12
