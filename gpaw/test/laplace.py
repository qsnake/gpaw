from math import sin, cos, pi
import numpy as np
from gpaw.fd_operators import NewGUCLaplace as Laplace
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import size

cells = [
    ('distorted hexagonal', 4,
     [(1, 0, 0),
      (1.02 * cos(pi / 3 - 0.02), 1.02 * sin(pi / 3 - 0.02), 0),
      (0, 0, 1.0)]),
    ('hexagonal', 4,
     [(1, 0, 0),
      (0.5, 3**0.5 / 2, 0),
      (0, 0, 1.1)]),
    ('fcc', 6,
     [(0, 1, 1),
      (1, 0, 1),
      (1, 1, 0)]),
    ('fcc-alternative', 6,
     [(1, 0, 0),
      (0.5, 3**0.5 / 2, 0),
      (0.5, 3**0.5 / 6, (2.0 / 3)**0.5)]),
    ('bcc', 4,
     [(-1, 1, 1),
      (1, -1, 1),
      (1, 1, -1)]),
    ('sc', 3,
     [1.1, 1.02, 1.03]),
    ('distorted sc', 6,
     [(1, 0, 0),
      (0.01, 1, 0),
      (0, 0.02, 1)]),
    ('rocksalt', 6,
     [(2 * np.sqrt(1.0 / 3), np.sqrt(1.0 / 8), -np.sqrt(1.0/ 24)),
      (2 * np.sqrt(1.0 / 3), -np.sqrt(1.0 / 8), -np.sqrt(1.0 / 24)),
      (2 * np.sqrt(1.0 / 3), 0, np.sqrt(1.0 / 6))]),
    ('nasty', 6,
     [(1, 0, 0),
      (0.0001, 1.03, 0),
      (0.0001, 0.0001, 1.0)]),
    ('Mike', 6,
     5 * np.array([(5.565 / 28, 0, 0),
                   (0.0001 / 28, 5.565 / 28, 0),
                   (0.0001 / 24, 0.0001 / 24, 4.684 / 24)]))
    ]

if size == 1:
    for name, D, cell in cells:
        print '------------------'
        print name, D
        print cell[0]
        print cell[1]
        print cell[2]
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
                        assert abs(e) < 2e-12, e
