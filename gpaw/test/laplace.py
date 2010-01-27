import numpy as np
from gpaw.fd_operators import GUCLaplace as Laplace
from gpaw.grid_descriptor import GridDescriptor

L = 8.0 * 2**0.5
x = L / 2
gd = GridDescriptor((8, 8, 8),
                    [(8.0, 0,0),
                     (4, 3**.5 * 4, 0),
                     (0,0, 9)])
if 1:
    gd = GridDescriptor((8, 8, 8),
                        [(8.0, 0,0),
                         (4, 3**.5 * 4, 0),
                         (4, 3**.5 * 4 / 3, (2.0 / 3)**0.5 * 8)])
#gd = GridDescriptor((8, 8, 8),
#                    [(0,x,x),(x,0,x),(x,x,0)])
b = gd.zeros()
r_Gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
a = ((r_Gv - (4, 6, 8))**2).sum(3)
lap = Laplace(gd, n=2)
print a.itemsize, a.dtype, a.shape
lap.apply(a, b)
print b[3:6, 3:6, 3:6]

