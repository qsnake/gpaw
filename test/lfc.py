import numpy as np

from gpaw.lfc import BasisFunctions
from gpaw.grid_descriptor import GridDescriptor
from gpaw.domain import Domain
from gpaw.spline import Spline

# Make two 's' splines of this form:
#
#                     :             ^ y
#       ---------   --:------       |
#      /         \ /  :      \     z+--> x
#     /           X   :       \
# +--/-----------/-\--:--------\----------+
# | |           |   | |         |         |
# | |           |   | :         |         |
# | |       x   |   | : x       |         |
# | |           |   | :         |         |
# | |           |   | :         |         |
# +--\-----------\-/--:--------/----------+
#     \           X   :       /
#      \         / \  :      /
#       ---------   --:------
#                     :
#
# ':' is the domain wall if split on two cpu's


domain = Domain(cell=[10., 2., 2.], pbc=(0, 1, 1))
#domain.set_decomposition(comm=, parsize_c=(2, 1, 1))
gd = GridDescriptor(domain, N_c=[40, 8, 8])
pos_ac = np.array([[.25, .5, .5], [.55, .5, .5]])
kpts_kc = None
spline = Spline(l=0, rmax=2.0, f_g=np.array([1, 0.9, 0.1, 0.0]),
                r_g=None, beta=None, points=25)
spline_aj = [[spline] for pos_c in pos_ac]

bfs = BasisFunctions(gd, spline_aj)
if kpts_kc is not None:
    bfs.set_k_points(kpts_kc)
bfs.set_positions(pos_ac)
