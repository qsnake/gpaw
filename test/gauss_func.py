import Numeric as num
from math import pi, sqrt
from gpaw.utilities.tools import coordinates
from gpaw.utilities.gauss import Gaussian
from gpaw.domain import Domain
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities import equal

# Test if multipole works
d  = Domain((14, 15, 16))     # Domain object
N  = 40                       # Number of grid points
Nc = (N, N, N)                # Tuple with number of grid point along each axis
gd = GridDescriptor(d, Nc)    # Grid-descriptor object
xyz, r2 = coordinates(gd)     # Matrix with the square of the radial coordinate
r  = num.sqrt(r2)             # Matrix with the values of the radial coordinate
nH = num.exp(-2 * r) / num.pi # Density of the hydrogen atom
gauss = Gaussian(gd)          # An instance of Gaussian

# Check if Gaussians are made correctly
for gL in range(2, 8):
    g = gauss.get_gauss(gL) # a gaussian of gL'th order
    print '\nGaussian of order', gL
    for mL in range(16):
        m = gauss.get_moment(g, mL) # the mL'th moment of g
        print '  %s\'th moment = %2.6f' %(mL, m)
        equal(m, gL == mL, 1e-4)

# Check the moments of the constructed 1s density
print '\nDensity of Hydrogen atom'
for L in range(4):
    m = gauss.get_moment(nH, L)
    print '  %s\'th moment = %2.6f' %(L, m)
    equal(m, (L == 0) / sqrt(4 * pi), 1e-3)

# Check that it is removed correctly
v = gauss.remove_moment(nH, 0)
m = gauss.get_moment(nH, 0)
print '\nZero\'th moment of compensated Hydrogen density =', m
equal(m, 0., 1e-7)

