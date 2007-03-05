import Numeric as num
from Numeric import sqrt, pi, exp
from tools import coordinates, erf3D

# computer generated code:
# use c/bmgs/sharmonic/construct_python_code(lmax) to generate more
Y_L = ['0.282094791774', '0.488602511903 * y', '0.488602511903 * z', '0.488602511903 * x', '1.09254843059 * x*y', '1.09254843059 * y*z', '0.315391565253 * (3*z*z-r2)', '1.09254843059 * x*z', '0.546274215296 * (x*x-y*y)', ]
gauss_L = ['sqrt(a0**3*4)/pi * exp(-a0*r2)', 'sqrt(a0**5*5.33333333333)/pi * y * exp(-a0*r2)', 'sqrt(a0**5*5.33333333333)/pi * z * exp(-a0*r2)', 'sqrt(a0**5*5.33333333333)/pi * x * exp(-a0*r2)', 'sqrt(a0**7*4.26666666667)/pi * x*y * exp(-a0*r2)', 'sqrt(a0**7*4.26666666667)/pi * y*z * exp(-a0*r2)', 'sqrt(a0**7*0.355555555556)/pi * (3*z*z-r2) * exp(-a0*r2)', 'sqrt(a0**7*4.26666666667)/pi * x*z * exp(-a0*r2)', 'sqrt(a0**7*1.06666666667)/pi * (x*x-y*y) * exp(-a0*r2)', ]
gausspot_L = ['2*sqrt(pi)*erf3D(sqrt(a0)*r)/r', '', '', '', '', '', '', '', '', ]

class Gaussian:
    """Class offering several utilities related to the generalized gaussians.

    Generalized gaussians are defined by::
    
                       _____                             2  
                      /  1       l!          l+3/2  -a0 r    l  m
       g (x,y,z) =   / ----- --------- (4 a0)      e        r  Y (x,y,z),
        L          \/  4 pi  (2l + 1)!                          l

    where a0 is the inverse width of the gaussian, and Y_l^m is a real
    spherical harmonic.
    The gaussians are centered in the middle of input grid-descriptor."""
    
    def __init__(self, gd, a0=21.):
        self.gd = gd
        self.xyz, self.r2 = coordinates(gd)
        self.set_width(a0)

    def set_width(self, a0=21.):
        """Set exponent of exp-function to a0 on the boundary."""
        self.a0 = 4 * a0 / min(self.gd.domain.cell_c)**2
        
    def get_gauss(self, L):
        a0 = self.a0
        x, y, z  = tuple(self.xyz)
        r2 = self.r2
        return eval(gauss_L[L]+'*'+Y_L[L])

    def get_gauss_pot(self, L):
        a0 = self. a0
        r2 = self.r2
        r  = num.sqrt(r2)
        if L == 0:
            return eval(gausspot_L[L])
        else:
            raise NotImplementedError

    def get_moment(self, n, L):
        r2 = self.r2
        x, y, z = tuple(self.xyz)
        return self.gd.integrate(n * eval(Y_L[L]))

    def remove_moment(self, n, L, q=None):
        # determine multipole moment
        if q == None:
            q = self.get_moment(n, L)

        # Don't do anything if moment is less than the tolerance
        if abs(q) < 1e-7:
            return 0.

        # remove moment from input density
        n -= q * self.get_gauss(L)

        # return correction
        return q * self.get_gauss_pot(L)        

    def plot_gauss(self, L):
        from ASE.Visualization.VTK import VTKPlotArray
        cell = num.identity(3, num.Float)
        VTKPlotArray(self.get_gauss(L), cell)

if __name__ == '__main__':
    from gpaw.domain import Domain
    from gpaw.grid_descriptor import GridDescriptor

    # plot gaussian
    ## d  = Domain((2,2,2))   # domain object
    ## N  = 2**5              # number of grid points
    ## Nc = (N,N,N)           # tuple with number of grid point along each axis
    ## gd = GridDescriptor(d,Nc) # grid-descriptor object
    ## gauss = Gaussian(gd)
    ## gauss.plot_gauss(1)

    # test if multipole works
    d  = Domain((14,15,16))   # domain object
    N  = 2**5                 # number of grid points
    Nc = (N,N,N)              # tuple with number of grid point along each axis
    gd = GridDescriptor(d,Nc) # grid-descriptor object
    xyz, r2 = coordinates(gd) # matrix with the square of the radial coordinate
    r  = num.sqrt(r2)         # matrix with the values of the radial coordinate
    nH = num.exp(-2*r)/num.pi # density of the hydrogen atom
    gauss = Gaussian(gd)
    g = gauss.get_gauss(0)

    # check that gaussians have the correct moment
    print gd.integrate(g) / (2*sqrt(pi)), gauss.get_moment(g, 0), 1
    v = gauss.remove_moment(g, 0)
    print gd.integrate(g)/(2*sqrt(pi)), gauss.get_moment(g, 0), 0

    # check the moments of the constructed 1s density
    print ''
    print gauss.get_moment(nH, 0), 1/(2*sqrt(pi)) 
    print gauss.get_moment(nH, 1), 0 
    print gauss.get_moment(nH, 2), 0
    print gauss.get_moment(nH, 3), 0
