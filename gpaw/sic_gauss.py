
import numpy as np
from numpy import sqrt, pi, exp, sin

import _gpaw
from gpaw import debug
from gpaw.utilities import erf, is_contiguous

# computer generated code:
# use c/bmgs/sharmonic.py::construct_gauss_code(lmax) to generate more
Y_L = [
  '0.28209479177387814',
  '0.48860251190291992 * y',
  '0.48860251190291992 * z',
  '0.48860251190291992 * x',
  '1.0925484305920792 * x*y',
  '1.0925484305920792 * y*z',
  '0.31539156525252005 * (3*z*z-r2)',
  '1.0925484305920792 * x*z',
  '0.54627421529603959 * (x*x-y*y)',
]
gauss_L = [
  'sqrt(a**3*4)/pi * exp(-a*r2)',
  'sqrt(a**5*5.333333333333333)/pi * y * exp(-a*r2)',
  'sqrt(a**5*5.333333333333333)/pi * z * exp(-a*r2)',
  'sqrt(a**5*5.333333333333333)/pi * x * exp(-a*r2)',
  'sqrt(a**7*4.2666666666666666)/pi * x*y * exp(-a*r2)',
  'sqrt(a**7*4.2666666666666666)/pi * y*z * exp(-a*r2)',
  'sqrt(a**7*0.35555555555555557)/pi * (3*z*z-r2) * exp(-a*r2)',
  'sqrt(a**7*4.2666666666666666)/pi * x*z * exp(-a*r2)',
  'sqrt(a**7*1.0666666666666667)/pi * (x*x-y*y) * exp(-a*r2)',
]
gausspot_L = [
  '2.0*1.7724538509055159*erf(sqrt(a)*r)/r',
  '1.1547005383792515*(1.7724538509055159*erf(sqrt(a)*r)-2*sqrt(a)*r*exp(-a*r2))/r/r2*y',
  '1.1547005383792515*(1.7724538509055159*erf(sqrt(a)*r)-2*sqrt(a)*r*exp(-a*r2))/r/r2*z',
  '1.1547005383792515*(1.7724538509055159*erf(sqrt(a)*r)-2*sqrt(a)*r*exp(-a*r2))/r/r2*x',
  '0.5163977794943222*(5.3173615527165481*erf(sqrt(a)*r)-(6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*x*y',
  '0.5163977794943222*(5.3173615527165481*erf(sqrt(a)*r)-(6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*y*z',
  '0.14907119849998599*(5.3173615527165481*erf(sqrt(a)*r)-(6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*(3*z*z-r2)',
  '0.5163977794943222*(5.3173615527165481*erf(sqrt(a)*r)-(6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*x*z',
  '0.2581988897471611*(5.3173615527165481*erf(sqrt(a)*r)-(6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*(x*x-y*y)',
]
# end of computer generated code

class Gaussian:
    """Class offering several utilities related to the generalized gaussians.

    Generalized gaussians are defined by::
    
                       _____                           2  
                      /  1       l!         l+3/2  -a r   l  m
       g (x,y,z) =   / ----- --------- (4 a)      e      r  Y (x,y,z),
        L          \/  4 pi  (2l + 1)!                       l

    where a is the inverse width of the gaussian, and Y_l^m is a real
    spherical harmonic.
    The gaussians are centered in the middle of input grid-descriptor."""
    
    def __init__(self, gd, cell, a=10., w=0.05):
        #
        self.cell  = cell 
        self.gd    = gd
        self.alpha = a
        self.width = w
        #
        # transformation from cartesian to direct coordinates
        self.T    = np.linalg.solve(self.cell.T, np.eye(3))
        self.S    = np.linalg.solve(self.T, np.eye(3))
        #
        # compute the metric from direct coordinates to
        # cartesian ones
        self.M    = np.dot(cell,cell.T)
        #
        self.r    = gd.get_grid_point_coordinates()
        self.x    = np.ndarray(self.r.shape    ,dtype=float)
        self.r2_G = np.ndarray(self.r.shape[1:],dtype=float)
        #
        x = self.x
        r = self.r
        T = self.T
        x[0,:] = T[0,0]*r[0,:] + T[0,1]*r[1,:] + T[0,2]*r[2,:]        
        x[1,:] = T[1,0]*r[0,:] + T[1,1]*r[1,:] + T[1,2]*r[2,:]
        x[2,:] = T[2,0]*r[0,:] + T[2,1]*r[1,:] + T[2,2]*r[2,:]
        #
        return
        
    def get_fields(self, x0, chg, pot, mask,
                    tiny=1e-12,huge=1e+12):
        
        """Constructs and returns matrices containing cartesian coordinates,
        and the square of the distance from the origin.
        
        The origin is placed in the center of the box described by the given
        grid-descriptor 'gd'.
        """
        #
        S      = self.S
        T      = self.T
        M      = self.M
        rc     = self.r
        xc     = self.x
        r2_G   = self.r2_G
        gd     = self.gd
        alpha  = self.alpha
        width  = self.width
        #
        wfac   = 0.5*pi/width
        #
        #x0     = np.array([0.51,0.51,0.51])
        #
        dx     = xc-x0[:,np.newaxis, np.newaxis, np.newaxis]
        #
        dx=np.where(     dx >   0.50, dx - 1.0, dx)
        dx=np.where(     dx <= -0.50, dx + 1.0, dx)
        #
        rsqmax = 1E+20
        for ix in range(xc.shape[1]):
            for iy in range(xc.shape[2]):
                for iz in range(xc.shape[3]):
                    dx0 = dx[:,ix,iy,iz]
                    mdx0 = np.max(np.abs(dx0)) - 0.25
                    if (mdx0 >= 0.0):
                        mask[ix,iy,iz]   =  0.0
                        r2_G[ix,iy,iz]   =  np.dot(dx0,np.dot(M,dx0))
                    elif (mdx0 > -width):
                        mask[ix,iy,iz]   =  sin(wfac*mdx0)**2
                        r2_G[ix,iy,iz]   =  np.dot(dx0,np.dot(M,dx0))
                        rsqmax           =  min(rsqmax,r2_G[ix,iy,iz])
                    else:
                        mask[ix,iy,iz]   =  1.0
                        r2_G[ix,iy,iz]   =  max(np.dot(dx0,np.dot(M,dx0)),tiny)
                        rsqmax           =  max(rsqmax,r2_G[ix,iy,iz])

        a        = alpha/rsqmax
        a1fac    = sqrt(a)
        for ix in range(xc.shape[1]):
            for iy in range(xc.shape[2]):
                for iz in range(xc.shape[3]):
                    #if mask[ix,iy,iz]!=0.0:
                        r2            = r2_G[ix,iy,iz]
                        r             = sqrt(r2)
                        chg[ix,iy,iz] = 2.0*a1fac*a/pi * exp(-a*r2)
                        pot[ix,iy,iz] = 2.0*1.7724538509055159*erf(a1fac*r)/r
                    #else:
                    #    chg[ix,iy,iz] = 0.0
                    #    pot[ix,iy,iz] = 0.0
        
        #
        return
        #
        for ix in range(xc.shape[1]):
            for iy in range(xc.shape[2]):
                for iz in range(xc.shape[3]):
                    print rc[0,ix,iy,iz],rc[1,ix,iy,iz],\
                          rc[2,ix,iy,iz],r2_G[ix,iy,iz],\
                          chg[ix,iy,iz],pot[ix,iy,iz],mask[ix,iy,iz]
                print ''
            print ''
        #
        return 

