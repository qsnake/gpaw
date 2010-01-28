
import numpy as np
from numpy import sqrt, pi, exp

import _gpaw
from gpaw import debug
from gpaw.utilities.tools import coordinates
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
    
    def __init__(self, gd, a=19., center=None):
        assert gd.orthogonal
        self.gd = gd
        self.xyz, self.r2 = coordinates(gd, center)
        self.set_width(a)

    def set_width(self, a):
        """Set exponent of exp-function to -a on the boundary."""
        self.a = 4 * a / min(self.gd.cell_cv.diagonal())**2
        
    def get_gauss(self, L):
        a = self.a
        x, y, z  = tuple(self.xyz)
        r2 = self.r2
        return eval(gauss_L[L])
    
    def get_gauss_pot(self, L):
        a = self. a
        x, y, z  = tuple(self.xyz)
        r2 = self.r2
        if not hasattr(self, 'r'):
            self.r = np.sqrt(r2)
        r = self.r
        return eval(gausspot_L[L])

    def get_moment(self, n, L):
        r2 = self.r2
        x, y, z = tuple(self.xyz)
        return self.gd.integrate(n * eval(Y_L[L]))

    def remove_moment(self, n, L, q=None):
        # Determine multipole moment
        if q == None:
            q = self.get_moment(n, L)

        # Don't do anything if moment is less than the tolerance
        if abs(q) < 1e-7:
            return 0.

        # Remove moment from input density
        n -= q * self.get_gauss(L)

        # Return correction
        return q * self.get_gauss_pot(L)


def gaussian_wave(r_vG, r0_v, sigma, k_v=None, A=None, dtype=float,
                  out_G=None):
    """Generates function values for atom-centered Gaussian waves.

    ::
    
                         _ _
        _            / -|r-r0|^2 \           _ _
      f(r) = A * exp( ----------- ) * exp( i k.r )
                     \ 2 sigma^2 /

    If the parameter A is not specified, the Gaussian wave is normalized::

                                                  oo
           /    ____        \ -3/2               /       _  2  2
      A = (    /    '        )        =>    4 pi | dr |f(r)|  r  = 1
           \ \/  pi   sigma /                    /
                                                   0

    Parameters:

    r_vG: ndarray
        Set of coordinates defining the grid positions.
    r0_v: ndarray
        Set of coordinates defining the center of the Gaussian envelope.
    sigma: float
        Specifies the spatial width of the Gaussian envelope.
    k_v: ndarray or None
        Set of reciprocal lattice coordinates defining the wave vector.
        An argument of None is interpreted as the gamma point i.e. k_v=0.
    A: float, complex or None
        Specifies the amplitude of the Gaussian wave. Normalizes if None.
    dtype: type, defaults to float
        Specifies the output data type. Only returns the real-part if float.
    out_G: ndarray or None
        Optional pre-allocated buffer to fill in values. Allocates if None.

    """
    if k_v is None:
        k_v = np.zeros(r0_v.shape)

    if A is None:
        # 4*pi*int(exp(-r^2/(2*sigma^2))^2 * r^2, r=0...infinity)
        # = sigma^3*pi^(3/2) = 1/A^2 -> A = (sqrt(Pi)*sigma)^(-3/2)
        A = 1/(sigma*np.pi**0.5)**1.5

    if debug:
        assert is_contiguous(r_vG, float)
        assert is_contiguous(r0_v, float)
        assert is_contiguous(k_v, float)
        assert r_vG.ndim >= 2 and r_vG.shape[0] > 0
        assert r0_v.ndim == 1 and r0_v.shape[0] > 0
        assert k_v.ndim == 1 and k_v.shape[0] > 0
        assert (r_vG.shape[0],) == r0_v.shape == k_v.shape
        assert sigma > 0

    if out_G is None:
        out_G = np.empty(r_vG.shape[1:], dtype=dtype)
    elif debug:
        assert is_contiguous(out_G)
        assert out_G.shape == r_vG.shape[1:]

    # slice_v2vG = [slice(None)] + [np.newaxis]*3
    # gw = lambda r_vG, r0_v, sigma, k_v, A=1/(sigma*np.pi**0.5)**1.5: \
    #    * np.exp(-np.sum((r_vG-r0_v[slice_v2vG])**2, axis=0)/(2*sigma**2)) \
    #    * np.exp(1j*np.sum(np.r_vG*k_v[slice_v2vG], axis=0)) * A
    _gpaw.utilities_gaussian_wave(A, r_vG, r0_v, sigma, k_v, out_G)
    return out_G

