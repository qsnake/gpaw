import Numeric as num
from Numeric import sqrt, pi, exp
from gpaw.utilities.tools import coordinates
from gpaw.utilities.tools import erf3D as erf

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
  '0.59004358992664352 * (-y*y*y+3*x*x*y)',
  '2.8906114426405538 * x*y*z',
  '0.45704579946446577 * (-y*r2+5*y*z*z)',
  '0.3731763325901154 * (5*z*z*z-3*z*r2)',
  '0.45704579946446577 * (5*x*z*z-x*r2)',
  '1.4453057213202769 * (x*x*z-y*y*z)',
  '0.59004358992664352 * (x*x*x-3*x*y*y)',
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
  'sqrt(a**9*0.10158730158730159)/pi * (-y*y*y+3*x*x*y) * exp(-a*r2)',
  'sqrt(a**9*2.4380952380952383)/pi * x*y*z * exp(-a*r2)',
  'sqrt(a**9*0.060952380952380952)/pi * (-y*r2+5*y*z*z) * exp(-a*r2)',
  'sqrt(a**9*0.040634920634920635)/pi * (5*z*z*z-3*z*r2) * exp(-a*r2)',
  'sqrt(a**9*0.060952380952380952)/pi * (5*x*z*z-x*r2) * exp(-a*r2)',
  'sqrt(a**9*0.60952380952380958)/pi * (x*x*z-y*y*z) * exp(-a*r2)',
  'sqrt(a**9*0.10158730158730159)/pi * (x*x*x-3*x*y*y) * exp(-a*r2)',
]

gausspot_L = [
  '2.0*(1.7724538509055159*erf(sqrt(a)*r))/r*1',
  '1.1547005383792515*(1.7724538509055159*erf(sqrt(a)*r)-(+2)*sqrt(a)*r*exp(-a*r2))/r/r2**1*y',
  '1.1547005383792515*(1.7724538509055159*erf(sqrt(a)*r)-(+2)*sqrt(a)*r*exp(-a*r2))/r/r2**1*z',
  '1.1547005383792515*(1.7724538509055159*erf(sqrt(a)*r)-(+2)*sqrt(a)*r*exp(-a*r2))/r/r2**1*x',
  '0.5163977794943222*(5.3173615527165481*erf(sqrt(a)*r)-(+6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*x*y',
  '0.5163977794943222*(5.3173615527165481*erf(sqrt(a)*r)-(+6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*y*z',
  '0.14907119849998599*(5.3173615527165481*erf(sqrt(a)*r)-(+6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*(3*z*z-r2)',
  '0.5163977794943222*(5.3173615527165481*erf(sqrt(a)*r)-(+6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*x*z',
  '0.2581988897471611*(5.3173615527165481*erf(sqrt(a)*r)-(+6+4*(sqrt(a)*r)**2)*sqrt(a)*r*exp(-a*r2))/r/r2**2*(x*x-y*y)',
  '0.039840953644479787*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*(-y*y*y+3*x*x*y)',
  '0.19518001458970666*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*x*y*z',
  '0.03086066999241838*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*(-y*r2+5*y*z*z)',
  '0.025197631533948481*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*(5*z*z*z-3*z*r2)',
  '0.03086066999241838*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*(5*x*z*z-x*r2)',
  '0.097590007294853329*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*(x*x*z-y*y*z)',
  '0.039840953644479787*(26.586807763582737*erf(sqrt(a)*r)-(+30+20*(sqrt(a)*r)**2+8*(sqrt(a)*r)**4)*sqrt(a)*r*exp(-a*r2))/r/r2**3*(x*x*x-3*x*y*y)',
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
    
    def __init__(self, gd, a=19.):
        self.gd = gd
        self.xyz, self.r2 = coordinates(gd)
        self.set_width(a)

    def set_width(self, a):
        """Set exponent of exp-function to -a on the boundary."""
        self.a = 4 * a / min(self.gd.domain.cell_c)**2
        
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
            self.r = num.sqrt(r2)
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

    def plot_gauss(self, L):
        from ASE.Visualization.VTK import VTKPlotArray
        cell = num.identity(3, num.Float)
        VTKPlotArray(self.get_gauss(L), cell)
