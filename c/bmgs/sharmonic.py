import Numeric as num
from Numeric import pi, sqrt
from tools import factorial
from tools import Rational as Q

"""
This is a script designed for construction of the real solid spherical
harmonics (RSSH) in cartesian form. These can be written as:

       m    |m|  l  |m|
Y  =  Y  = C    r  P  (cos theta) Phi (phi)
 L     l    l       l                m

where C_l^|m| is a normalization constant
P_l^|m| is the associatied legendre polynomium
and
            / cos(m phi) , m > 0
Phi (phi) = |   1        , m = 0
   m        \ sin(-m phi), m < 0

The first few harmonics are listed below:
+----+---------------------+-__---------------------------------------+
|  L | l |  m | r^l * Y    | \/ (r^l * Y)                             |
+----+---s----+------------+------------------------------------------+
|  0 | 0 |  0 | 1          | (0, 0, 0)                                |
+----+---p----+------------+------------------------------------------+
|  1 | 1 | -1 | y          | (0, 1, 0)                                |
|  2 | 1 |  0 | z          | (0, 0, 1)                                |
|  3 | 1 |  1 | x          | (1, 0, 0)                                |
+----+---d----+------------+------------------------------------------+
|  4 | 2 | -2 | xy         | ( y,  x,  0)                             |
|  5 | 2 | -1 | yz         | ( 0,  z,  y)                             |
|  6 | 2 |  0 | 3z^2-r^2   | (-x, -y, 2z)                             |
|  7 | 2 |  1 | xz         | ( z,  0,  x)                             |
|  8 | 2 |  2 | x^2-y^2    | ( x, -y,  0)                             |
+----+---f----+------------+------------------------------------------+
|  9 | 3 | -3 | 3x^2y-y^3  | (          2xy,       x^2-y^2,        0) |
| 10 | 3 | -2 | xyz        | (           yz,            xz,       xy) |
| 11 | 3 | -1 | 5yz^2-yr^2 | (         -2xy, 4z^2-x^2-3y^2,      8yz) |
| 12 | 3 |  0 | 5z^3-3zr^2 | (         -2xz,          -2yz, 3z^2-r^2) |
| 13 | 3 |  1 | 5xz^2-xr^2 | (4z^2-3x^2-y^2,          -2xy,      8xz) |
| 14 | 3 |  2 | x^2z-y^2z  | (          2xz,          -2yz,  x^2-y^2) |
| 15 | 3 |  3 | x^3-3xy^2  | (      x^2-y^2,          -2xy,        0) |
+----+--------+----------+--------------------------------------------+

Y_lm is represented as a polynomium in x, y, and z

The function consists of three parts: a normalization constant accessed by
class 'Normalization(l, m)', a polynomium in z accessed with method
'legendre(l, m)', and a polynomium in x and y accessed with method 'Phi(l, m)'

The normalization and the z-polynomium are both invariant of the sign of m
The z-polynomium has powers l-|m|, l-|m|-2, l-|m|-4, l-..., i.e. it is strictly odd (even) if l-|m| is odd (even)
The combined power of x and y is |m| in all terms
"""

#--------------------------- RELEVANT USER METHODS ---------------------------
def L_to_lm(L):
    """convert L index to (l, m) index"""
    l = 0
    while L / (l+1.)**2 >= 1:  l += 1
    return l, L - l**2 - l

def lm_to_L(l,m):
    """convert (l, m) index to L index"""
    return l**2 + l + m

def Y_to_string(l, m, deriv=None, multiply=None, numeric=False):
    """                                 l
       Return string representation of r  * Y (x, y, z)
                                             lm
       if deriv is None.
       If deriv == q, return string is the derivative of above with respect
       to x, y or z if q is 0, 1 or 2 respectively.
       multiply=q indicates that the entire expression should be multiplied by
       q
       numeric=True/False indicates wheter the normalization constant should be
       written as a number or an algebraic expression.
    """
    assert deriv == None or deriv in range(3)
    assert multiply == None or multiply in range(3)
    
    if deriv == None:
        norm, xyzs = Y_collect(l, m)
    else:
        norm, xyzs = dYdq(l, m, deriv)

    if multiply != None:
        xyzs = q_times_xyzs(xyzs, multiply)

    string = to_string(l, xyzs, deriv != None)
    if string == '0': return '0'
    else: return norm.tostring(numeric) + (' * ' + string) * (string != '1')

def gauss_to_string(l, m, numeric=False):
    """Return string representation of the generalized gaussian
                      _____                             2  
        m            /  1       l!          l+3/2  -a0 r    l  m
       g (x,y,z) =  / ----- --------- (4 a0)      e        r  Y (x,y,z)
        l         \/  4 pi  (2l + 1)!                          l

       numeric=True/False indicates wheter the normalization constant should be
       written as a number or an algebraic expression.
    """
    norm, xyzs = Y_collect(l, m)

    ng = Q(2**(2*l+3) * factorial(l),2 * factorial(2 * l + 1))
    norm.multiply(ng)

    string = to_string(l, xyzs)
    string = (' * ' + string) * (string != '1')
    if numeric:
        snorm = str(eval(str(norm.norm)))
    else:
        snorm = str(norm.norm)
    string = 'sqrt(a0**%s*'%(2*l+3) + snorm + ')/pi' + string
    string += ' * exp(-a0*r2)'

    return string

#----------------------------- TECHNICAL METHODS -----------------------------
def to_string(l, xyzs, deriv=False):
    """Return string representation of of xyz dictionary"""
    if xyzs == {}: return '0'
    out = ''

    for xyz, coef in zip(xyzs.keys(), xyzs.values()):
        x, y, z = xyz
        r = l - x - y - z - deriv
        one = abs(coef) != 1 or (x == 0 and y == 0 and z == 0 and r == 0)
        out += sign(coef) + str(abs(coef)) * one
        out += ('*x'*x + '*y'*y + '*z'*z + '*r2'*(r/2))[1 - one:]

    if out[0] == '+': out = out[1:]
    if len(xyzs) > 1: out = '(' + out + ')'
    return out

def sign(x):
    """Return string representation of the sign of x"""
    if x >= 0: return '+'
    else: return '-'

class Normalization:
    """Determine normalization factor of spherical harmonic
                   ______________
             /    / 2l+1   (l-m)!
             |   /  ---- * ------  , m != 0
             | \/   2 pi   (l+m)!
       C  = <      _____
        L    |    / 2l+1
             |   /  ----           , m = 0
             \ \/   4 pi
    """
    def __init__(self, l, m):
        m = abs(m)
        if m == 0:
            self.norm = Q(2 * l + 1, 4)
        else:
            self.norm = Q((2 * l + 1) * factorial(l - m), 2 * factorial(l + m))

    def multiply(self, x):
        self.norm *= x**2

    def tostring(self, numeric=False):
        n = self.norm
        sn = sqrt(float(n))
        if int(sn) == sn:
            string = str(sn) + '/sqrt(pi)'
        else:
            string = 'sqrt(' + str(n.nom) + \
                     ('./' + str(n.denom)) * (n.denom != 1) + '/pi)'
        if numeric:
            return str(eval(string))
        else:
            return string

def legendre(l, m):
    """Determine z dependence of spherical harmonic.
       Returns vector, where the p'th element is the coefficient of
       z^p r^(l-|m|-p).
    """
    m = abs(m)
    assert l >= 0 and 0 <= m <=l
    result = num.zeros(l - m + 1, 'O')
    if l == m == 0:
        """Use that
             0
            P (z) = 1
             0
        """
        result[0] = Q(1)
    elif l == m:
        """Use the recursion relation
            m              m-1
           P (z) = (2m-1) P   (z)
            m              m-1
        """
        result[:] += (2 * m - 1) * legendre(l - 1, m - 1)
    elif l == m + 1:
        """Use the recursion relation
            l-1        l
           P  (z) = z P (z)
            l          l
            
        """
        result[1:] += legendre(l, l)
    else:
        """Use the recursion relation
            m     2l-1    m       l+m-1  2  m
           P (z)= ---- z P  (z) - ----- r  P  (z)
            l      l-m    l-1      l-m      l-2
        """
        result[1:] += num.multiply(legendre(l - 1, m), Q(2 * l - 1, l - m))
        result[:(l - 2) - m + 1] -= num.multiply(legendre(l - 2, m),
                                                 Q(l + m - 1, l - m))
    return result

def Phi(m):
    """ Determine the x and y dependence of the spherical harmonics from
                      |m|   |m|
                   / r   sin  (theta) cos(|m| phi), m > 0
                   |
       Phi (phi) = | 1                            , m = 0
          m        |
                   |  |m|   |m|
                   \ r   sin  (theta) sin(|m| phi), m < 0
       Returns dictionary of format {(i, j): c} where c is the coefficient
       of x^i y^j
    """
    if   m ==  0: return {(0, 0): 1} # use that Phi_0  = 1
    elif m ==  1: return {(1, 0): 1} # use that Phi_1  = x
    elif m == -1: return {(0, 1): 1} # use that Phi_-1 = y
    else:
        """Use the recurrence formula
        
           m > 0:  Phi (x,y) = x Phi   (x,y) - y Phi   (x,y)
                     |m|           |m|-1           1-|m|

           m < 0:  Phi (x,y) = y Phi   (x,y) + x Phi   (x,y)
                     |m|           |m|-1           1-|m|           
        """
        xys  = {}
        phi1 = Phi(abs(m) - 1)
        phi2 = Phi(1 - abs(m))
        for x, y in phi1:
            new = (x + (m > 0), y + (m < 0))
            xys[new] = xys.get(new, 0) +  phi1[(x, y)]
        for x,y in phi2:
            new = (x + (m < 0), y + (m > 0))
            sign = 2 * (m < 0) - 1
            xys[new] = xys.get(new, 0) + sign * phi2[(x, y)]
        return xys

def Y_collect(l, m):
    """Collect all necessary parts of spherical harmonic and return in
       simplified format.
       Return dictionary xyzs has format {(i, j, k): c} where c is the
       coefficient of x^i y^j z^k r^(l-|m|-k), or (since i+j = |m|) the
       coefficient of x^i y^j z^k r^(l-i-j-k), from which it is clear that all
       terms are of power l in x, y and z collectively.
    """
    zs = legendre(l, m)
    xys = Phi(m)

    xyzs = {}
    for xy in xys:
        if xys[xy] != 0:
            for p in range(len(zs)):
                if zs[p] != 0:
                    xyzs[xy + (p,)] = xys[xy] * zs[p]

    # get normalization constant and simplify
    norm = Normalization(l, m)
    norm.multiply(simplify(xyzs))
    
    return norm, xyzs

def dYdq(l, m, q):
    """Returns a normalization constant, and a dictionary discribing
       the functional form of the derivative of r^l Y_l^m(x,y,z) with
       respect to x, y or z if q is either 0, 1 or 2 respectively. The
       format of the output dictionary is {(i, j, k): c}, where c is the
       coefficient of x^i y^j z^k r^(l-i-j-k-1).
    """
    norm, xyzs = Y_collect(l, m)
    dxyzs = {}
    
    for xyz, coef in zip(xyzs.keys(), xyzs.values()):
        x, y, z = xyz
        r = l - x - y - z

        if xyz[q] != 0:
            dxyz = list(xyz)
            dxyz[q] -= 1
            dxyz = tuple(dxyz)
            dxyzs[dxyz] = dxyzs.get(dxyz, 0) + xyz[q] * coef
        
        if r != 0:
            dxyz = list(xyz)
            dxyz[q] += 1
            dxyz = tuple(dxyz)
            dxyzs[dxyz] = dxyzs.get(dxyz, 0) + r * coef

    # remove zeros from list
    for dxyz in dxyzs.keys():
        if dxyzs[dxyz] == 0: dxyzs.pop(dxyz)

    # simplify
    if dxyzs != {}: norm.multiply(simplify(dxyzs))
    return norm, dxyzs

def simplify(xyzs):
    """Rescale coeeficients to smallest integer value"""
    norm = Q(1)
    numxyz = num.array(xyzs.values())

    # up-scale all 'xyz' coefficients to integers
    for xyz in numxyz:
        numxyz *= xyz.denom
        norm /= xyz.denom

    # determine least common divisor for 'xyz' coefficients
    dmax = 1
    num_max = max(abs(num.floor(numxyz)))
    for d in range(2, num_max + 1):
        test = numxyz / d
        if num.alltrue(test == num.floor(test)): dmax = d

    # Update simplified dictionary
    norm *= dmax
    for i, xyz in enumerate(xyzs):
        xyzs[xyz] = numxyz[i] / dmax

    return norm

def q_times_xyzs(xyzs, q):
    qxyzs = {}
    for xyz, c in zip(xyzs.keys(), xyzs.values()):
        qxyz = list(xyz)
        qxyz[q] += 1
        qxyz = tuple(qxyz)
        
        qxyzs[qxyz] = c
    return qxyzs

#--------------------- TEST AND CODE CONSTRUCTING METHODS ---------------------
def orthogonal(L1, L2):
    """Perform the integral
          2pi pi
           /  /
       I = |  |sin(theta) d(theta) d(phi) Y (theta, phi) * Y (theta, phi)
           /  /                            L1               L2
           0  0
       which should be a kronecker delta in L1 and L2
    """
    I = 0.0
    N = 40

    for theta in num.arange(0, pi, pi / N):
        for phi in num.arange(0, 2 * pi, 2 * pi / N):
            x = num.cos(phi) * num.sin(theta)
            y = num.sin(phi) * num.sin(theta)
            z = num.cos(theta)
            r2 = x*x + y*y + z*z
            
            Y1 = eval(Y_to_string(*L_to_lm(L1)))
            Y2 = eval(Y_to_string(*L_to_lm(L2)))

            I += num.sin(theta) * Y1 * Y2
    I *= 2 * (pi / N)**2

    return I

def check_orthogonality():
    """Check orthogonality for all combinations of the first 10 harmonics"""
    N = 10
    all_passed = True
    for L1 in range(N+1):
        for L2 in range(L1, N+1):
            I = orthogonal(L1, L2)
            passed =  abs(I - (L1 == L2)) < 3e-3
            all_passed *= passed
            print 'L1 = %s,  L2 = %s, passed = %s, I = %s' %(L1, L2, passed, I)
    if all_passed: print 'All tests passed'
    else: print 'Some tests failed'

def symmetry1(lmax, display=True):
    """Make dictionary of format
       diff = {(l1, m1, q1): (nrel, l2, m2, q2)}
       indicating that
            m1              m2
         d Y             d Y
            l1              l2
         ------ = nrel * ------
          d q1            d q2       
    """
    diff = {} # diff[(l1, m1, q1)] = (nrel, l2, m2, q2)
    unique_L = [] # unique_L[L] = (l, m, q, norm, dxyzs)
    for L in range((lmax + 1)**2):
        l, m = L_to_lm(L)
        for q in range(3):
            identical = False
            name = (l, m, 'xyz'[q])

            norm, dxyzs = dYdq(l, m, q)

            for unique in unique_L:
                if dxyzs == unique[4]:
                    diff[name] = (norm / unique[3],) + unique[0:3]
                    identical = True
                    break
            if identical == False:
                unique_L.append(name + (norm, dxyzs))
    if display:
        for key, value in zip(diff.keys(), diff.values()):
            print str(key) + ' = ' + str(value[0]) + ' * ' + str(value[1:])
    else: return diff

def symmetry2(l, display=True):
    """Make dictionary of format
       diff = {(l1, m1, q1): (nrel, l2, m2, q2)}
       indicating that
              m1              m2
           d Y             d Y
              l1              l2
           ------ = nrel * ------
            d q1            d q2
       and
                m1                m2
          q1 * Y   = nrel * q2 * Y
                l1                l2
    """
    diff = {} # diff[(l1, m1, q1)] = (nrel, l2, m2, q2)
    unique_L = [] # unique_L[L] = (l, m, q, dnorm, dxyzs, qnorm, qxyzs)
    for m in range(-l, l+1):
        for q in range(3):
            identical = False
            name = (l, m, q)

            qnorm, xyzs = Y_collect(l, m)
            qxyzs = q_times_xyzs(xyzs, q)
            dnorm, dxyzs = dYdq(l, m, q)
            
            for unique in unique_L:
                if dxyzs == unique[4] and qxyzs == unique[6]:
                    dnrel = dnorm / unique[3]
                    qnrel = qnorm / unique[5]
                    print dnrel == qnrel
                    if dnrel == qnrel:
                        diff[name] = (dnrel,) + unique[0:3]
                        identical = True
                        break
            if identical == False:
                unique_L.append(name + (dnorm, dxyzs, qnorm, qxyzs))
    if display:
        for key, value in zip(diff.keys(), diff.values()):
            print str(key) + ' = ' + str(value[0]) + ' * ' + str(value[1:])
    else: return diff

def construct_c_code(file='temp.c', lmax=2):
    txt = '//Computer generated code! Hands off!'
    start_func = """
    
// inserts values of f(r) r^l Y_lm(theta, phi) in elements of input array 'a'
void bmgs_radial3(const bmgsspline* spline, int m, 
		  const int n[3], 
		  const double C[3],
		  const double h[3],
		  const double* f, double* a)
{
  int l = spline->l;
  if (l == 0)
    for (int q = 0; q < n[0] * n[1] * n[2]; q++)
      a[q] = 0.28209479177387814 * f[q];
"""
    start_deriv = """

// insert values of
// d( f(r) * r^l Y_l^m )                           d( r^l Y_l^m )
// --------------------- = g(r) q r^l Y_l^m + f(r) --------------
//        dq                                             dq
// where q={x, y, z} and g(r) = 1/r*(df/dr)
void bmgs_radiald3(const bmgsspline* spline, int m, int q, 
		  const int n[3], 
		  const double C[3],
		  const double h[3],
		  const double* f, const double* g, double* a)
{
  int q = 0;
  int l = spline->l;
"""
    start_case = """
    {
      int q = 0;
      double x = C[0];
      for (int i0 = 0; i0 < n[0]; i0++)
        {
          double y = C[1];
          for (int i1 = 0; i1 < n[1]; i1++)
            {
              double z = C[2];
	      for (int i2 = 0; i2 < n[2]; i2++, q++)
		{
"""
    end_case = """
                  z += h[2];
		}
	      y += h[1];
	    }
	  x += h[0];
	}
    }
"""
    
    # insert code for evaluating the function
    txt += start_func
    for l in range(1, lmax + 1):
        txt += '  else if (l == %s)' %l
        txt += start_case
        case = ''
        for m in range(-l, l+1):
            if m == -l: case += ' ' * 18 + 'if (m == %s)\n' %m
            elif m == l: case += '\n' + ' ' * 18 +'else\n'
            else: case += '\n' + ' ' * 18 + 'else if (m == %s)\n' %m
            case += ' ' * 20 + 'a[q] = f[q] * '
            case += Y_to_string(l,m) + ';'
        if 'r2' in case: txt += ' ' * 18 + 'double r2 = x*x+y*y+z*z;\n'
        txt += case
        txt += end_case
    txt += """  else
    {
      printf Error: Requested l quantum number not implemented yet!
      printf run (gridpaw.sharmonic) construct_c_code(file=temp.c, lmax=3)
      printf to generate the required code
      assert 0 == 1
    }
}
"""
    
    # insert code for evaluating the derivative
    txt += start_deriv
    for q in range(3):
        txt += '  // ' + 'xyz'[q] + '\n'
        for l in range(0, lmax + 1):
            if l == 0 and q == 0:
                txt += '  if (q == 0 && l == 0)'
            else: txt += '  else if (q == %s && l == %s)' %(q, l)
        
            txt += start_case
            case = ''
            for m in range(-l, l+1):
                if m == -l: case += ' ' * 18 + 'if (m == %s)\n' %m
                elif m == l: case += '\n' + ' ' * 18 + 'else\n'
                else: case += '\n' + ' ' * 18 + 'else if (m == %s)\n' %m
                case += ' ' * 20 + 'a[q] = g[q] * '
                case += Y_to_string(l, m, multiply=q)
                diff = Y_to_string(l, m, deriv=q)
                if diff != '0':
                    case += ' + f[q] * ' + diff
                case += ';'
            if 'r2' in case: txt += ' ' * 18 + ' double r2 = x*x+y*y+z*z;\n'
            txt += case
            txt += end_case
    txt += """  else
    {
      printf Error: Requested l quantum number not implemented yet!
      printf run (gridpaw.sharmonic) construct_c_code(file=temp.c, lmax=3)
      printf to generate the required code
      assert 0 == 1
    }
}
"""
    f = open(file, 'w')
    print >>f, txt
    f.close()
    
def construct_python_code(file='temp.py', Lmax=8):
    out= 'Y_L = ['
    for L in range(Lmax + 1):
        l, m = L_to_lm(L)
        out+= '\'' + Y_to_string(l, m, numeric=True) + '\', '
    out += ']'

    out += '\ngauss_L = ['
    for L in range(Lmax + 1):
        l, m = L_to_lm(L)
        out += '\'' + gauss_to_string(l, m, numeric=True) + '\', '
    out += ']'
    
    out += '\ngausspot_L = ['
    for L in range(Lmax + 1):
        l, m = L_to_lm(L)
        if L == 0:
            out += '\'2*sqrt(pi)*erf3D(sqrt(a0)*r)/r\', '
        else:
            out += '\'' + '\', '
    out += ']'
    
    f = open(file, 'w')
    print >>f, out
    f.close()        
    
def plot_spherical(l, m):
    # for l in range(5): for m in range(-l, l+1): plot_spherical(l,m)
    Ntheta = 45
    Nphi = 90
    eps = 1e-7
    f = open('Y_%s_%s.dat'%(l, m), 'w')
    for theta in num.arange(0, num.pi+eps, num.pi/Ntheta):
        phis = ''
        for phi in num.arange(0, 2*num.pi+eps, 2*num.pi/Nphi):
            x = num.cos(phi) * num.sin(theta)
            y = num.sin(phi) * num.sin(theta)
            z = num.cos(theta)
            r2 = x*x + y*y + z*z
            assert abs(r2 - 1) < eps
            phis += str(eval(Y_to_string(l, m))) + ' '
        print >>f, phis
    f.close()
    """ For plotting in Matlab, write function:
function plot_spherical(l, m)

Ylm = load(sprintf('Y_%d_%d.dat',l,m));
[Nt, Np] = size(Ylm);
[PH, TH] = meshgrid(0:2*pi/(Np-1):2*pi, pi/2:-pi/(Nt-1):-pi/2);
[X, Y, Z] = sph2cart(PH, TH, Ylm.^2);

fig = figure(1);
surf(X, Y, Z, Ylm, 'facecolor', 'interp', 'linestyle', 'none')
axis tight, axis equal, caxis([-.5 .5])
box on, grid off
xlabel('x'), ylabel('y'), zlabel('z')
title(sprintf('|Y_l^m(x,y,z)|^2   l, m = %d, %d', l, m),...
    'fontweight', 'bold', 'fontsize', 14)
light('Position',[0.1,-1.2,.3]);
saveas(fig, sprintf('Y_%d_%d.eps',l,m),'epsc2')
    """
