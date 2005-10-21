import LinearAlgebra as linalg
import Numeric as num
from math import sqrt
from math import pi
from ASE.ChemicalElements import Element


def dimerfit(b, e, s, p):

    """Dimerfit function
    Solve the linear equation for the variation of the bond lengths and the
    related atomization energy. Results in finding the coefficients for the
    cubic polynomium: b is the bond lengths, e the related energies (which are
    found by the dimer_calc function), s indicates the atomic symbol and p is
    the number of variation of the bond length at each grid space.
    """

    r = int(p)
    D = num.zeros((r, 4), num.Float)

    for k in range(r):
        for i in range(4):
            D[k,i] = b[k]**(-i)

    x = num.dot(num.transpose(D),D)
    y = num.dot(num.transpose(e),D)
    a = linalg.solve_linear_equations(x,y)

    # Vibrational frequency

    k = (sqrt(-3 * (8 * a[1] * a[3] - 3 * a[2]**2 )) - 3 * a[2]) / (2 * a[1])

    e = Element(s)
    m = (e.mass) / 2
    h = 6.6260755e-34
    amu = 1.66054e-27
    eV = 1.602177e-19
   
    hw = sqrt(k * 100 / (m * amu)) * (h / (2 * pi * eV)) * 1e3
    fhw = hw

    # Minimum bond length

    b_min = (sqrt(a[2]**2 - 3 * a[1] * a[3]) - a[2]) / (a[1])
    fb = b_min
 
    # Then the atomization energy at the minimum bond length

    E_a = a[0] + a[1] / (b_min) + a[2] / (b_min**2) + a[3] / (b_min**3)
    fEa = E_a

    return fhw, fb, fEa
