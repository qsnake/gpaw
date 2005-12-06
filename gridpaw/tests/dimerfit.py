import LinearAlgebra as linalg
import Numeric as num
from math import sqrt, pi
from ASE.ChemicalElements import Element


def dimerfit(b, e, symbol):

    """Dimerfit function
    Solve the linear equation for the variation of the bond lengths and the
    related atomization energy. Results in finding the coefficients for the
    cubic polynomium: b is the bond lengths, e is the related energies (which
    are found by the dimer_calc function), s indicates the atomic symbol and p
    is the number of variation of the bond length at each grid space.
    """

    p = len(b)
    D = num.zeros((p, 4), num.Float)

    for k in range(p):
        for i in range(4):
            D[k,i] = b[k]**(-i)

    x = num.dot(num.transpose(D), D)
    y = num.dot(num.transpose(e), D)
    a = linalg.solve_linear_equations(x, y)

    # Vibrational frequency:
    k = (sqrt(-3 * (8 * a[1] * a[3] - 3 * a[2]**2 )) - 3 * a[2]) / (2 * a[1])
   
    e = Element(symbol)
    m = (e.mass) / 2
    h = 6.6260755e-34
    amu = 1.66054e-27
    eV = 1.602177e-19
   
    hw = sqrt(k * eV / 1e-20 / (m * amu)) * (h / (2 * pi * eV)) *1e3
    
    # Minimum bond length:
    b_min = (sqrt(a[2]**2 - 3 * a[1] * a[3]) - a[2]) / (a[1])
     
    # Atomization energy at the minimum bond length:
    E_a = a[0] + a[1] / (b_min) + a[2] / (b_min**2) + a[3] / (b_min**3)
    
    return hw, b_min, E_a
