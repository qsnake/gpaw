import Numeric as num
import LinearAlgebra as linalg
from math import sqrt, pi


def bulkfit(a, c ,n, crys):
    
    """Bulkfit function
    Solve the linear equation for the variation of the lattice constants and
    the related cohesive energy. It results in finding the coefficient for the
    cubic polynomium: a is the lattice constants, e is the cohesive energies, n
    is the number of variation of the lattice constants at each gridpoint set
    and crys is the type of crystal structure.
    """

    D = num.zeros((n, 4), num.Float)
   
    for k in range(n):
        for i in range(4):
            D[k,i] = a[k]**(-i)

    f_test = open('test_bulk.dat', 'w')

    print >> f_test, a, c
    
    x = num.dot(num.transpose(D),D)
    y = num.dot(num.transpose(c),D)
    z = linalg.solve_linear_equations(x,y)

    # Minimum lattice constant
           
    z_min = (-sqrt(z[2]**2 - 3*z[1]*z[3]) - z[2]) / (z[1])
    fz = z_min

    # Then the cohesive energy at the minimum lattice constant

    E_c = z[0] + z[1] / (z_min) + z[2] / (z_min**2) + z[3] / (z_min**3)
    fEc = E_c

    if crys == 'diamond':
        B = 16 / (9 * z_min**2)

    elif crys == 'fcc':
        B = 8 / (9 * z_min**2)
        
    elif crys == 'bcc':
        B = 4 / (9 * z_min**2)
        
    return fz, fEc, B

