import Numeric as num
import LinearAlgebra as linalg
from math import sqrt, pi


def bulkfit(a, c, crys):
    
    """Make a fit to the energy as a function of lattice constant.

    ``a`` is the lattice constant, ``c`` is the cohesive energies and
    ``crys`` is the crystal structure.
    """

    n = len(a)
    D = num.zeros((n, 4), num.Float)
   
    for k in range(n):
        for i in range(4):
            D[k,i] = a[k]**(-i)
    
    x = num.dot(num.transpose(D), D)
    y = num.dot(num.transpose(c), D)
    z = linalg.solve_linear_equations(x, y)

    # Minimum lattice constant:
    z_min = (-sqrt(z[2]**2 - 3*z[1]*z[3]) - z[2]) / z[1]
    
    # Then the cohesive energy at the minimum lattice constant:
    E_c = z[0] + z[1] / z_min + z[2] / z_min**2 + z[3] / z_min**3

    eV = 1.602177e-19
    if crys == 'diamond':
        B = 16 * eV / 1e-30 / (9 * z_min**2) * 1e-9
        
    elif crys == 'fcc':
        B = 8 * eV / 1e-30 / (9 * z_min**2) * 1e-9
        
    else:
        assert crys == 'bcc'
        B = 4 * eV / 1e-30 / (9 * z_min**2) * 1e-9

    return z_min, E_c, B

