import Numeric as num
import LinearAlgebra as linalg
from math import sqrt, pi


def hcpfit(a, cov, coh):
    
    """hcpfit function.
    
    Solve the linear equation for the variation of the lattice constants and
    the cohesive energies. It results in finding the coefficients for the
    double cubic polynomium: ``a`` is the lattice constants, ``cov`` is the
    relation c/a and ``coh`` is the cohesive energies. The gridpoints are
    usually different in the three directions for hcp.
    """
    g = int(sqrt(len(a)))
    D = num.zeros((g, 4), num.Float)
    F = num.zeros((g, 4), num.Float)
    amin = num.zeros(g, num.Float)
    Eamin = num.zeros(g, num.Float)
    
    # Minimum lattice constant a and energy at a fix covera:    
    for t in range(g):

        Acoh = coh[t]
 
        for k in range(g):
            for i in range(4):
                D[k,i] = a[k]**(-i)
    
        x = num.dot(num.transpose(D), D)
        y = num.dot(num.transpose(Acoh), D)
        z = linalg.solve_linear_equations(x, y)

        a_min = (-sqrt(z[2]**2 - 3 * z[1] * z[3]) - z[2]) / (z[1])
        amin[t] = a_min
        E_a = z[0] + z[1] / (a_min) + z[2] / (a_min**2) + z[3] / (a_min**3)
        Eamin[t] = E_a

    # Minimum covera and minimum cohesive energy of the Eamin energies:
    for n in range(g):
        for m in range(4):
            F[n,m] = cov[n]**(-m)

    Fx = num.dot(num.transpose(F), F)
    Fy = num.dot(num.transpose(Eamin), F)
    Fz = linalg.solve_linear_equations(Fx, Fy)
    
    cov_min = (-sqrt(Fz[2]**2 - 3 * Fz[1] * Fz[3]) - Fz[2]) / (Fz[1])
    E_min = (Fz[0] + Fz[1] / (cov_min) + Fz[2] / (cov_min**2) +
             Fz[3] / (cov_min**3))

    # Real minimum of a:
    Dx = num.dot(num.transpose(F), F)
    Dy = num.dot(num.transpose(amin), F)
    Dz = linalg.solve_linear_equations(Dx, Dy)

    min_of_a = (Dz[0] + Dz[1] / (cov_min) + Dz[2] / (cov_min**2) +
                Dz[3] / (cov_min**3))
   
    # Calculate the Bulk modulus:
    eV = 1.602177e-19
    bulk_mod = 8 * sqrt(3) * eV / 1e-30 / (27 * min_of_a**2 * cov_min) * 1e-9

    return min_of_a, cov_min, bulk_mod, E_min
