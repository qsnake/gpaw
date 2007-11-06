import Numeric as num
from gpaw.utilities import hartree
SMALL_NUMBER = 1e-8

#################################################################################
#                                                                               #
# Implementation of few helper functions begins                                 #
# -Todo: Move these for example to gpaw.utilities                               #
#################################################################################

def factorial(x):
    if x == 0:
        return 1.0
    if x == 1:
        return 1.0
    return 1.0 * x * factorial(x-1)


def wigner3j_0(l1,l2,l):
    """Calculates the wigner3j symbol with all m:s zero.
       TODO: Tabulate this!
    """
    J = l1+l2+l
    if J % 2 == 1:
        return 0
    g = J / 2
    return (-1)**g * num.sqrt( factorial(2*g-2*l1) * factorial(2*g- 2*l2) * factorial(2*g - 2*l) /
           factorial(2*g+1)) * factorial(g) / (factorial(g-l1)*factorial(g-l2)*factorial(g-l))

def calculate_slater_energy_density(rgd, u_j, f_j, l_j, V_s, exp_j=None, exclude=None, method='and'):
    """Calculates the slater energy density in 1D-radial grid.
       This is the Slater-potential multiplied with density.

       exp_j The expectation values of <\psi_i| \hat V_x^{nl} | \psi_i>
    """

    if exp_j == None:
        exp_j = num.zeros(len(f_j), num.Float)
    else:
        exp_j[:] = 0.0

    beta = 0.4 # Grr... this is assumed
    N = len(u_j[0])

    V_s[:] = 0.0

    # The exchange energy density multiplied with r
    Vr_x = num.zeros(N, num.Float)

    for i, (u1, f1, l1) in enumerate(zip(u_j, f_j, l_j)):
         for j, (u2, f2, l2) in enumerate(zip(u_j, f_j, l_j)):
             for L in range(abs(l1-l2),l1+l2+1):
                 # Calculate the exchange density
                 nnr = num.where(abs(u1) < 1e-160, 0, u1) * num.where(abs(u2) < 1e-160, 0, u2)
                 nnr[1:] /= rgd.r_g[1:]**2 * 4 * num.pi
                 nnr[0] = nnr[1]

                 # Solve the poisson equation
                 hartree(L, nnr * rgd.r_g * rgd.dr_g, beta, N, Vr_x)
                 w = wigner3j_0(L, l1, l2)**2
                 integrand = Vr_x * nnr
                 add_V_s = True
                 if exclude is not None:
                     if method == 'and':
                         print "and"
                         if i in exclude and j in exclude:
                             add_V_s = False
                     else:
                         print "or"
                         if i in exclude or j in exclude:
                             add_V_s = False
                 if add_V_s:
                     V_s -= (2*L+1) * (f1 / 2) * (f2 / 2) * w * integrand
                 exp_j[i] += w * (f2 / 2) * (2*L+1) * num.sum(integrand * rgd.r_g * rgd.dr_g) * (4 * num.pi)

    # Divide the extra r out
    V_s[1:] /= rgd.r_g[1:]
    # Fix the zero value
    V_s[0] = V_s[1]

    #print "V_S = ",2*V_s[0:125]
    #print "V_S = reshape(V_S', 126,1)"
    #print "plot(V_S,'r'); hold on;"

    # The exchange energy will be integral over V_s
    return num.dot(V_s, rgd.dr_g * rgd.r_g **2) * 4 * num.pi

def safe_sqr(u_j):
    return num.where(abs(u_j) < 1e-160, 0, u_j)**2

def construct_density1D(gd, u_j, f_j):
    """
    Creates one dimensional density from specified wave functions and occupations.

    =========== ==========================================================
    Parameters:
    =========== ==========================================================
    gd          Radial grid descriptor
    u_j         The wave functions
    f_j         The occupation numbers
    =========== ==========================================================
    """


    n_g = num.dot(f_j, safe_sqr(u_j))
    n_g[1:] /=  4 * num.pi * gd.r_g[1:]**2
    n_g[0] = n_g[1]
    return n_g

def find_fermi_level1D(f_j, e_j):
    """
       Finds the fermilevel from occupations and eigenvalue energies.
       Uses tolerance 1e-5 for occupied orbital.
       =========== ==========================================================
        Parameters:
       =========== ==========================================================
        f_j         The occupations list
        e_j         The eigenvalues list
       =========== ==========================================================
    """

    fermi_level = -1000
    for f,e in zip(f_j, e_j):
        if f > 1e-5:
            if fermi_level < e:
                fermi_level = e
    return fermi_level

def find_nucleus(nuclei, a):
    nucleus = None
    for nuc in nuclei:
        if a == nuc.a:
            nucleus = nuc
    assert(nucleus is not None)
    return nucleus


