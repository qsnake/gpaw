import Numeric as num

def rSquared(gd):
    """Constructs and returns a matrix containing the square of the
       distance from the origin which is placed in the center of the box
       described by the given grid-descriptor 'gd'.
    """
    I  = num.indices(gd.n_c)
    dr = num.reshape(gd.h_c, (3, 1, 1, 1))
    r0 = num.reshape(gd.h_c * gd.beg0_c - .5 * gd.domain.cell_c, (3,1,1,1))
    r0 = num.ones(I.shape)*r0
    r2 = num.sum((r0 + I * dr)**2)

    # remove singularity at origin and replace with small number
    middle = gd.N_c / 2.
    # check that middle is a gridpoint and that it is on this CPU
    if num.alltrue(middle == num.floor(middle)) and \
           num.alltrue(gd.beg0_c <= middle < gd.end_c):
        m = (middle - gd.beg0_c).astype(int)
        r2[m[0], m[1], m[2]] = 1e-12

    # return r^2 matrix
    return r2

def erf3D(M):
    """Return matrix with the value of the error function evaluated for
       each element in input matrix 'M'.
    """
    from gridpaw.utilities import erf

    dim = M.shape
    res = num.zeros(dim,num.Float)
    for k in range(dim[0]):
        for l in range(dim[1]):
            for m in range(dim[2]):
                res[k, l, m] = erf(M[k, l, m])
    return res

def construct_gauss(gd, a0=25.):
    """Construct gaussian density and potential"""
    # determine r^2 and r matrices
    r2 = rSquared(gd)
    r  = num.sqrt(r2)

    # 'width' of gaussian distribution
    # ng ~ exp(-a0 / 4) on the boundary of the domain
    a = a0 / min(gd.domain.cell_c)**2

    # gaussian density
    ng = num.exp(-a * r2) * (a / num.pi)**(1.5)

    # gaussian potential
    vg = erf3D(num.sqrt(a) * r) / r

    # gaussian self energy
    #Eg = -num.sqrt(0.5 * a / num.pi)

    return ng, vg#, Eg

                        
