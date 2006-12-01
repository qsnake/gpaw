import Numeric as num

def factorial(x):
    """Return x!, where x is a non-negative integer"""
    if x < 2: return 1
    else: return x * factorial(x - 1)

def L_to_lm(L):
    """convert L index to (l, m) index"""
    l = int(num.sqrt(L))
    m = L - l**2 - l
    return l, m

def lm_to_L(l,m):
    """convert (l, m) index to L index"""
    return l**2 + l + m

def core_states(symbol):
    """method returning the number of core states for given element"""
    from gpaw.atom.configurations import configurations
    from gpaw.atom.generator import parameters

    try:
        core, rcut = parameters[symbol]
        extra = None
    except ValueError:
        core, rcut, extra = parameters[symbol]
    
    # Parse core string:
    j = 0
    if core.startswith('['):
        a, core = core.split(']')
        core_symbol = a[1:]
        j = len(configurations[core_symbol][1])

    Njcore = j + len(core)/2
    return Njcore

def pack(M2, symmetric=True, tol=1e-6):
    """new pack method"""
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] =  M2[r, c] + num.conjugate(M2[c,r])
            p += 1
            if symmetric:
                error = abs(M2[r, c] - num.conjugate(M2[c, r]))
                assert error < tol, 'Not symmetric:\n' +\
                       'error = %s = |%s-%s|'%(error, M2[r,c], M2[c,r])
    assert p == len(M)
    return M

def pack2(M2, symmetric=True, tol=1e-6):
    """new pack method"""
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] =  (M2[r, c] + num.conjugate(M2[c,r])) / 2. # <- divide by 2!!
            p += 1
            if symmetric:
                error = abs(M2[r, c] - num.conjugate(M2[c, r]))
                assert error < tol, 'Not symmetric:\n' +\
                       'error = %s = |%s-%s|'%(error, M2[r,c], M2[c,r])
    assert p == len(M)
    return M

def get_kpoint_dimensions(kpts):
    """Returns number of k-points along each axis of input Monkhorst pack.
       The set of k-points must not have been symmetry reduced.
    """
    nkpts = len(kpts)
    if nkpts == 1: return num.ones(3)
    tol = 1e-5
    Nk_c = num.zeros(3)
    for c in range(3):
        # sort kpoints in ascending order along current axis
        slist = num.argsort(kpts[:, c])
        skpts = num.take(kpts, slist)

        # determine increment between kpoints along current axis
        DeltaK = max([skpts[n + 1, c] - skpts[n, c] for n in range(nkpts - 1)])

        #determine number of kpoints as inverse of distance between kpoints
        if DeltaK > tol: Nk_c[c] = int(round(1. / DeltaK))
        else: Nk_c[c] = 1
    return Nk_c

def construct_reciprocal(gd):
    """Construct the reciprocal lattice vectors correspoding to the
       grid defined in input grid-descriptor 'gd'
    """
    # calculate reciprocal lattice vectors
    dim = num.reshape(gd.n_c, (3, 1, 1, 1))
    dk = 2 * num.pi / gd.domain.cell_c
    dk.shape = (3, 1, 1, 1)
    k = ((num.indices(gd.n_c) + dim / 2)%dim - dim / 2) * dk
    k2 = sum(k**2)
    k2[0,0,0] = 1.0

    # determine N^3
    N3 = gd.n_c[0] * gd.n_c[1] * gd.n_c[2]

    return k2, N3

def coordinates(gd):
    """Constructs and returns matrices containing cartesian coordinates,
       and the square of the distance from the origin.
       The origin is placed in the center of the box described by the given
       grid-descriptor 'gd'.
    """    
    I  = num.indices(gd.n_c)
    dr = num.reshape(gd.h_c, (3, 1, 1, 1))
    r0 = num.reshape(gd.h_c * gd.beg_c - .5 * gd.domain.cell_c, (3,1,1,1))
    r0 = num.ones(I.shape)*r0
    xyz = r0 + I * dr
    r2 = num.sum(xyz**2)

    # remove singularity at origin and replace with small number
    middle = gd.N_c / 2.
    # check that middle is a gridpoint and that it is on this CPU
    if num.alltrue(middle == num.floor(middle)) and \
           num.alltrue(gd.beg_c <= middle < gd.end_c):
        m = (middle - gd.beg_c).astype(int)
        r2[m[0], m[1], m[2]] = 1e-12

    # return r^2 matrix
    return xyz, r2

def erf3D(M):
    """Return matrix with the value of the error function evaluated for
       each element in input matrix 'M'.
    """
    from gpaw.utilities import erf

    dim = M.shape
    res = num.zeros(dim, num.Float)
    for k in range(dim[0]):
        for l in range(dim[1]):
            for m in range(dim[2]):
                res[k, l, m] = erf(M[k, l, m])
    return res

class Translate:
    """Class used to translate wave functions / densities."""
    def __init__(self, sgd, lgd, type=num.Complex):
        self.Ns = sgd.N_c
        self.Nl = lgd.N_c
        self.Nr = float(self.Nl / self.Ns)

        # ensure that the large grid-descriptor is an integer number of times
        # bigger than the small grid-descriptor
        assert num.alltrue(self.Nr == num.around(self.Nr))
        self.tmp = num.zeros(self.Nl, type)

    def translate(self, w, R):
        """Translate input array 'w' defined in the large grid-descriptor 'lgd'
           distance 'R' measured in units of the small grid-descriptor 'sgd'.
        """
        R = num.array(R)
        tmp = self.tmp

        # do nothing, if array is not moved
        if num.alltrue(R == 0): return
        
        # ensure that R is within allowed range and of correct type
        assert num.alltrue(R > 0 and R < self.Nr)

        # determine the size of the blocks to be moved
        B = R * self.Ns
        A = self.Nl - B

        # translate 1. axis
        tmp[:] = w
        w[:A[0]] = tmp[B[0]:]
        w[A[0]:] = tmp[:B[0]]
        
        # translate 2. axis
        tmp[:] = w
        w[:, :A[1]] = tmp[:, B[1]:]
        w[:, A[1]:] = tmp[:, :B[1]]
        
        # translate 3. axis
        tmp[:] = w
        w[:, :, :A[2]] = tmp[:, :, B[2]:]
        w[:, :, A[2]:] = tmp[:, :, :B[2]]

def energy_cutoff_to_gridspacing(E, E_unit='Hartree', h_unit='Ang'):
    """Convert planewave energy cutoff to a real-space gridspacing
       using the conversion formula::
       
                pi
        h =   -----
            \/ 2 E
       
       in atomic units (Hartree and Bohr)
    """
    from ASE.Units import Convert
    E = Convert(E, E_unit, 'Hartree')
    h = num.pi / num.sqrt(2 * E)
    h = Convert(h, 'Bohr', h_unit)
    return h
    
def gridspacing_to_energy_cutoff(h, h_unit='Ang', E_unit='Hartree'):
    """Convert real-space gridspacing to planewave energy cutoff
       using the conversion formula::
       
             1   pi  2
        E  = - ( -- )
         c   2   h
       
       in atomic units (Hartree and Bohr)
    """
    from ASE.Units import Convert
    h = Convert(h, h_unit, 'Bohr')
    E = .5 * (num.pi / h)**2
    E = Convert(E, 'Hartree', E_unit)
    return E
