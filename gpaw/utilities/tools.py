import numpy as npy

def function_timer(func, *args, **kwargs):
    if 'timeout' in kwargs:
        out = kwargs.pop('timeout')
    else:
        from sys import stdout as out

    from time import time
    t1 = time()
    r = func(*args, **kwargs)
    t2 = time()
    print >>out, t2 - t1
    return r

def L_to_lm(L):
    """Convert L index to (l, m) index."""
    l = int(npy.sqrt(L))
    m = L - l**2 - l
    return l, m

def lm_to_L(l, m):
    """Convert (l, m) index to L index."""
    return l**2 + l + m

def core_states(symbol):
    """Method returning the number of core states for given element."""
    from gpaw.atom.configurations import configurations
    from gpaw.atom.generator import parameters

    core = parameters[symbol].get('core', '')

    # Parse core string:
    j = 0
    if core.startswith('['):
        a, core = core.split(']')
        core_symbol = a[1:]
        j = len(configurations[core_symbol][1])

    Njcore = j + len(core) / 2
    return Njcore

def split_formula(formula):
    """Count elements in a chemical formula.

    E.g. split_formula('C2H3Mg') -> ['C', 'C', 'H', 'H', 'H', 'Mg']
    """
    res = []
    for c in formula:
        if c.isupper():
            res.append(c)
        elif c.islower():
            res[-1] += c
        else:
            res.extend([res[-1],] * (eval(c) - 1))
    return res

def get_kpoint_dimensions(kpts):
    """Returns number of k-points along each axis of input Monkhorst pack.
       The set of k-points must not have been symmetry reduced.
    """
    nkpts = len(kpts)
    if nkpts == 1:
        return npy.ones(3, int)

    tol = 1e-5
    Nk_c = npy.zeros(3)
    for c in range(3):
        # Sort kpoints in ascending order along current axis
        slist = npy.argsort(kpts[:, c])
        skpts = npy.take(kpts, slist)

        # Determine increment between kpoints along current axis
        DeltaK = max([skpts[n + 1, c] - skpts[n, c] for n in range(nkpts - 1)])

        # Determine number of kpoints as inverse of distance between kpoints
        if DeltaK > tol: Nk_c[c] = int(round(1. / DeltaK))
        else: Nk_c[c] = 1
    return Nk_c

def construct_reciprocal(gd):
    """Construct the reciprocal lattice vectors correspoding to the
       grid defined in input grid-descriptor 'gd'.
    """
    # Calculate reciprocal lattice vectors
    dim = npy.reshape(gd.n_c, (3, 1, 1, 1))
    dk = 2 * npy.pi / gd.cell_c
    dk.shape = (3, 1, 1, 1)
    k = ((npy.indices(gd.n_c) + dim / 2) % dim - dim / 2) * dk
    k2 = sum(k**2)
    k2[0, 0, 0] = 1.0

    # Determine N^3
    N3 = gd.n_c[0] * gd.n_c[1] * gd.n_c[2]

    return k2, N3

def coordinates(gd):
    """Constructs and returns matrices containing cartesian coordinates,
       and the square of the distance from the origin.

       The origin is placed in the center of the box described by the given
       grid-descriptor 'gd'.
    """
    I  = npy.indices(gd.n_c)
    dr = npy.reshape(gd.h_c, (3, 1, 1, 1))
    r0 = npy.reshape(gd.h_c * gd.beg_c - .5 * gd.cell_c, (3, 1, 1, 1))
    r0 = npy.ones(I.shape) * r0
    xyz = r0 + I * dr
    r2 = npy.sum(xyz**2, axis=0)

    # Remove singularity at origin and replace with small number
    middle = gd.N_c / 2.
    # Check that middle is a gridpoint and that it is on this CPU
    if (npy.alltrue(middle == npy.floor(middle)) and
        npy.alltrue(gd.beg_c <= middle) and
        npy.alltrue(middle < gd.end_c)):
        m = (middle - gd.beg_c).astype(int)
        r2[m[0], m[1], m[2]] = 1e-12

    # Return r^2 matrix
    return xyz, r2

def pick(a_ix, i):
    """Take integer index of a, or a linear combination of the elements of a"""
    if isinstance(i, int):
        return a_ix[i]
    shape = a_ix.shape
    a_x = npy.dot(i, a_ix[:].reshape(shape[0], -1))
    return a_x.reshape(shape[1:])

def dagger(a):
    """Return Hermitian conjugate of input"""
    return npy.conj(a.T)

def dagger_self(a):
    """Hermitian conjugate input"""
    a = a.T
    if a.dtype == complex:
        a.imag *= -1

def project(a, b):
    """Return the projection of b onto a."""
    return a * (npy.dot(a.conj(), b) / npy.linalg.norm(a))

def normalize(U):
    """Normalize columns of U."""
    for col in U.T:
        col /= npy.linalg.norm(col)

def gram_schmidt(U):
    """Orthonormalize columns of U according to the Gram-Schmidt procedure."""
    for i, col in enumerate(U.T):
        for col2 in U.T[:i]:
            col -= col2 * npy.dot(col2.conj(), col)
        col /= npy.linalg.norm(col)

def lowdin(U, S=None):
    """Orthonormalize columns of U according to the Lowdin procedure.

    If the overlap matrix is know, it can be specified in S.
    """
    if S is None:
        S = npy.dot(dagger(U), U)
    eig, rot = npy.linalg.eigh(S)
    rot = npy.dot(rot / npy.sqrt(eig), dagger(rot))
    U[:] = npy.dot(U, rot)

def lowdin_svd(U):
    """Orthogonalize according to the Lowdin procedure
       using singular value decomposition.

       U is an N x M matrix containing M vectors as its columns.
    """
    Z, D, V = npy.linalg.svd(U, full_matrices=0)
    return npy.dot(Z, V)

def rotate_matrix(h, U):
    """U contains the new basis as its columns"""
    return npy.dot(dagger(U), npy.dot(h, U))

def permute_basis(h, basis):
    """Permute basis of h according to list basis, specifying new order"""
    assert h.ndim == 2 and h.shape[0] == h.shape[1]
    return h.take(basis, 0).take(basis, 1)

def symmetrize(matrix):
    """Symmetrize input matrix."""
    npy.add(dagger(matrix), matrix, matrix)
    npy.multiply(.5, matrix, matrix)
    return matrix

def tri2full(H_nn, UL='L'):
    """Fill in values of hermitian matrix.

    Fill values in lower or upper triangle of H_nn based on the opposite
    triangle, such that the resulting matrix is symmetric/hermitian.

    UL='U' will copy (conjugated) values from upper triangle into the
    lower triangle.

    UL='L' will copy (conjugated) values from lower triangle into the
    upper triangle.
    """
    N, tmp = H_nn.shape
    assert N == tmp, 'Matrix must be square'
    #assert npy.isreal(H_nn.diagonal()).all(), 'Diagonal should be real'
    if UL != 'L':
        H_nn = H_nn.T

    for n in range(N - 1):
        H_nn[n, n + 1:] = H_nn[n + 1:, n].conj()

def apply_subspace_mask(H_nn, f_n):
    """Uncouple occupied and unoccupied subspaces.

    This method forces the H_nn matrix into a block-diagonal form
    in the occupied and unoccupied states respectively.
    """
    occ = 0
    nbands = len(f_n)
    while occ < nbands and f_n[occ] > 1e-3: occ += 1
    H_nn[occ:, :occ] = H_nn[:occ, occ:] = 0

def cutoff2gridspacing(E):
    """Convert planewave energy cutoff to a real-space gridspacing."""
    from ase import Hartree, Bohr
    return .5 * npy.pi / npy.sqrt(E / Hartree) * Bohr

def gridspacing2cutoff(h):
    """Convert real-space gridspacing to planewave energy cutoff."""
    # In Hartree units, E = k^2 / 2, where k_max is approx. given by pi / h
    # See PRB, Vol 54, 14362 (1996)
    from ase import Hartree, Bohr
    return (.5 * npy.pi * Bohr / h)**2 * Hartree

def geth(cell, h=.2, nodes=None):
    """Convert suggested gridspacing to the actual gridspacing used by gpaw.

    Given a number of nodes, the domian decomposition will also be printed.
    This does not take into account spin-kpoint parallelization, which will
    usually be done first.
    """
    L_c = [npy.linalg.norm(axis) for axis in cell]
    N_c = [max(4, int(.25 * L / h + .5) * 4) for L in L_c]
    print 'Grid points:', N_c
    print 'Grid spacing:', npy.divide(L_c, N_c)
    if nodes is not None:
        from gpaw.domain import decompose_domain
        print 'Domain decomposition:', decompose_domain(N_c, nodes)

def tridiag(a, b, c, r, u):
    """Solve linear system with tridiagonal coefficient matrix.

    a is the lower band, b is the diagonal, c is the upper band, and
    r is the right hand side.
    The solution is returned in u.


    [b1 c1  0  ...            ] [u1]   [r1]
    [a1 b2 c2 0 ...           ] [ :]   [ :]
    [ 0 a2 b3 c3 0 ...        ] [  ] = [  ]
    [                         ] [  ]   [  ]
    [     ... 0 an-2 bn-1 cn-1] [ :]   [ :]
    [          ... 0 an-1 bn  ] [un]   [rn]
    """
    n = len(b)
    tmp = np.zeros(n-1) # necessary temporary array
    if b[0] == 0:
        raise RuntimeError, 'System is effectively order N-1'

    beta = b[0]
    u[0] = r[0] / beta
    for i in range(1, n):
        # Decompose and forward substitution
        tmp[i-1] = c[i-1] / beta
        beta = b[i] - a[i-1] * tmp[i-1]
        if beta == 0:
            raise RuntimeError, 'Method failure'
        u[i] = (r[i] - a[i-1] * u[i-1]) / beta

    for i in range(n-1, 0, -1):
        # Backward substitution
        u[i-1] -= tmp[i-1] * u[i]

# from http://hg.stuvel.eu/flickrapi/file/c96a2d1288ef/flickrapi/__init__.py
def md5_hash():
    """
    Use hashlib when available - replace md5, deprecated in Python 2.5.
    """
    try:
        # To replace md5, deprecated in Python 2.5
        from hashlib import md5
    except ImportError:
        # Fallback to md5 to ensure 2.4 compatibility
        import md5
    return md5

md5 = md5_hash()

class Spline:
    def __init__(self, xi, yi, leftderiv=None, rightderiv=None):
        """Cubic spline approximation class.

        xi, yi specifies the known data points.

        leftderiv and rightderiv specifies the first derivative on the
        boundaries. If set to None, the second derivative is set to zero.

        Example usage::

          >>> xi = arange(.1, 5, .5)    # known data points
          >>> yi = cos(xi)              # known data points
          >>> sp = Spline(xi, yi)       # make spline
          >>> x = arange(-.5, 5.5, .05) # points to interpolate to
          >>> y = sp(x)  # get spline value on an entire list
          >>> y2 = sp(4) # get spline value at a single point

        Based on 'Numerical recipes in c'
        """
        self.xy = (xi, yi)
        N = len(xi)
        self.ypp = u = np.zeros(N) # The second derivatives y''
        tmp = np.zeros(N - 1)

        # Set left boundary condition
        if leftderiv is None: # natural spline - second derivative is zero
            tmp[0] = u[0] = 0.0
        else: # clamped spline - first derivative is fixed
            tmp[0] = 3 / (xi[1] - xi[0]) * (
                (yi[1] - yi[0]) / (xi[1] - xi[0]) - leftderiv)
            u[0] = -.5

        for i in range(1, N - 1):
            sig = (xi[i] - xi[i - 1]) / (xi[i + 1] - xi[i - 1])
            p = sig * u[i - 1] + 2
            u[i] = (sig - 1) / p
            tmp[i] = (yi[i + 1] - yi[i]) / (xi[i + 1] - xi[i]) - \
                     (yi[i] - yi[i - 1]) / (xi[i] - xi[i - 1])
            tmp[i] = (6 * tmp[i] / (xi[i +1] - xi[i-1]) - sig * tmp[i - 1]) / p

        # Set right boundary condition
        if rightderiv is None: # natural spline - second derivative is zero
            qn = tmpn = 0.0
        else: # clamped spline - first derivative is fixed
            qn = .5
            tmpn = 3 / (xi[N - 1] - xi[N - 2]) * (
                rightderiv - (yi[N - 1] - yi[N - 2]) / (xi[N - 1] - xi[N - 2]))

        u[N - 1] = (tmpn - qn * tmp[N - 2]) / (qn * u[N - 1] + 1)
        for k in range(N - 2, -1, -1): # backsubstitution step
            u[k] = u[k] * u[k + 1] + tmp[k]

    def __call__(self, x):
        """Evaluate spline for each point in input argument.

        The value in point x[i-1] < x <= x[i] is determined by::

                                    ''       ''
          y(x) = a y    + b y  + c y    + d y
                    i-1      i      i-1      i

        """
        x = np.array(x, float)
        if x.ndim == 0: x.shape = (1,)
        y = np.zeros_like(x)
        xi, yi = self.xy

        i = None
        for j, xval in enumerate(x):
            i = self.locate(xval, i)
            h = xi[i] - xi[i - 1]
            a = (xi[i] - xval) / h
            b = 1. - a
            c = (a**3 - a) * h**2 / 6.
            d = (b**3 - b) * h**2 / 6.
            y[j] = (a * yi[i - 1] + b * yi[i] +
                    c * self.ypp[i - 1] + d * self.ypp[i])
        return y

    def locate(self, x, guess=None):
        """return i such that x[i-1] < x <= xi[i]

        1 or len(xi) - 1 is returned if x is outside list range.
        """
        xi = self.xy[0]
        if x <= xi[0]: return 1
        elif x > xi[-1]: return len(xi) - 1
        elif guess and xi[guess - 1] < x <= xi[guess]: return guess
        else: return np.searchsorted(xi, x)
