import Numeric as num

def factorial(x):
    """Return x!, where x is a non-negative integer."""
    if x < 2: return 1
    else: return x * factorial(x - 1)

def L_to_lm(L):
    """Convert L index to (l, m) index."""
    l = int(num.sqrt(L))
    m = L - l**2 - l
    return l, m

def lm_to_L(l,m):
    """Convert (l, m) index to L index."""
    return l**2 + l + m

def core_states(symbol):
    """Method returning the number of core states for given element."""
    from gpaw.atom.configurations import configurations
    from gpaw.atom.generator import parameters

    core = parameters[symbol][0]
    
    # Parse core string:
    j = 0
    if core.startswith('['):
        a, core = core.split(']')
        core_symbol = a[1:]
        j = len(configurations[core_symbol][1])

    Njcore = j + len(core) / 2
    return Njcore

def get_kpoint_dimensions(kpts):
    """Returns number of k-points along each axis of input Monkhorst pack.
       The set of k-points must not have been symmetry reduced.
    """
    nkpts = len(kpts)
    if nkpts == 1: return num.ones(3)
    tol = 1e-5
    Nk_c = num.zeros(3)
    for c in range(3):
        # Sort kpoints in ascending order along current axis
        slist = num.argsort(kpts[:, c])
        skpts = num.take(kpts, slist)

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
    dim = num.reshape(gd.n_c, (3, 1, 1, 1))
    dk = 2 * num.pi / gd.domain.cell_c
    dk.shape = (3, 1, 1, 1)
    k = ((num.indices(gd.n_c) + dim / 2) % dim - dim / 2) * dk
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
    I  = num.indices(gd.n_c)
    dr = num.reshape(gd.h_c, (3, 1, 1, 1))
    r0 = num.reshape(gd.h_c * gd.beg_c - .5 * gd.domain.cell_c, (3, 1, 1, 1))
    r0 = num.ones(I.shape) * r0
    xyz = r0 + I * dr
    r2 = num.sum(xyz**2)

    # Remove singularity at origin and replace with small number
    middle = gd.N_c / 2.
    # Check that middle is a gridpoint and that it is on this CPU
    if num.alltrue(middle == num.floor(middle)) and \
           num.alltrue(gd.beg_c <= middle < gd.end_c):
        m = (middle - gd.beg_c).astype(int)
        r2[m[0], m[1], m[2]] = 1e-12

    # Return r^2 matrix
    return xyz, r2

def dagger(matrix, copy=True):
    """Return hermitian conjugate of input matrix.

    If copy is False, the input matrix will be changed (no new allocation
    of memory).
    """
    # First change the axis: (Does not allocate a new array)
    dag = num.swapaxes(matrix, 0, 1)

    if copy: # Allocate space for new array
        return num.conjugate(dag)
    else: # The input array is used for output
        if dag.typecode() == num.Complex:
            num.multiply(dag.imag, -1, dag.imag)
        return dag

def symmetrize(matrix):
    """Symmetrize input matrix."""
    num.add(dagger(matrix), matrix, matrix)
    num.multiply(.5, matrix, matrix)
    return matrix

def erf3D(M):
    """Return matrix with the value of the error function evaluated for
       each element in input matrix 'M'.
    """
    from gpaw.utilities import erf
    return elementwise_apply(M, erf, copy=True)

def elementwise_apply(array, function, copy=True):
    """Apply ``function`` to each element of input ``array``. If copy is False,
       the input matrix will be changed (no new allocation of memory).
    """
    if copy: # Allocate space for new array
        result = array.copy()
    else: # The input array is used for output
        result = array
    
    for n in range(len(array.flat)):
        result.flat[n] = function(array.flat[n])

    return result

def apply_subspace_mask(H_nn, f_n):
    """Uncouple occupied and unoccupied subspaces.

    This method forces the H_nn matrix into a block-diagonal form
    in the occupied and unoccupied states respectively.
    """
    occ = 0
    nbands = len(f_n)
    while occ < nbands and f_n[occ] > 1e-3: occ +=1
    H_nn[occ:, :occ] = H_nn[:occ, occ:] = 0

def energy_cutoff_to_gridspacing(E, E_unit='Hartree', h_unit='Ang'):
    """Convert planewave energy cutoff to a real-space gridspacing.

       The method use the conversion formula::
       
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
    """Convert real-space gridspacing to planewave energy cutoff.

       The method use the conversion formula::
       
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
