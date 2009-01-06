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
    dk = 2 * npy.pi / gd.domain.cell_c
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
    r0 = npy.reshape(gd.h_c * gd.beg_c - .5 * gd.domain.cell_c, (3, 1, 1, 1))
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
    return npy.reshape(a_x, shape[1:])

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

def symmetrize(matrix):
    """Symmetrize input matrix."""
    npy.add(dagger(matrix), matrix, matrix)
    npy.multiply(.5, matrix, matrix)
    return matrix

def fill(H_nn, side='upper'):
    """Fill in values of hermitian matrix.

    Fill values in lower or upper triangle of H_nn based on the opposite
    triangle, such that the resulting matrix is symmetric/hermitian.

    side='lower' will copy (conjugated) values from upper triangle into the
    lower triangle.
    """
    N, tmp = H_nn.shape
    assert N == tmp, 'Matrix must be square'
    #assert npy.isreal(H_nn.diagonal()).all(), 'Diagonal should be real'
    if side != 'upper':
        H_nn = H_nn.T

    for n in range(N - 1):
        lower = H_nn[n + 1:, n] # Lower coulumn
        upper = H_nn[n, n + 1:] # Upper row
        upper[:] = lower.conj()
        
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
    from ase import Hartree, Bohr
    return (.5 * npy.pi * Bohr / h)**2 * Hartree

def get_HS_matrices(atoms, nt_sg, D_asp, psit_unG):
    """Determine matrix elements of the Hamiltonian defined by the
    input density and density matrices, with respect to input wave functions.

    Extra parameters like k-points and xc functional can be set on the
    calculator attached to the input ListOfAtoms.
    
    ============== =========================================================
    Input argument Description
    ============== =========================================================
    ``atoms``      A ListOfAtoms object with a suitable gpaw calculator
                   attached.
                   
    ``nt_sg``      The pseudo electron spin-density on the fine grid.

    ``D_asp``      The atomic spin-density matrices in packed format.

    ``psit_unG``   The periodic part of the wave function at combined spin-
                   kpoint index ``u`` and band ``n``. Must be given on the
                   coarse grid.
    ============== =========================================================

    Output tuple (H_unn, S_unn), where H_unn[u] is the Hamiltonian matrix at
    the combined spin-kpoint index u. S_unn is the overlap matrix.
    """
    nspins = len(nt_sg)
    nbands = len(psit_unG[0])
    gpts = npy.array(psit_unG[0][0].shape)
    paw = atoms.GetCalculator()
    
    # Ensure that a paw object is initialized
    paw.set(nbands=nbands, out=None)
    paw.initialize()

    # Sanity checks
    assert len(atoms) == len(D_asp)
    assert len(psit_unG) == nspins * len(calc.GetIBZKPoints())
    assert npy.alltrue(gpts * 2 == nt_sg[0].shape)
    assert (nspins - 1) == calc.GetSpinPolarized()

    # Set density on paw-object
    paw.density.nt_sg[:] = nt_sg
    for D_sp, nucleus in zip(D_asp, paw.nuclei):
        nucleus.D_sp = D_sp

    # Update Hamiltonian
    paw.set_positions() # make the localized functions
    for kpt, psit_nG in zip(paw.kpt_u, psit_unG):
        kpt.psit_nG = psit_nG
        for nucleus in paw.pt_nuclei:
            nucleus.calculate_projections(kpt) # calc P = <p|psi>
    paw.hamiltonian.update(paw.density) # solve poisson

    # Determine Hamiltonian and overlap
    Htpsit_nG = npy.zeros( (nbands,) + tuple(gpts), paw.dtype) # temp array
    H_unn = npy.zeros((len(psit_unG), nbands, nbands), paw.dtype)
    S_unn = npy.zeros((len(psit_unG), nbands, nbands), paw.dtype)
    for H_nn, S_nn, kpt in zip(H_unn, S_unn, paw.kpt_u):
        Htpsit_nG[:] = 0.0
        psit_nG = kpt.psit_nG
        u, s = kpt.u, kpt.s
        comm, root = kpt.comm, kpt.root
        phase_cd = kpt.phase_cd

        # Fill in the lower triangle of the Hamiltonian matrix:
        paw.hamiltonian.kin.apply(psit_nG, Htpsit_nG, phase_cd)           
        Htpsit_nG += psit_nG * paw.hamiltonian.vt_sG[s]
        r2k(0.5 * paw.gd.dv, psit_nG, Htpsit_nG, 1.0, H_nn)
        for nucleus in paw.hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[u]
            H_nn += npy.dot(P_ni, npy.dot(unpack(nucleus.H_sp[s]),
                                          cc(npy.transpose(P_ni))))
        comm.sum(H_nn, root)

        # Fill in the lower triangle of the overlap matrix:
        rk(paw.gd.dv, psit_nG, 0.0, S_nn)
        for nucleus in paw.my_nuclei:
            P_ni = nucleus.P_uni[u]
            S_nn += npy.dot(P_ni, cc(inner(nucleus.setup.O_ii, P_ni)))
        comm.sum(S_nn, root)

    return H_unn, S_unn
