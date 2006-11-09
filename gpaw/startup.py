import Numeric as num
from ASE.ChemicalElements.symbol import symbols
from ASE.Units import Convert

from gpaw.nucleus import Nucleus
from gpaw.rotation import rotation
from gpaw.domain import Domain
from gpaw.symmetry import Symmetry
from gpaw.paw import Paw
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import gcd
import gpaw.mpi as mpi
from gpaw.utilities.timing import Timer

def create_paw_object(out, a0, Ha,
                      pos_ac, Z_a, magmom_a, cell_c, bc_c, angle,
                      h, N_c, xcname,
                      nbands, spinpol, kT,
                      charge,
                      bzk_kc,
                      softgauss, stencils, usesymm, mix, old, fixdensity,
                      hund, lmax, tolerance, maxiter,
                      convergeall, eigensolver, relax, ae,
                      parsize_c,
                      restart_file):

    timer = Timer()
    timer.start('Init')
    if angle is not None:
        usesymm = False
        if lmax not in [None, 0]:
            raise NotImplementedError
    
    magmom_a = num.array(magmom_a)
    magnetic = num.sometrue(magmom_a)

    # Is this a gamma-point calculation?
    gamma = (len(bzk_kc) == 1 and not num.sometrue(bzk_kc[0]))

    # Default values:
    if spinpol is None:
        spinpol = magnetic
    if hund is True and (not spinpol or len(Z_a) != 1):
        hund = False
    if kT is None:
        if gamma:
            kT = 0
        else:
            kT = Convert(0.1, 'eV', 'Hartree') * Ha

    if magnetic and not spinpol:
        raise ValueError('Non-zero initial magnetic moment for a ' +
                         'spin-paired calculation!')

    xcfunc = XCFunctional(xcname)

    if spinpol:
        nspins = 2
    else:
        nspins = 1

    if ae:
        from gpaw.ae import AllElectronSetup as Setup
        softgauss = True
        lmax = 0
    else:
        from gpaw.setup import Setup
        
    # Construct necessary PAW-setup objects and count the number of
    # valence electrons:
    setups = {}    # mapping from atomic numbers to PAW-setup objects
    for Z in Z_a:
        if Z not in setups:
            symbol = symbols[Z]
            setup = Setup(symbol, xcfunc, lmax, nspins, softgauss)
            setup.print_info(out)
            setups[Z] = setup
            assert Z == setup.Z

    # Default value for grid spacing:
    if N_c is None:
        if h is None:
            print >> out, 'Using default value for grid spacing.'
            h = Convert(0.2, 'Ang', 'Bohr') * a0
        # N_c should be a multiplum of 4:
        N_c = [max(4, int(L / h / 4 + 0.5) * 4) for L in cell_c]
    N_c = num.array(N_c)

    # Create a Domain object:
    domain = Domain(cell_c / a0, bc_c, angle)

    # Brillouin-zone stuff:
    if gamma:
        typecode = num.Float
        symmetry = None
        weights_k = [1.0]
        ibzk_kc = num.zeros((1, 3), num.Float)
        nkpts = 1
        print >> out, 'Gamma-point calculation'
    else:
        typecode = num.Complex
        # Reduce the the k-points to those in the irreducible part of
        # the Brillouin zone:
        symmetry, weights_k, ibzk_kc = reduce_kpoints(
            bzk_kc, pos_ac / a0, Z_a, magmom_a, domain, usesymm)

        if symmetry is not None:
            symmetry.print_symmetries(out)

        nkpts = len(ibzk_kc)
        print >> out
        print >> out, (('%d k-point%s in the irreducible part of the ' +
                       'Brillouin zone (total: %d)') %
                       (nkpts, ' s'[1:nkpts], len(bzk_kc)))
        print >> out

    if usesymm and symmetry is not None:
        # Find rotation matrices for spherical harmonics:
        R_slmm = [[rotation(l, symm) for l in range(3)]
                    for symm in symmetry.symmetries]

        for setup in setups.values():
            setup.calculate_rotations(R_slmm)

    # Build list of nuclei:
    nuclei = [Nucleus(setups[Z], a, typecode) for a, Z in enumerate(Z_a)]

    # Sum up the number of valence electrons:
    nvalence = 0
    nao = 0
    for nucleus in nuclei:
        nvalence += nucleus.setup.Nv
        nao += nucleus.setup.niAO
    nvalence -= charge
    if nvalence < 0:
        raise ValueError(
            'Charge %f is not possible - not enough valence electrons' %
            charge)

    if nbands is None:
        # Default value for number of bands:
        #nbands = (nvalence + 7) // 2 + int(num.sum(magmom_a) / 2)
        nbands = nao
    elif nbands <= 0:
        nbands = (nvalence + 1) // 2 + (-nbands)
        
    if nvalence > 2 * nbands:
        raise ValueError('Too few bands!')

    # Get the local number of spins and k-points, and return a
    # domain_comm and kpt_comm for this processor:
    domain_comm, kpt_comm = distribute_kpoints_and_spins(nspins, nkpts)

    domain.set_decomposition(domain_comm, parsize_c, N_c)

    timer.stop()
    # We now have all the parameters needed to construct a PAW object:
    paw = Paw(a0, Ha,
              setups, nuclei, domain, N_c, symmetry, xcfunc,
              nvalence, charge, nbands, nspins,
              typecode, bzk_kc, ibzk_kc, weights_k,
              stencils, usesymm, mix, old, fixdensity, maxiter,
              convergeall, eigensolver, relax, pos_ac / a0, timer, kT / Ha,
              tolerance, kpt_comm, restart_file, hund, magmom_a,
              out)

    return paw

    
def reduce_kpoints(bzk_kc, pos_ac, Z_a, magmom_a, domain, usesymm):
    """Reduce the number of k-points using symmetry.

    Returns symmetry object, weights and k-points in the irreducible
    part of the BZ."""

    for c in range(3):
        if not domain.periodic_c[c] and num.sometrue(bzk_kc[:, c]):
            raise ValueError('K-points can only be used with PBCs!')

    # Construct a Symmetry instance containing the identity
    # operation only:
    symmetry = Symmetry(Z_a, magmom_a, domain)

    if usesymm:
        # Find symmetry operations of atoms:
        symmetry.analyze(pos_ac)

    # Reduce the set of k-points:
    ibzk_kc, weights_k = symmetry.reduce(bzk_kc)

    if usesymm:
        symmetry = symmetry
    else:
        symmetry = None

    return symmetry, weights_k, ibzk_kc

def new_communicator(ranks):
    if len(ranks) == 1:
        return mpi.serial_comm
    elif len(ranks) == mpi.size:
        return mpi.world
    else:
        return mpi.world.new_communicator(num.array(ranks))


def distribute_kpoints_and_spins(nspins, nkpts):
    ntot = nspins * nkpts
    size = mpi.size
    rank = mpi.rank

    ndomains = size // gcd(ntot, size)

    r0 = (rank // ndomains) * ndomains
    ranks = range(r0, r0 + ndomains)
    domain_comm = new_communicator(ranks)

    r0 = rank % ndomains
    ranks = range(r0, r0 + size, ndomains)
    kpt_comm = new_communicator(ranks)

    return domain_comm, kpt_comm
