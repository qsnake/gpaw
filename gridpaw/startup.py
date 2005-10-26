import Numeric as num
from ASE.ChemicalElements.symbol import symbols
from ASE.Units import Convert

from gridpaw.nucleus import Nucleus
from gridpaw.rotation import rotation
from gridpaw.setup import Setup
from gridpaw.domain import Domain
from gridpaw.symmetry import Symmetry
from gridpaw.paw import Paw
from gridpaw.xc_functional import XCFunctional
from gridpaw.parallel import distribute_kpoints_and_spins


def create_paw_object(out, a0, Ha,
                      pos_ai, Z_a, magmom_a, cell_i, bc_i, angle,
                      h, N_i, xcname,
                      nbands, spinpol, kT,
                      bzk_ki,
                      softgauss, order, usesymm, mix, old, fixdensity,
                      idiotproof, hund, lmax, onohirose, tolerance,
                      # Parallel stuff:
                      parsize_i,
                      restart_file):

    magmom_a = num.array(magmom_a)
    magnetic = num.sometrue(magmom_a)

    # Is this a gamma-point calculation?
    gamma = (len(bzk_ki) == 1 and not num.sometrue(bzk_ki[0]))

    # Default values:
    if spinpol is None:
        spinpol = magnetic
    if hund is True and not spinpol:
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
    
    # Construct a dictionary, mapping atomic numbers to PAW-setup objects:
    setups = construct_setups(Z_a, xcfunc, lmax, nspins, softgauss)

    # Default value for grid spacing:
    if N_i is None:
        if h is None:
            # Find the smalles recommended grid spacing:
            h = 1000.0
            for setup in setups.values():
                h = min(h, a0 * setup.get_recommended_grid_spacing())

        # N_i should be a multiplum of 4:
        N_i = [max(4, int(L / h / 4 + 0.5) * 4) for L in cell_i]
    else:
        if h is not None:
            raise TypeError("""You can't use both "gpts" and "h"!""")
    N_i = num.array(N_i)

    # Create a Domain object:
    domain = Domain(cell_i / a0, bc_i, angle)

    # Brillouin-zone stuff:
    if gamma:
        typecode = num.Float
        symmetry = None
        weights_k = [1.0]
        ibzk_ki = num.zeros((1, 3), num.Float)
    else:
        typecode = num.Complex
        # Reduce the the k-points to those in the irreducible part of
        # the Brillouin zone:
        symmetry, weights_k, ibzk_ki = reduce_kpoints(
            bzk_ki, pos_ai / a0, Z_a, domain, usesymm)

    if usesymm and symmetry is not None:
        # Find rotation matrices for spherical harmonics:
        R_slm1m2 = [[rotation(l, symm) for l in range(3)]
                    for symm in symmetry.symmetries]

        for setup in setups.values():
            setup.calculate_rotations(R_slm1m2)

    # Build list of nuclei:
    nuclei = [Nucleus(setups[Z], a, typecode, onohirose)
              for a, Z in enumerate(Z_a)]

    # Sum up the number of valence electrons:
    nvalence = 0
    for nucleus in nuclei:
        nvalence += nucleus.setup.get_number_of_valence_electrons()

    if nbands is None:
        # Default value for number of bands:
        nbands = (nvalence + 7) // 2 # make room for magnetisation XXX
    elif nvalence > 2 * nbands:
        raise ValueError('Too few bands!')

    # Get the local number of spins and k-points, and return a
    # domain_comm and kpt_comm for this processor:
    myspins, myibzk_ki, myweights_k, domain_comm, kpt_comm = \
             distribute_kpoints_and_spins(nspins, ibzk_ki, weights_k)

    domain.set_decomposition(domain_comm, parsize_i, N_i)

    # We now have all the parameters needed to construct a PAW object:
    paw = Paw(a0, Ha,
              setups, nuclei, domain, N_i, symmetry, xcfunc,
              nvalence, nbands, nspins, kT,
              typecode, bzk_ki, ibzk_ki, weights_k,
              order, usesymm, mix, old, fixdensity, idiotproof,
              # Parallel stuff:
              myspins, myibzk_ki, myweights_k, kpt_comm,
              out)

    if restart_file is None:
        paw.set_positions(pos_ai / a0)
        paw.initialize_density_and_wave_functions(hund, magmom_a)
    else:
        paw.initialize_from_netcdf(restart_file)
        
    paw.set_convergence_criteria(tolerance)
    return paw

    
def reduce_kpoints(bzk_ki, pos_ai, Z_a, domain, usesymm):
    """Reduce the number of k-points using symmetry.

    Returns symmetry object, weights and k-points in the irreducible
    part of the BZ."""

    for i in range(3):
        if not domain.periodic_i[i] and num.sometrue(bzk_ki[:, i]):
            raise ValueError('K-points can only be used with PBCs!')

    # Construct a Symmetry instance containing the identity
    # operation only:
    symmetry = Symmetry(Z_a, domain)

    if usesymm:
        # Find symmetry operations of atoms:
        symmetry.analyze(pos_ai)

    # Reduce the set of k-points:
    ibzk_ki, weights_k = symmetry.reduce(bzk_ki)

    if usesymm:
        symmetry = symmetry
    else:
        symmetry = None

    return symmetry, weights_k, ibzk_ki


def construct_setups(Z_a, xcfunc, lmax, nspins, softgauss):
    # Construct necessary PAW-setup objects and count the number of
    # valence electrons:
    setups = {}
    for Z in Z_a:
        if Z not in setups:
            symbol = symbols[Z]
            setup = Setup(symbol, xcfunc, lmax, nspins, softgauss)
            setups[Z] = setup
            assert Z == setup.Z
    return setups
