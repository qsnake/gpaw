import numpy as np
from ase.units import Bohr

from gpaw.symmetry import Symmetry


def reduce_kpoints(atoms, bzk_kc, setups, usesymm):
    """Reduce the number of k-points using symmetry.

    Returns symmetry object, weights and k-points in the irreducible
    part of the BZ.

    usesymm can be None, False, or True, to use either
    no symmetries, inversion symmetry only (if present), or all symmetries
    """

    #if np.logical_and(np.logical_not(atoms.pbc), bzk_kc.any(axis=0)).any():
    if (~atoms.pbc & bzk_kc.any(0)).any():
        raise ValueError('K-points can only be used with PBCs!')

    if usesymm is None:
        nkpts = len(bzk_kc)
        return None, np.ones(nkpts) / nkpts, bzk_kc.copy()
    
    # Round off:
    magmom_a = atoms.get_initial_magnetic_moments().round(decimals=3)
    id_a = zip(magmom_a, setups.id_a)

    # Construct a Symmetry instance containing the identity
    # operation only:
    symmetry = Symmetry(id_a, atoms.get_cell() / Bohr, atoms.get_pbc())

    if usesymm:
        # Find symmetry operations of atoms:
        symmetry.analyze(atoms.get_scaled_positions())
    else:
        symmetry.prune_symmetries(atoms.get_scaled_positions())

    # Reduce the set of k-points: (and add inversion if not already detected)
    ibzk_kc, weight_k = symmetry.reduce(bzk_kc)

    if usesymm:
        setups.set_symmetry(symmetry)
    else:
        symmetry = None

    return symmetry, weight_k, ibzk_kc
