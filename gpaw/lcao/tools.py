from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
from ase.units import Hartree
import cPickle as pickle
import numpy as np
from gpaw.mpi import world, rank, MASTER
from gpaw.basis_data import Basis
from gpaw.setup import types2atomtypes
from ase.calculators.singlepoint import SinglePointCalculator


def get_bf_centers(atoms, basis=None):
    calc = atoms.get_calculator()
    if calc is None or isinstance(calc, SinglePointCalculator):
        symbols = atoms.get_chemical_symbols()
        basis_a = types2atomtypes(symbols, basis, 'dzp')
        nao_a = [Basis(symbol, type).nao
                 for symbol, type in zip(symbols, basis_a)]
    else:
        if not calc.initialized:
            calc.initialize(atoms)
        nao_a = [calc.wfs.setups[a].niAO for a in range(len(atoms))]
    pos_ic = []
    for pos, nao in zip(atoms.get_positions(), nao_a):
        pos_ic.extend(pos[None].repeat(nao, 0))
    return np.array(pos_ic)


def get_bfi(calc, a_list):
    """basis function indices from a list of atom indices.
       a_list: atom indices
       Use: get_bfi(calc, [0, 4]) gives the functions indices
       corresponding to atom 0 and 4"""
    bfs_list = []
    for a in a_list:
        M = calc.wfs.basis_functions.M_a[a]
        bfs_list += range(M, M + calc.wfs.setups[a].niAO)
    return bfs_list


def get_bfi2(symbols, basis, a_list):
    """Same as get_bfi, but does not require an LCAO calc"""
    basis = types2atomtypes(symbols, basis, default='dzp')
    bfs_list = []
    i = 0
    for a, symbol in enumerate(symbols):
        nao = Basis(symbol, basis[a]).nao
        if a in a_list:
            bfs_list += range(i, i + nao)
        i += nao
    return bfs_list
    

def get_mulliken(calc, a_list):
    """Mulliken charges from a list of atom indices (a_list)."""
    Q_a = {}
    for a in a_list:
        Q_a[a] = 0.0
    for kpt in calc.wfs.kpt_u:
        S_MM = calc.wfs.S_qMM[kpt.q]
        nao = S_MM.shape[0]
        rho_MM = np.empty((nao, nao), calc.wfs.dtype)
        calc.wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, rho_MM)
        Q_M = np.dot(rho_MM, S_MM).diagonal()
        for a in a_list:
            M1 = calc.wfs.basis_functions.M_a[a]
            M2 = M1 + calc.wfs.setups[a].niAO
            Q_a[a] += np.sum(Q_M[M1:M2])
    return Q_a        


def get_realspace_hs(h_skmm, s_kmm, bzk_kc, weight_k,
                     R_c=(0, 0, 0), direction='x', 
                     usesymm=None):

    from gpaw.symmetry import Symmetry
    from ase.dft.kpoints import get_monkhorst_shape, monkhorst_pack
    
    if usesymm is True:
        raise NotImplementedError, 'Only None and False have been implemented'

    nspins, nk, nbf = h_skmm.shape[:3]
    dir = 'xyz'.index(direction)
    transverse_dirs = np.delete([0, 1, 2], [dir])
    dtype = float
    if len(bzk_kc) > 1 or np.any(bzk_kc[0] != [0, 0, 0]):
        dtype = complex

    kpts_grid = get_monkhorst_shape(bzk_kc)

    # kpts in the transport direction
    nkpts_p = kpts_grid[dir]
    bzk_p_kc = monkhorst_pack((nkpts_p,1,1))[:, 0]
    weight_p_k = 1. / nkpts_p

   # kpts in the transverse directions
    bzk_t_kc = monkhorst_pack(tuple(kpts_grid[transverse_dirs]) + (1, ))
    if usesymm == False:
        #XXX a somewhat ugly hack:
        # By default GPAW reduces inversion sym in the z direction. The steps 
        # below assure reduction in the transverse dirs.
        # For now this part seems to do the job, but it may be written
        # in a smarter way in the future.
        symmetry = Symmetry([1], np.eye(3))
        symmetry.prune_symmetries([[0, 0, 0]])
        ibzk_kc, ibzweight_k = symmetry.reduce(bzk_kc)[:2]
        ibzk_t_kc, weights_t_k = symmetry.reduce(bzk_t_kc)[:2]
        ibzk_t_kc = ibzk_t_kc[:, :2]
        nkpts_t = len(ibzk_t_kc)
    else: # usesymm = None
        ibzk_kc = bzk_kc.copy()
        ibzk_t_kc = bzk_t_kc
        nkpts_t = len(bzk_t_kc)
        weights_t_k = [1. / nkpts_t for k in range(nkpts_t)]

    h_skii = np.zeros((nspins, nkpts_t, nbf, nbf), dtype)
    if s_kmm is not None:
        s_kii = np.zeros((nkpts_t, nbf, nbf), dtype)

    tol = 7
    for j, k_t in enumerate(ibzk_t_kc):
        for k_p in bzk_p_kc:
            k = np.zeros((3,))
            k[dir] = k_p
            k[transverse_dirs] = k_t
            kpoint_list = [list(np.round(k_kc, tol)) for k_kc in ibzk_kc]
            if list(np.round(k, tol)) not in kpoint_list:
                k = -k # inversion
                index = kpoint_list.index(list(np.round(k,tol)))
                h = h_skmm[:, index].conjugate()
                if s_kmm is not None:
                    s = s_kmm[index].conjugate()
                k=-k
            else: # kpoint in the ibz
                index = kpoint_list.index(list(np.round(k, tol)))
                h = h_skmm[:, index]
                if s_kmm is not None:
                    s = s_kmm[index]

            c_k = np.exp(2.j * np.pi * np.dot(k, R_c)) * weight_p_k
            h_skii[:, j] += c_k * h
            if s_kmm is not None:
                s_kii[j] += c_k * s 
    
    if s_kmm is None:
        return ibzk_t_kc, weights_t_k, h_skii
    else:
        return ibzk_t_kc, weights_t_k, h_skii, s_kii


def remove_pbc(atoms, h, s=None, d=0, centers_ic=None, cutoff=None):
    if h.ndim > 2:
        raise KeyError('You have to run remove_pbc for each '
                       'spin/kpoint seperately.')
    L = atoms.cell[d, d]
    nao = len(h)
    dtype = h.dtype
    if centers_ic is None:
        centers_ic = get_bf_centers(atoms) # requires an attached LCAO calc
    ni = len(centers_ic)
    if nao != ni:
        assert nao == 2 * ni
        centers_ic = np.vstack((centers_ic, centers_ic))
        centers_ic[ni:, d] += L
        if cutoff is None:
            cutoff = L - 1e-3
    elif cutoff is None:
        cutoff = 0.5 * L - 1e-3
    pos_i = centers_ic[:, d]
    for i in range(nao):
        dpos_i = abs(pos_i - pos_i[i])
        mask_i = (dpos_i < cutoff).astype(dtype)
        h[i, :] *= mask_i
        h[:, i] *= mask_i
        if s != None:
            s[i, :] *= mask_i
            s[:, i] *= mask_i


def dump_hamiltonian(filename, atoms, direction=None):
    h_skmm, s_kmm = get_hamiltonian(atoms)
    if direction != None:
        d = 'xyz'.index(direction)
        for s in range(atoms.calc.nspins):
            for k in range(atoms.calc.nkpts):
                if s==0:
                    remove_pbc(atoms, h_skmm[s, k], s_kmm[k], d)
                else:
                    remove_pbc(atoms, h_skmm[s, k], None, d)
    
    if atoms.calc.master:
        fd = file(filename,'wb')
        pickle.dump((h_skmm, s_kmm), fd, 2)
        atoms_data = {'cell':atoms.cell, 'positions':atoms.positions,
                      'numbers':atoms.numbers, 'pbc':atoms.pbc}
        
        pickle.dump(atoms_data, fd, 2)
        calc_data ={'weight_k':atoms.calc.weight_k, 
                    'ibzk_kc':atoms.calc.ibzk_kc}
        
        pickle.dump(calc_data, fd, 2)
        fd.close()

    world.barrier()


def dump_hamiltonian_parallel(filename, atoms, direction=None):
    """
        Dump the lcao representation of H and S to file(s) beginning
        with filename. If direction is x, y or z, the periodic boundary
        conditions will be removed in the specified direction. 
        If the Fermi temperature is different from zero,  the
        energy zero-point is taken as the Fermi level.

        Note:
        H and S are parallized over spin and k-points and
        is for now dumped into a number of pickle files. This
        may be changed into a dump to a single file in the future.

    """
    if direction != None:
        d = 'xyz'.index(direction)

    calc = atoms.calc
    wfs = calc.wfs
    nao = wfs.setups.nao
    nq = len(wfs.kpt_u) // wfs.nspins
    H_qMM = np.empty((wfs.nspins, nq, nao, nao), wfs.dtype)
    calc_data = {'k_q':{},
                 'skpt_qc':np.empty((nq, 3)), 
                 'weight_q':np.empty(nq)}

    S_qMM = wfs.S_qMM
   
    for kpt in wfs.kpt_u:
        calc_data['skpt_qc'][kpt.q] = calc.wfs.ibzk_kc[kpt.k]
        calc_data['weight_q'][kpt.q] = calc.wfs.weight_k[kpt.k]
        calc_data['k_q'][kpt.q] = kpt.k
##         print ('Calc. H matrix on proc. %i: '
##                '(rk, rd, q, k) = (%i, %i, %i, %i)') % (
##             wfs.world.rank, wfs.kpt_comm.rank,
##             wfs.gd.domain.comm.rank, kpt.q, kpt.k)
        H_MM = wfs.eigensolver.calculate_hamiltonian_matrix(calc.hamiltonian,
                                                            wfs, 
                                                            kpt)

        H_qMM[kpt.s, kpt.q] = H_MM

        tri2full(H_qMM[kpt.s, kpt.q])
        if kpt.s==0:
            tri2full(S_qMM[kpt.q])
            if direction!=None:
                remove_pbc(atoms, H_qMM[kpt.s, kpt.q], S_qMM[kpt.q], d)
        else:
            if direction is not None:
                remove_pbc(atoms, H_qMM[kpt.s, kpt.q], None, d)
        if calc.occupations.width > 0:
            H_qMM[kpt.s, kpt.q] -= S_qMM[kpt.q] * \
                                   calc.occupations.get_fermi_level()    
    
    if wfs.gd.comm.rank == 0:
        fd = file(filename+'%i.pckl' % wfs.kpt_comm.rank, 'wb')
        H_qMM *= Hartree
        pickle.dump((H_qMM, S_qMM),fd , 2)
        pickle.dump(calc_data, fd, 2) 
        fd.close()


def get_lcao_hamiltonian(calc):
    """Return H_skMM, S_kMM on master, (None, None) on slaves. H is in eV."""
    if calc.wfs.S_qMM is None:
        calc.wfs.set_positions(calc.get_atoms().get_scaled_positions() % 1)
    dtype = calc.wfs.dtype
    NM = calc.wfs.eigensolver.nao
    Nk = calc.wfs.nibzkpts
    Ns = calc.wfs.nspins
    
    S_kMM = np.zeros((Nk, NM, NM), dtype)
    H_skMM = np.zeros((Ns, Nk, NM, NM), dtype)
    for kpt in calc.wfs.kpt_u:
        H_MM = calc.wfs.eigensolver.calculate_hamiltonian_matrix(
            calc.hamiltonian, calc.wfs, kpt)
        if kpt.s == 0:
            S_kMM[kpt.k] = calc.wfs.S_qMM[kpt.q]
            tri2full(S_kMM[kpt.k])
        H_skMM[kpt.s, kpt.k] = H_MM * Hartree
        tri2full(H_skMM[kpt.s, kpt.k])
    calc.wfs.kpt_comm.sum(S_kMM, MASTER)
    calc.wfs.kpt_comm.sum(H_skMM, MASTER)
    if rank == MASTER:
        return H_skMM, S_kMM
    else:
        return None, None


def get_lead_lcao_hamiltonian(calc, direction='x'):
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    if rank == MASTER:
        return lead_kspace2realspace(H_skMM, S_kMM,
                                     bzk_kc=calc.wfs.bzk_kc,
                                     weight_k=calc.wfs.weight_k,
                                     direction=direction,
                                     usesymm=calc.input_parameters['usesymm'])
    else:
        return None, None, None, None


def lead_kspace2realspace(h_skmm, s_kmm, bzk_kc, weight_k, direction='x', 
                          usesymm=None):
    """Convert a k-dependent Hamiltonian to tight-binding onsite and coupling.

    For each transverse k-point:
    Convert k-dependent (in transport direction) Hamiltonian
    representing a lead to a real space tight-binding Hamiltonian
    of double size representing two principal layers and the coupling between.
    """

    dir = 'xyz'.index(direction)
    if usesymm is True:
        raise NotImplementedError

    R_c = [0, 0, 0]
    ibz_t_kc, weight_t_k, h_skii, s_kii = get_realspace_hs(
        h_skmm, s_kmm, bzk_kc, weight_k, R_c, direction, usesymm)

    R_c[dir] = 1
    h_skij, s_kij = get_realspace_hs(
        h_skmm, s_kmm, bzk_kc, weight_k, R_c, direction, usesymm)[-2:]

    nspins, nk, nbf = h_skii.shape[:-1]

    h_skmm = np.zeros((nspins, nk, 2 * nbf, 2 * nbf), h_skii.dtype)
    s_kmm = np.zeros((nk, 2 * nbf, 2 * nbf), h_skii.dtype)
    h_skmm[:, :, :nbf, :nbf] = h_skmm[:, :, nbf:, nbf:] = h_skii
    h_skmm[:, :, :nbf, nbf:] = h_skij
    h_skmm[:, :, nbf:, :nbf] = h_skij.swapaxes(2, 3).conj()

    s_kmm[:, :nbf, :nbf] = s_kmm[:, nbf:, nbf:] = s_kii
    s_kmm[:, :nbf, nbf:] = s_kij
    s_kmm[:, nbf:, :nbf] = s_kij.swapaxes(1,2).conj()

    return ibz_t_kc, weight_t_k, h_skmm, s_kmm


def zeta_pol(basis):
    """Get number of zeta func. and polarization func. indices in Basis."""
    zeta = []
    pol = []
    for bf in basis.bf_j:
        if 'polarization' in bf.type:
            pol.append(2 * bf.l + 1)
        else:
            zeta.append(2 * bf.l + 1)
    zeta = sum(zeta)
    pol = sum(pol)
    assert zeta + pol == basis.nao
    return zeta, pol


def basis_subset(symbol, largebasis, smallbasis):
    """Title.

    Determine which basis function indices from ``largebasis`` are also
    present in smallbasis.
    """
    blarge = Basis(symbol, largebasis)
    zeta_large, pol_large = zeta_pol(blarge)
    
    bsmall = Basis(symbol, smallbasis)
    zeta_small, pol_small = zeta_pol(bsmall)

    assert zeta_small <= zeta_large
    assert pol_small <= pol_large

    insmall = np.zeros(blarge.nao, bool)
    insmall[:zeta_small] = True
    insmall[zeta_large:zeta_large + pol_small] = True
    return insmall


def basis_subset2(symbols, largebasis='dzp', smallbasis='sz'):
    """Same as basis_subset, but for an entire list of atoms."""
    largebasis = types2atomtypes(symbols, largebasis, default='dzp')
    smallbasis = types2atomtypes(symbols, smallbasis, default='sz')
    mask = []
    for symbol, large, small in zip(symbols, largebasis, smallbasis):
        mask.extend(basis_subset(symbol, large, small))
    return np.asarray(mask, bool)
