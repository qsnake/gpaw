from gpaw.utilities import unpack
from gpaw.utilities.tools import tri2full
from ase.units import Hartree
import cPickle as pickle
import numpy as np
from gpaw.mpi import world, rank, MASTER
from gpaw.basis_data import Basis
from gpaw.setup import types2atomtypes


def get_bf_centers(atoms, basis=None):
    calc = atoms.get_calculator()
    if calc is None:
        basis_a = types2atomtypes(symbols, basis, 'dzp')
        symbols = atoms.get_chemical_symbols()
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
    """mulliken charges from a list atom indices (a_list). """
    Q_a = {}
    for kpt in calc.wfs.kpt_u:
        S_MM = calc.wfs.S_qMM[kpt.q]
        nao = S_MM.shape[0]
        rho_MM = np.empty((nao, nao), calc.wfs.dtype)
        calc.wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, rho_MM)
        Q_M = np.dot(rho_MM, S_MM).diagonal()
        for a in a_list:
            M1 = calc.wfs.basis_functions.M_a[a]
            M2 = M1 + calc.wfs.setups[a].niAO
            Q_a[a] = np.sum(Q_M[M1:M2])
    return Q_a        


def get_realspace_hs(h_skmm, s_kmm, ibzk_kc, weight_k, R_c=(0, 0, 0),
                     usesymm=None):
    # usesymm=False only works if the k-point reduction is only along one
    # direction.
    # For more functionality, see: gpaw/transport/tools.py
    
    nspins, nk, nbf = h_skmm.shape[:-1]
    c_k = np.exp(2.j * np.pi * np.dot(ibzk_kc, R_c)) * weight_k
    c_k.shape = (nk, 1, 1)

    if usesymm is None:
        h_smm = np.sum((h_skmm * c_k), axis=1)
        if s_kmm is not None:
            s_mm = np.sum((s_kmm * c_k), axis=0)
    elif usesymm is False:
        h_smm = np.sum((h_skmm * c_k).real, axis=1)
        if s_kmm is not None:
            s_mm = np.sum((s_kmm * c_k).real, axis=0)
    else: #usesymm is True:
        raise NotImplementedError, 'Only None and False have been implemented'

    if s_kmm is None:
        return h_smm
    return h_smm, s_mm


def remove_pbc(atoms, h, s=None, d=0, centers_ic=None, cutoff=None):
    L = atoms.cell[d, d]
    nao = len(h)
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
        mask_i = (dpos_i < cutoff).astype(int)
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
#        print 'Calc. H matrix on proc. %i: (rk, rd, q, k)=(%i, %i, %i, %i)' % (wfs.world.rank, wfs.kpt_comm.rank, wfs.gd.domain.comm.rank, kpt.q, kpt.k)
        wfs.eigensolver.calculate_hamiltonian_matrix(calc.hamiltonian,
                                                     wfs, 
                                                     kpt)

        H_qMM[kpt.s, kpt.q] = wfs.eigensolver.H_MM

        tri2full(H_qMM[kpt.s, kpt.q])
        if kpt.s==0:
            tri2full(S_qMM[kpt.q])
            if direction!=None:
                remove_pbc(atoms, H_qMM[kpt.s, kpt.q], S_qMM[kpt.q], d)
        else:
            if direction!=None:
                remove_pbc(atoms, H_qMM[kpt.s, kpt.q], None, d)
        if calc.occupations.kT>0:
            H_qMM[kpt.s, kpt.q] -= S_qMM[kpt.q] * \
                                   calc.occupations.get_fermi_level()    
    
    if wfs.gd.comm.rank == 0:
        fd = file(filename+'%i.pckl' % wfs.kpt_comm.rank, 'wb')
        H_qMM *= Hartree
        pickle.dump((H_qMM, S_qMM),fd , 2)
        calc_data
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
        calc.wfs.eigensolver.calculate_hamiltonian_matrix(
            calc.hamiltonian, calc.wfs, kpt)
        if kpt.s == 0:
            S_kMM[kpt.k] = calc.wfs.S_qMM[kpt.q]
            tri2full(S_kMM[kpt.k])
        H_skMM[kpt.s, kpt.k] = calc.wfs.eigensolver.H_MM * Hartree
        tri2full(H_skMM[kpt.s, kpt.k])
    calc.wfs.kpt_comm.sum(S_kMM, MASTER)
    calc.wfs.kpt_comm.sum(H_skMM, MASTER)
    if rank == MASTER:
        return H_skMM, S_kMM
    else:
        return None, None


def get_lead_lcao_hamiltonian(calc, usesymm=False, direction='x'):
    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    if rank == MASTER:
        return lead_kspace2realspace(H_skMM, S_kMM, calc.wfs.ibzk_kc,
                                     calc.wfs.weight_k, direction, usesymm)
    else:
        return None, None


def lead_kspace2realspace(h_skmm, s_kmm, ibzk_kc, weight_k,
                          direction='x', usesymm=None):
    """Convert a k-dependent (in transport dir) Hamiltonian representing
    a lead, to a realspace hamiltonian of double size representing two
    principal layers and the coupling between."""
    dir = 'xyz'.index(direction)
    nspin, nk, nbf = h_skmm.shape[:-1]
    h_smm = np.zeros((nspin, 2 * nbf, 2 * nbf), h_skmm.dtype)
    s_mm = np.zeros((2 * nbf, 2 * nbf), h_skmm.dtype)

    R_c = [0, 0, 0]
    h_sii, s_ii = get_realspace_hs(h_skmm, s_kmm, ibzk_kc, weight_k,
                                   R_c, usesymm)
    R_c[dir] = 1.
    h_sij, s_ij = get_realspace_hs(h_skmm, s_kmm, ibzk_kc, weight_k,
                                   R_c, usesymm)

    h_smm[:, :nbf, :nbf] = h_smm[:, nbf:, nbf:] = h_sii
    h_smm[:, :nbf, nbf:] = h_sij
    h_smm[:, nbf:, :nbf] = h_sij.swapaxes(1, 2).conj()

    s_mm[:nbf, :nbf] = s_mm[nbf:, nbf:] = s_ii
    s_mm[:nbf, nbf:] = s_ij
    s_mm[nbf:, :nbf] = s_ij.T.conj()

    return h_smm, s_mm


def zeta_pol(basis):
    """Get number of zeta func. and polarization func. indices in Basis."""
    zeta = 0
    for bf in basis.bf_j:
        if 'polarization' in bf.type:
            break
        zeta += 2 * bf.l + 1
    pol = basis.nao - zeta
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
