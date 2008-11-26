from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as npy
from gpaw.mpi import world, rank

def tri2full(M,UL='L'):
    """UP='L' => fill upper triangle from lower triangle
       such that M=M^d"""
    nbf = len(M)
    if UL=='L':
        for i in range(nbf-1):
            M[i,i:] = M[i:,i].conjugate()
    elif UL=='U':
        for i in range(nbf-1):
            M[i:,i] = M[i,i:].conjugate()

def get_bf_centers(atoms):
    calc = atoms.get_calculator()
    if not calc.initialized:
        calc.initialize(atoms)
    nbf = calc.nao
    pos_ac = atoms.get_positions()
    natoms = len(pos_ac)
    pos_ic = npy.zeros((nbf,3), npy.float)
    index = 0
    for a in xrange(natoms):
        nao = calc.nuclei[a].get_number_of_atomic_orbitals()
        pos_c = pos_ac[a]
        pos_c.shape = (1,3)
        pos_ic[index:index+nao] = npy.repeat(pos_c, nao, axis=0)
        index += nao
    return pos_ic

def get_realspace_hs(h_skmm,s_kmm, ibzk_kc, weight_k, R_c=(0,0,0)):
    phase_k = npy.dot(2 * npy.pi * ibzk_kc, R_c)
    c_k = npy.exp(1.0j * phase_k) * weight_k
    c_k.shape = (len(ibzk_kc),1,1)

    if h_skmm != None:
        nbf = h_skmm.shape[-1]
        nspins = len(h_skmm)
        h_smm = npy.empty((nspins,nbf,nbf),complex)
        for s in range(nspins):
            h_smm[s] = npy.sum((h_skmm[s] * c_k), axis=0)
    elif s_kmm != None:
        nbf = s_kmm.shape[-1]
        s_mm = npy.empty((nbf,nbf),complex)
        s_mm[:] = npy.sum((s_kmm * c_k), axis=0)      
    if h_skmm != None and s_kmm != None:
        return h_smm, s_mm
    elif h_skmm == None:
        return s_mm
    elif s_kmm == None:
        return h_smm

def get_kspace_hs(h_srmm, s_rmm, R_vector, kvector=(0,0,0)):
    phase_k = npy.dot(2 * npy.pi * R_vector, kvector)
    c_k = npy.exp(-1.0j * phase_k)
    c_k.shape = (len(R_vector), 1, 1)
    
    if h_srmm != None:
        nbf = h_srmm.shape[-1]
        nspins = len(h_srmm)
        h_smm = npy.empty((nspins, nbf, nbf), complex)
        for s in range(nspins):
            h_smm[s] = npy.sum((h_srmm[s] * c_k), axis=0)
    elif s_rmm != None:
        nbf = s_rmm.shape[-1]
        s_mm = npy.empty((nbf, nbf), complex)
        s_mm[:] = npy.sum((s_rmm * c_k), axis=0)
    if h_srmm != None and s_rmm != None:    
        return h_smm, s_mm
    elif h_srmm == None:
        return s_mm
    elif s_rmm == None:
        return h_smm

def remove_pbc(atoms, h, s=None, d=0):
    calc = atoms.get_calculator()
    if not calc.initialized:
        calc.initialize(atoms)
    nbf = calc.nao
    
    cutoff = atoms.get_cell()[d,d] * 0.5 
    pos_i = get_bf_centers(atoms)[:,d]
    for i in xrange(nbf):
        dpos_i = npy.absolute(pos_i - pos_i[i])
        mask_i = (dpos_i < cutoff).astype(int)
        h[i,:] = h[i,:] * mask_i
        h[:,i] = h[:,i] * mask_i
        if s != None:
            s[i,:] = s[i,:] * mask_i
            s[:,i] = s[:,i] * mask_i

def dump_hamiltonian(filename, atoms, direction=None):
    
    h_skmm, s_kmm = get_hamiltonian(atoms)
    if direction!=None:
        d = {'x':0, 'y':1, 'z':2}[direction]
        for s in range(atoms.calc.nspins):
            for k in range(atoms.calc.nkpts):
                if s==0:
                    remove_pbc(atoms, h_skmm[s,k], s_kmm[k], d)
                else:
                    remove_pbc(atoms, h_skmm[s,k], None, d)

    
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

def get_hamiltonian(atoms):
    """Calculate the Hamiltonian and overlap matrix."""
    calc = atoms.calc
    Ef = calc.get_fermi_level()
    eigensolver = calc.eigensolver
    hamiltonian = calc.hamiltonian
    Vt_skmm = eigensolver.Vt_skmm
    print "Calculating effective potential matrix (%i)" % rank 
    hamiltonian.calculate_effective_potential_matrix(Vt_skmm)
    ibzk_kc = calc.ibzk_kc
    nkpts = len(ibzk_kc)
    nspins = calc.nspins
    weight_k = calc.weight_k
    nao = calc.nao
    h_skmm = npy.zeros((nspins, nkpts, nao, nao), complex)
    s_kmm = npy.zeros((nkpts, nao, nao), complex)
    for k in range(nkpts):
        s_kmm[k] = hamiltonian.S_kmm[k]
        tri2full(s_kmm[k])
        for s in range(nspins):
            h_skmm[s,k] = calc.eigensolver.get_hamiltonian_matrix(hamiltonian,
                                                                  k=k,
                                                                  s=s)
            tri2full(h_skmm[s, k])
            h_skmm[s,k] *= Hartree
            h_skmm[s,k] -= Ef * s_kmm[k]

    return h_skmm, s_kmm


