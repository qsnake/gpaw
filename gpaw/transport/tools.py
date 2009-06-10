from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as np
from gpaw.mpi import world, rank
from gpaw.utilities.blas import gemm


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

def dagger(matrix):
    return np.conj(matrix.T)

#def get_realspace_hs(h_skmm,s_kmm, ibzk_kc, weight_k, R_c=(0,0,0)):
def k2r_hs(h_skmm,s_kmm, ibzk_kc, weight_k, R_c=(0,0,0)):
    phase_k = np.dot(2 * np.pi * ibzk_kc, R_c)
    c_k = np.exp(1.0j * phase_k) * weight_k
    c_k.shape = (len(ibzk_kc),1,1)

    if h_skmm != None:
        nbf = h_skmm.shape[-1]
        nspins = len(h_skmm)
        h_smm = np.empty((nspins,nbf,nbf),complex)
        for s in range(nspins):
            h_smm[s] = np.sum((h_skmm[s] * c_k), axis=0)
    if s_kmm != None:
        nbf = s_kmm.shape[-1]
        s_mm = np.empty((nbf,nbf),complex)
        s_mm[:] = np.sum((s_kmm * c_k), axis=0)     
    if h_skmm != None and s_kmm != None:
        return h_smm, s_mm
    elif h_skmm == None:
        return s_mm
    elif s_kmm == None:
        return h_smm

def r2k_hs(h_srmm, s_rmm, R_vector, kvector=(0,0,0)):
    phase_k = np.dot(2 * np.pi * R_vector, kvector)
    c_k = np.exp(-1.0j * phase_k)
    c_k.shape = (len(R_vector), 1, 1)
   
    if h_srmm != None:
        nbf = h_srmm.shape[-1]
        nspins = len(h_srmm)
        h_smm = np.empty((nspins, nbf, nbf), complex)
        for s in range(nspins):
            h_smm[s] = np.sum((h_srmm[s] * c_k), axis=0)
    if s_rmm != None:
        nbf = s_rmm.shape[-1]
        s_mm = np.empty((nbf, nbf), complex)
        s_mm[:] = np.sum((s_rmm * c_k), axis=0)
    if h_srmm != None and s_rmm != None:   
        return h_smm, s_mm
    elif h_srmm == None:
        return s_mm
    elif s_rmm == None:
        return h_smm

def get_hs(atoms):
    """Calculate the Hamiltonian and overlap matrix."""
    calc = atoms.calc
    wfs = calc.wfs
    Ef = calc.get_fermi_level()
    eigensolver = wfs.eigensolver
    ham = calc.hamiltonian
    S_qMM = wfs.S_qMM.copy()
    for S_MM in S_qMM:
        tri2full(S_MM)
    H_sqMM = np.empty((wfs.nspins,) + S_qMM.shape, complex)
    for kpt in wfs.kpt_u:
        eigensolver.calculate_hamiltonian_matrix(ham, wfs, kpt)
        H_MM = eigensolver.H_MM
        tri2full(H_MM)
        H_MM *= Hartree
        H_MM -= Ef * S_qMM[kpt.q]
        H_sqMM[kpt.s, kpt.q] = H_MM
    return H_sqMM, S_qMM

def substract_pk(d, npk, ntk, kpts, k_mm, hors='s', position=[0, 0, 0]):
    weight = np.array([1.0 / ntk] * ntk )
    if hors not in 'hs':
        raise KeyError('hors should be h or s!')
    if hors == 'h':
        dim = k_mm.shape[:]
        dim = (dim[0],) + (dim[1] / ntk,) + dim[2:]
        pk_mm = np.empty(dim, k_mm.dtype)
        dim = (dim[0],) + (ntk,) + dim[2:]
        tk_mm = np.empty(dim, k_mm.dtype)
    elif hors == 's':
        dim = k_mm.shape[:]
        dim = (dim[0] / ntk,) + dim[1:]
        pk_mm = np.empty(dim, k_mm.dtype)
        dim = (ntk,) + dim[1:]
        tk_mm = np.empty(dim, k_mm.dtype)

    tkpts = pick_out_tkpts(d, npk, ntk, kpts)
    for i in range(npk):
        n = i * ntk
        for j in range(ntk):
            if hors == 'h':
                tk_mm[:, j] = np.copy(k_mm[:, n + j])
            elif hors == 's':
                tk_mm[j] = np.copy(k_mm[n + j])
        if hors == 'h':
            pk_mm[:, i] = k2r_hs(tk_mm, None, tkpts, weight, position)
        elif hors == 's':
            pk_mm[i] = k2r_hs(None, tk_mm, tkpts, weight, position)
    return pk_mm   

def pick_out_tkpts(d, npk, ntk, kpts):
    tkpts = np.zeros([ntk, 3])
    for i in range(ntk):
        tkpts[i, d] = kpts[i, d]
    return tkpts

def count_tkpts_num(d, kpts):
    tol = 1e-6
    tkpts = [kpts[0]]
    for kpt in kpts:
        flag = False
        for tkpt in tkpts:
            if abs(kpt[d] - tkpt[d]) < tol:
                flag = True
        if not flag:
            tkpts.append(kpt)
    return len(tkpts)
    
def dot(a, b):
    assert len(a.shape) == 2 and a.shape[1] == b.shape[0]
    dtype = complex
    if a.dtype == complex and b.dtype == complex:
        c = a
        d = b
    elif a.dtype == float and b.dtype == complex:
        c = np.array(a, complex)
        d = b
    elif a.dtype == complex and b.dtype == float:
        d = np.array(b, complex)
        c = a
    else:
        dtype = float
        c = a
        d = b
    e = np.zeros([c.shape[0], d.shape[1]], dtype)
    gemm(1.0, d, c, 0.0, e)
    return e

def plot_diag(mtx, ind=1):
    import pylab
    dim = mtx.shape
    if len(dim) != 2:
        print 'Warning! check the dimenstion of the matrix'
    if dim[0] != dim[1]:
        print 'Warinng! check if the matrix is square'
    diag_element = np.diag(mtx)
    y_data = pick(diag_element, ind)
    x_data = range(len(y_data))
    pylab.plot(x_data, y_data,'b-o')
    pylab.show()

class P_info:
    def __init__(self):
        P.x = 0
        P.y = 0
        P.z = 0
        P.Pxsign = 1
        P.Pysign = 1
        P.Pzsign = 1
        P.N = 0
class D_info:
    def __init__(self):
        D.xy = 0
        D.xz = 0
        D.yz = 0
        D.x2y2 = 0
        D.z2r2 = 0
        D.N = 0

def PutP(index, X, P, T):
    if P.N == 0:
        P.x = index
    if P.N == 1:
        P.y = index
    if P.N == 2:
        P.z = index
    P.N += 1
    
    if P.N == 3:
        bs = np.array([P.x, P.y, P.z])
        c = np.array([P.Pxsign, P.Pysign, P.Pzsign])
        c = np.resize(c, [3, 3])
        cf = c / c.T
        ind = np.resize(bs, [3, 3])
        T[ind.T, ind] = X * cf 
        P.__init__()
        
def PutD(index, X, D, T):
    if D.N == 0:
        D.xy = index
    if D.N == 1:
        D.xz = index
    if D.N == 2:
        D.yz = index
    if D.N == 3:
        D.x2y2 = index
    if D.N == 4:
        D.z2r2 = index
        
    D.N += 1
    if D.N == 5:
        sqrt = np.sqrt
        Dxy = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]])
        D2xy = np.dot(X, Dxy)
        D2xy = np.dot(D2xy, X.T)
        
        Dxz = np.array([[0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 0]])
        D2xz = np.dot(X, Dxz)
        D2xz = np.dot(D2xz, X.T)
        
        Dyz = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0]])
        D2yz = np.dot(X, Dyz)
        D2yz = np.dot(D2yz, X.T)

        Dx2y2 = np.array([[1, 0 , 0],
                          [0, -1, 0],
                          [0, 0,  0]])
        D2x2y2 = np.dot(X, Dx2y2)
        D2x2y2 = np.dot(D2x2y2, X.T)
        
        Dz2r2 = np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0,  0, 2]]) / sqrt(3)
        D2z2r2 = np.dot(X, D2z2r2)
        D2z2r2 = np.dot(D2z2r2, X.T)
        
        T[D.xy, D.xy] = D2xy[0, 1]               
        T[D.xz, D.xy] = D2xy[0, 2]               
        T[D.yz, D.xy] = D2xy[1, 2]               
        T[D.x2y2, D.xy] = (D2xy[0, 0] - D2xy[1, 1]) / 2 
        T[D.z2r2, D.xy] = sqrt(3) / 2 * D2xy[2, 2]     

        T[D.xy, D.xz] = D2xz[0, 1]               
        T[D.xz, D.xz] = D2xz[0, 2]               
        T[D.yz, D.xz] = D2xz[1, 2]               
        T[D.x2y2, D.xz] = (D2xz[0, 0] - D2xz[1, 1]) / 2 
        T[D.z2r2, D.xz] = sqrt(3) / 2 * D2xz[2,2];     

        T[D.xy , D.yz] = D2yz[0, 1]               
        T[D.xz , D.yz] = D2yz[0, 2]               
        T[D.yz , D.yz] = D2yz[1, 2]               
        T[D.x2y2, D.yz] = (D2yz[0, 0] - D2yz[1, 1]) / 2 
        T[D.z2r2, D.yz] = sqrt(3) / 2 * D2yz[2, 2]     

        T[D.xy , D.x2y2] = D2x2y2[0, 1]               
        T[D.xz , D.x2y2] = D2x2y2[0, 2]               
        T[D.yz , D.x2y2] = D2x2y2[1, 2]               
        T[D.x2y2, D.x2y2] = (D2x2y2[0, 0] - D2x2y2[1, 1]) / 2 
        T[D.z2r2, D.x2y2] = sqrt(3) / 2 * D2x2y2[2, 2]     

        T[D.xy, D.z2r2] = D2z2r2[0, 1]               
        T[D.xz, D.z2r2] = D2z2r2[0, 2]               
        T[D.yz, D.z2r2] = D2z2r2[1, 2]               
        T[D.x2y2, D.z2r2] = (D2z2r2[0, 0] - D2z2r2[1, 1]) / 2 
        T[D.z2r2, D.z2r2] = sqrt(3) / 2 * D2z2r2[2, 2]     
        
        D.__init__()      
        
def orbital_matrix_rotate_transformation(mat, X, basis_info):
    nb = len(basis_info)
    assert len(X) == 3 and nb == len(mat)
    T = np.zeros([nb, nb])
    P = P_info()
    D = D_info()
    for i in range(nb):
        if basis_info[i] == 's':
            T[i, i] = 1
        elif basis_info[i] == 'p':
            PutP(i, X, P, T)
        elif basis_info[i] == 'd':
            PutD(i, X, D, T)
        else:
            raise NotImplementError('undown shell name')

