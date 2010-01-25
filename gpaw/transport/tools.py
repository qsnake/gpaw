from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as np
from gpaw.mpi import world, rank
from gpaw.utilities.blas import gemm
from gpaw.utilities.timing import Timer
from gpaw.utilities.lapack import inverse_general
import copy
import _gpaw

def tw(mat, filename):
    fd = file(filename, 'wb')
    pickle.dump(mat, fd, 2)
    fd.close()

def tr(filename):
    fd = file(filename, 'r')
    mat = pickle.load(fd)
    fd.close()
    return mat

def write(filename, name, data, dimension, dtype=float):
    import gpaw.io.tar as io
    if world.rank == 0:
        w = io.Writer(filename)
        dim = ()
        for i in range(len(dimension)):
            w.dimension(str(i), dimension[i])
            dim += (str(i),)
        w.add(name, dim, dtype=dtype)
        w.fill(data)
        w.close()

def fermidistribution(energy, kt):
    #fermi level is fixed to zero
    return 1.0 / (1.0 + np.exp(energy / kt) )

def get_tri_type(mat):
    #mat is lower triangular or upper triangular matrix
    tol = 1e-10
    mat = abs(mat)
    dim = mat.shape[-1]
    sum = [0, 0]
    for i in range(dim):
        sum[0] += np.trace(mat, -j)
        sum[1] += np.trace(mat, j)
    diff = sum[0] - sum[1]
    if diff >= 0:
        ans = 'L'
    elif diff < 0:
        ans = 'U'
    if abs(diff) < tol:
        print 'Warning: can not define the triangular matrix'
    return ans
    
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

def get_matrix_index(ind1, ind2=None):
    if ind2 == None:
        dim1 = len(ind1)
        return np.resize(ind1, (dim1, dim1))
    else:
        dim1 = len(ind1)
        dim2 = len(ind2)
    return np.resize(ind1, (dim2, dim1)).T, np.resize(ind2, (dim1, dim2))
    
def aa1d(a, d=2):
    # array average in one dimension
    dim = a.shape
    b = [np.sum(np.take(a, [i], axis=d)) for i in range(dim[d])]
    b = np.array(b)
    b = (b * dim[d]) / np.product(dim)
    return b
    
def aa2d(a, d=0):
    # array average in two dimensions
    b = np.sum(a, axis=d) / a.shape[d]
    return b

#def get_realspace_hs(h_skmm,s_kmm, ibzk_kc, weight_k, R_c=(0,0,0)):
def k2r_hs(h_skmm, s_kmm, ibzk_kc, weight_k, R_c=(0,0,0)):
    phase_k = np.dot(2 * np.pi * ibzk_kc, R_c)
    c_k = np.exp(1.0j * phase_k) * weight_k
    c_k.shape = (len(ibzk_kc),1,1)

    if h_skmm != None:
        nbf = h_skmm.shape[-1]
        nspins = len(h_skmm)
        h_smm = np.empty((nspins, nbf, nbf),complex)
        for s in range(nspins):
            h_smm[s] = np.sum((h_skmm[s] * c_k), axis=0)
    if s_kmm != None:
        nbf = s_kmm.shape[-1]
        s_mm = np.empty((nbf, nbf),complex)
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

def collect_lead_mat(lead_hsd, lead_couple_hsd, s, pk, flag='S'):
    diag_h = []
    upc_h = []
    dwnc_h = []
    for i, hsd, c_hsd in zip(range(len(lead_hsd)), lead_hsd, lead_couple_hsd):
        if flag == 'S':
            band_mat, cp_mat = hsd.S[pk], c_hsd.S[pk]
        elif flag == 'H':
            band_mat, cp_mat = hsd.H[s][pk], c_hsd.H[s][pk]
        else:
            band_mat, cp_mat = hsd.D[s][pk], c_hsd.D[s][pk]
        diag_h.append(band_mat)
        upc_h.append(cp_mat.recover('c'))
        dwnc_h.append(cp_mat.recover('n'))
    return diag_h, upc_h, dwnc_h        
        
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
        H_MM = eigensolver.calculate_hamiltonian_matrix(ham, wfs, kpt)
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
    
def dot(a, b, transa='n'):
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
    assert d.flags.contiguous and c.flags.contiguous
    gemm(1.0, d, c, 0.0, e, transa)
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

def get_atom_indices(subatoms, setups):
    basis_list = [setup.niAO for setup in setups]
    index = []
    for j, lj  in zip(subatoms, range(len(subatoms))):
        begin = np.sum(np.array(basis_list[:j], int))
        for n in range(basis_list[j]):
            index.append(begin + n) 
    return np.array(index, int)    

def mp_distribution(e, kt, n=1):
    x = e / kt
    re = 0.5 * error_function(x)
    for i in range(n):
        re += coff_function(i + 1) * hermite_poly(2 * i + 1, x) * np.exp(-x**2) 
    return re        

def coff_function(n):
    return (-1)**n / (np.product(np.arange(1, n + 1)) * 4.** n * np.sqrt(np.pi))
    
def hermite_poly(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * hermite_poly(n - 1, x) \
                                      - 2 * (n - 1) * hermite_poly(n - 2 , x)

def error_function(x):
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * np.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def sum_by_unit(x, unit):
    dim = x.shape[0]
    dim1 = int(np.ceil(dim / unit))
    y = np.empty([dim1], dtype=x.dtype)
    for i in range(dim1 - 1):
        y[i] = np.sum(x[i * unit: (i + 1) * unit]) / unit
    y[0] = y[1]
    y[-1] = y[-2]
    return y

def diag_cell(cell):
    if len(cell.shape) == 2:
        cell = np.diag(cell)
    return cell
    
def get_pk_hsd(d, ntk, kpts, hl_skmm, sl_kmm, dl_skmm, txt=None,
                                                  dtype=complex, direction=0):
    npk = len(kpts) / ntk
    position = [0, 0, 0]
    hl_spkmm = substract_pk(d, npk, ntk, kpts, hl_skmm, hors='h')
    dl_spkmm = substract_pk(d, npk, ntk, kpts, dl_skmm, hors='h')
    sl_pkmm = substract_pk(d, npk, ntk, kpts, sl_kmm, hors='s')
    
    if direction==0:
        position[d] = 1.0
    else:
        position[d] = -1.0
    
    hl_spkcmm = substract_pk(d, npk, ntk, kpts, hl_skmm, 'h', position)
    dl_spkcmm = substract_pk(d, npk, ntk, kpts, dl_skmm, 'h', position)
    sl_pkcmm = substract_pk(d, npk, ntk, kpts, sl_kmm, 's', position)
    
    tol = 1e-6
    position[d] = 2.0
    s_test = substract_pk(d, npk, ntk, kpts, sl_kmm, 's', position)
    
    matmax = np.max(abs(s_test))
    if matmax > tol:
        if txt != None:
            txt('Warning*: the principle layer should be larger, \
                                                      matmax=%f' % matmax)
        else:
            print 'Warning*: the principle layer should be larger, \
                                                      matmax=%f' % matmax
    if dtype == float:
        hl_spkmm = np.real(hl_spkmm).copy()
        sl_pkmm = np.real(sl_pkmm).copy()
        dl_spkmm = np.real(dl_spkmm).copy()
        hl_spkcmm = np.real(hl_spkcmm).copy()
        sl_pkcmm = np.real(sl_pkcmm).copy()
        dl_spkcmm = np.real(dl_spkcmm).copy()
    return hl_spkmm, sl_pkmm, dl_spkmm * ntk, hl_spkcmm, \
                                                    sl_pkcmm, dl_spkcmm * ntk
   
def get_lcao_density_matrix(calc):
    wfs = calc.wfs
    kpts = wfs.ibzk_qc
    nq = len(kpts)
    my_ns = len(wfs.kpt_u) / nq
    nao = wfs.setups.nao
    
    # calculate_density_matrix involves gemm and doesn't work well with empty()
    d_skmm = np.zeros([my_ns, nq, nao, nao], wfs.dtype)
    for kpt in wfs.kpt_u:
        if my_ns == 1:
            wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, d_skmm[0, kpt.q])
        else:
            wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, d_skmm[kpt.s, kpt.q])            
    return d_skmm

def generate_selfenergy_database(atoms, ntk, filename, direction=0, kt=0.1,
                                 bias=[-3,3], depth=3):
    from gpaw.transport.sparse_matrix import Banded_Sparse_HSD, CP_Sparse_HSD, Se_Sparse_Matrix
    from gpaw.transport.selfenergy import LeadSelfEnergy
    from gpaw.transport.contour import Contour
    hl_skmm, sl_kmm = get_hs(atoms)
    dl_skmm = get_lcao_density_matrix(atoms.calc)
    fermi = atoms.calc.get_fermi_level()
    wfs = atoms.calc.wfs
    hl_spkmm, sl_pkmm, dl_spkmm,  \
    hl_spkcmm, sl_pkcmm, dl_spkcmm = get_pk_hsd(2, ntk,
                                                wfs.ibzk_qc,
                                                hl_skmm, sl_kmm, dl_skmm,
                                                None, wfs.dtype,
                                                direction=direction)    
    my_npk = len(wfs.ibzk_qc) / ntk
    my_nspins = len(wfs.kpt_u) / ( my_npk * ntk)
    
    lead_hsd = Banded_Sparse_HSD(wfs.dtype, my_nspins, my_npk)
    lead_couple_hsd = CP_Sparse_HSD(wfs.dtype, my_nspins, my_npk)
    for pk in range(my_npk):
        lead_hsd.reset(0, pk, sl_pkmm[pk], 'S', init=True)
        lead_couple_hsd.reset(0, pk, sl_pkcmm[pk], 'S', init=True)
        for s in range(my_nspins):
            lead_hsd.reset(s, pk, hl_spkmm[s, pk], 'H', init=True)     
            lead_hsd.reset(s, pk, dl_spkmm[s, pk], 'D', init=True)
            lead_couple_hsd.reset(s, pk, hl_spkcmm[s, pk], 'H', init=True)     
            lead_couple_hsd.reset(s, pk, dl_spkcmm[s, pk], 'D', init=True)          
    lead_se = LeadSelfEnergy(lead_hsd, lead_couple_hsd)
    contour = Contour(kt, fermi, bias, depth)    
    contour.get_dense_contour()
    
    index = []
    se = []
    fd = file(filename, 'w')
    for path in contour.paths:
        for nid in path.nids:
            flags = path.get_flags(nid, path_flag=True)
            energy = path.get_energy(flags[1:])
            index.append([nid, energy])
            se.append(lead_se(energy))
    pickle.dump((index, se), fd, 2)
    fd.close()    

def test_selfenergy_interpolation(atoms, ntk, filename, begin, end, base, scale, direction=0):
    from gpaw.transport.sparse_matrix import Banded_Sparse_HSD, CP_Sparse_HSD, Se_Sparse_Matrix
    from gpaw.transport.selfenergy import LeadSelfEnergy
    from gpaw.transport.contour import Contour
    hl_skmm, sl_kmm = get_hs(atoms)
    dl_skmm = get_lcao_density_matrix(atoms.calc)
    fermi = atoms.calc.get_fermi_level()
    wfs = atoms.calc.wfs
    hl_spkmm, sl_pkmm, dl_spkmm,  \
    hl_spkcmm, sl_pkcmm, dl_spkcmm = get_pk_hsd(2, ntk,
                                                wfs.ibzk_qc,
                                                hl_skmm, sl_kmm, dl_skmm,
                                                None, wfs.dtype,
                                                direction=direction)    
    my_npk = len(wfs.ibzk_qc) / ntk
    my_nspins = len(wfs.kpt_u) / ( my_npk * ntk)
    
    lead_hsd = Banded_Sparse_HSD(wfs.dtype, my_nspins, my_npk)
    lead_couple_hsd = CP_Sparse_HSD(wfs.dtype, my_nspins, my_npk)
    for pk in range(my_npk):
        lead_hsd.reset(0, pk, sl_pkmm[pk], 'S', init=True)
        lead_couple_hsd.reset(0, pk, sl_pkcmm[pk], 'S', init=True)
        for s in range(my_nspins):
            lead_hsd.reset(s, pk, hl_spkmm[s, pk], 'H', init=True)     
            lead_hsd.reset(s, pk, dl_spkmm[s, pk], 'D', init=True)
            lead_couple_hsd.reset(s, pk, hl_spkcmm[s, pk], 'H', init=True)     
            lead_couple_hsd.reset(s, pk, dl_spkcmm[s, pk], 'D', init=True)          
    lead_se = LeadSelfEnergy(lead_hsd, lead_couple_hsd)
    begin += fermi
    end += fermi
    
    ee = np.linspace(begin, end, base)
    cmp_ee = np.linspace(begin, end, base * scale)
  
    se = []
    cmp_se = []
    from scipy import interpolate

    for e in ee:
        se.append(lead_se(e).recover())
    se = np.array(se)
    ne, ny, nz= se.shape
    nie = len(cmp_ee)
    data = np.zeros([nie, ny, nz], se.dtype)
    for yy in range(ny):
        for zz in range(nz):
            ydata = se[:, yy, zz]
            f = interpolate.interp1d(ee, ydata)
            data[:, yy, zz] = f(cmp_ee)
    inter_se_linear = data
    
    for e in cmp_ee:
        cmp_se.append(lead_se(e).recover())
    
    fd = file(filename, 'w')
    pickle.dump((cmp_se, inter_se_linear, ee, cmp_ee), fd, 2)
    fd.close()
    
    for i,e in enumerate(cmp_ee):
        print e, np.max(abs(cmp_se[i] - inter_se_linear[i])), 'linear', np.max(abs(cmp_se[i]))



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

def interpolate_2d(mat):
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.transformers import Transformer
    nn = 10
    N_c = np.zeros([3], dtype=int)
    N_c[1:] = mat.shape[:2]
    N_c[0] = nn
    bmat = np.resize(mat, N_c)
    gd = GridDescriptor(N_c, N_c)
    finegd = GridDescriptor(N_c * 2, N_c)
    interpolator = Transformer(gd, finegd, 3, allocate=False)
    interpolator.allocate()
    fine_bmat = finegd.zeros()
    interpolator.apply(bmat, fine_bmat)
    return fine_bmat[0]
    
def interpolate_array(array, gd, h, di='+'):
    try:
        from scipy import interpolate
        ip = True
    except ImportError:
        ip = False
    if not ip:
        return array

    dim = len(array.shape)
    assert dim == 3 or dim == 4
    spin_relate = dim == 4
    if h <= gd.h_c[2]:
        if di == '+':
            x = np.arange(gd.N_c[2]) * gd.h_c[2]
            xnew = np.arange(gd.N_c[2]) * h
        else:
            x = np.arange(-gd.N_c[2], 0) * gd.h_c[2]
            xnew = np.arange(-gd.N_c[2], 0) * h            
    else:
        if di == '+':
            x = np.arange(gd.N_c[2] * 2) * gd.h_c[2]
            xnew = np.arange(gd.N_c[2]) * h
        else:
            x = np.arange(-gd.N_c[2] * 2, 0) * gd.h_c[2]
            xnew = np.arange(-gd.N_c[2], 0) * h         
        
    if spin_relate:
        ns, nx, ny, nz = array.shape
        array.shape = (ns * nx * ny, nz)
        new_array = gd.zeros(ns, global_array=True)
        new_array.shape = (ns * nx * ny, nz)
    else:
        nx, ny, nz = array.shape
        array.shape = (nx * ny, nz)
        new_array = gd.zeros(global_array=True)
        new_array.shape = (nx * ny, nz)
      
    if h > gd.h_c[2]:
        array = np.append(array, array, 1)
        
    for i, line in enumerate(array):
        tck = interpolate.splrep(x, line, s=0)
        new_array[i] = interpolate.splev(xnew, tck, der=0)
    
    if spin_relate:
        new_array.shape = (ns, nx, ny, nz)
    else:
        new_array.shape = (nx, ny, nz)
    
    return new_array
        
def eig_states_norm(orbital, s_mm):
    #normalize orbital to satisfy orbital.T.conj()*SM*orbital=unit
    norm_error = 1e-10
    ortho_error = 1e-8
    nstates = orbital.shape[1]
    d_mm = np.dot(orbital.T.conj(), s_mm)
    d_mm = np.dot(d_mm, orbital)
    for i in range(1, nstates):
        for j in range(i):
            if abs(d_mm[j ,i]) > ortho_error:
                orbital[:, i] -= orbital[:, j] * d_mm[j, i] / d_mm[j ,j]
                d_mm = np.dot(orbital.T.conj(), s_mm)
                d_mm = np.dot(d_mm, orbital)
    for i in range(nstates):
        orbital[:, i] /= np.sqrt(d_mm[i, i])

    if orbital.shape[-1] == 0:
        error = 0
    else:
        error = np.max(np.dot(np.dot(orbital.T.conj(), s_mm), orbital) -
                   np.eye(nstates)) / nstates
  
    if  abs(error) > norm_error:
        print 'Warning! Normalization error %f' % error
    return orbital

def find(condition):
    return np.nonzero(condition)[0]
    
def gather_to_list(comm, data):
    #data is a numpy array, maybe has different shape in different cpus
    #this function gather them to the all_data in master, all_data is
    # a list with the lenth world.size, all_data[i] = data {on i}
    all_data = []
    dim = len(data.shape)
    shape_array = np.zeros([comm.size, dim], int)
    shape_array[comm.rank] = data.shape
    comm.sum(shape_array)
    
    if comm.rank == 0:
        all_data.append(data)
        for i in range(1, comm.size):
            tmp = np.zeros(shape_array[i], dtype=data.dtype)
            comm.receive(tmp, i, 546)
            all_data.append(tmp[:])
    else:
        comm.ssend(data, 0, 546)
    
    return all_data            
            
        
    
    
    
