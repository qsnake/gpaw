from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as np
from gpaw.mpi import world, rank
from gpaw.utilities.blas import gemm
from gpaw.utilities.timing import Timer
import copy
import _gpaw

class PathInfo:
    def __init__(self, type, nlead):
        self.type = type
        self.num = 0
        self.lead_num = nlead
        self.energy = []
        self.weight = []
        self.nres = 0
        self.sigma = []
        for i in range(nlead):
            self.sigma.append([])
        if type == 'eq':
            self.fermi_factor = []
        elif type == 'ne':
            self.fermi_factor = []
            for i in range(nlead):
                self.fermi_factor.append([[], []])
        else:
            raise TypeError('unkown PathInfo type')

    def add(self, elist, wlist, flist, siglist):
        self.num += len(elist)
        self.energy += elist
        self.weight += wlist
        if self.type == 'eq':
            self.fermi_factor += flist
        elif self.type == 'ne':
            for i in range(self.lead_num):
                for j in [0, 1]:
                    self.fermi_factor[i][j] += flist[i][j]
        else:
            raise TypeError('unkown PathInfo type')
        for i in range(self.lead_num):
            self.sigma[i] += siglist[i]

    def set_nres(self, nres):
        self.nres = nres


class Banded_Sparse_HSD:
    #for lead's hamiltonian, overlap, and density matrix
    def __init__(self, dtype, ns, npk, index=None):
        self.band_index = index
        self.dtype = dtype
        self.H = []
        self.S = []
        self.D = []
        self.ns = ns
        self.npk = npk
        self.s = 0
        self.pk = 0
        for s in range(ns):
            self.H.append([])
            self.D.append([])
            for k in range(npk):
                self.H[s].append([])
                self.D[s].append([])
        for k in range(npk):
            self.S.append([])        

    def reset(self, s, pk, mat, flag='S', init=False):
        assert mat.dtype == self.dtype
        if flag == 'S':
            spar = self.S
        elif flag == 'H':
            spar = self.H[s]
        elif flag == 'D':
            spar = self.D[s]
        if not init:
            spar[pk].reset(mat)            
        elif self.band_index != None:
            spar[pk] = Banded_Sparse_Matrix(self.dtype, mat, self.band_index)
        else:
            spar[pk] = Banded_Sparse_Matrix(self.dtype, mat)
            self.band_index = spar[pk].band_index
       
class Banded_Sparse_Matrix:
    def __init__(self, dtype, mat=None, band_index=None, tol=1e-12):
        self.tol = tol
        self.dtype = dtype
        self.band_index = band_index
        if mat != None:
            if band_index == None:
                self.initialize(mat)
            else:
                self.reset(mat)
        
    def initialize(self, mat):
        # the indexing way needs mat[0][-1] = 0,otherwise will recover a
        # unsymmetric full matrix
        assert self.dtype == mat.dtype
        dim = mat.shape[-1]
        ku = -1
        kl = -1
        mat_sum = np.sum(abs(mat))
        spar_sum = 0
        while abs(mat_sum - spar_sum) > self.tol * 10:
            ku += 1
            kl += 1
            ud_sum = 1
            dd_sum = 1
            while(ud_sum > self.tol):
                ku += 1
                ud_sum = np.sum(np.diag(abs(mat), ku))
            while(dd_sum > self.tol):
                kl += 1
                dd_sum = np.sum(np.diag(abs(mat), -kl))
            ku -= 1
            kl -= 1
   
            # storage in the tranpose, bacause column major order for zgbsv_ function
            self.spar = np.zeros([dim, 2 * kl + ku + 1], self.dtype)
                
            index1 = np.zeros([dim, 2 * kl + ku + 1], int)
            index2 = np.zeros([dim, 2 * kl + ku + 1], int)
            
            for i in range(dim):
                index1[i] = i
                for j in range(2 * kl + ku + 1):
                    tmp = i + j - (kl + ku)
                    if 0 <= tmp <= dim -1:
                        index2[i][j] = tmp
                    else:
                        index1[i][j] = 0
                        index2[i][j] = -1
            
            self.band_index = (kl, ku, index1, index2)
            self.spar = mat[index1, index2]
            spar_sum = np.sum(abs(self.recover()))

    def recover(self):
        dim = self.spar.shape[0]
        mat = np.zeros([dim, dim], self.dtype)
        index1, index2 = self.band_index[-2:]
        mat[index1, index2] = self.spar
        return mat
 
    def reset(self, mat):
        index1, index2 = self.band_index[-2:]
        assert self.dtype == mat.dtype
        self.spar = mat[index1, index2]
    
    def reset_from_others(self, bds_mm1, bds_mm2, c1, c2):
        assert self.dtype == complex
        self.spar = c1 * bds_mm1.spar + c2 * bds_mm2.spar 
            
    def reset_minus(self, mat, full=False):
        assert self.dtype == complex
        index1, index2 = self.band_index[-2:]
        if full:
            self.spar -= mat[index1, index2]
        else:
            self.spar -= mat.recover()[index1, index2]
    
    def reset_plus(self, mat, full=False):
        assert self.dtype == complex
        index1, index2 = self.band_index[-2:]
        if full:
            self.spar += mat[index1, index2]
        else:
            self.spar += mat.recover()[index1, index2]           

    def inv(self, keep_data=False):
        kl, ku = self.band_index[:2]
        dim = self.spar.shape[0]
        inv_mat = np.eye(dim, dtype=complex)
        ldab = 2*kl + ku + 1
        if keep_data:
            source_mat = self.spar.copy()
        else:
            source_mat = self.spar
        assert source_mat.flags.contiguous
        info = _gpaw.linear_solve_band(source_mat, inv_mat, kl, ku)            
        return inv_mat
       
class Tp_Sparse_HSD:
    def __init__(self, dtype, ns, npk, ll_index, ex=True):
        self.dtype = dtype
        self.ll_index = ll_index
        self.extended = ex
        self.H = []
        self.S = []
        self.D = []
        self.G = []
        self.ns = ns
        self.npk = npk
        self.s = 0
        self.pk = 0
        self.band_indices = None
        for s in range(ns):
            self.H.append([])
            self.D.append([])
            for k in range(npk):
                self.H[s].append([])
                self.D[s].append([])
        for k in range(npk):
            self.S.append([])
        self.G = Tp_Sparse_Matrix(complex, self.ll_index,
                                                    None, None, self.extended)
    
    def reset(self, s, pk, mat, flag='S', init=False):
        if flag == 'S':
            spar = self.S
        elif flag == 'H':
            spar = self.H[s]
        elif flag == 'D':
            spar = self.D[s]
        if not init:
            spar[pk].reset(mat)
        elif self.band_indices == None:
            spar[pk] = Tp_Sparse_Matrix(self.dtype, self.ll_index, mat,
                                                          None, self.extended)
            self.band_indices = spar[pk].band_indices
        else:
            spar[pk] = Tp_Sparse_Matrix(self.dtype, self.ll_index, mat,
                                             self.band_indices, self.extended)

    def append_lead_as_buffer(self, lead_hsd, lead_couple_hsd, ex_index):
        assert self.extended == True
        clm = collect_lead_mat
        for pk in range(self.npk):
            diag_h, upc_h, dwnc_h = clm(lead_hsd, lead_couple_hsd, 0, pk)    
            self.S[pk].append_ex_mat(diag_h, upc_h, dwnc_h, ex_index)
            for s in range(self.ns):
                diag_h, upc_h, dwnc_h = clm(lead_hsd,
                                                  lead_couple_hsd, s, pk, 'H')              
                self.H[s][pk].append_ex_mat(diag_h, upc_h, dwnc_h, ex_index)                    
                diag_h, upc_h, dwnc_h = clm(lead_hsd,
                                                  lead_couple_hsd, s, pk, 'D')                 
                self.D[s][pk].append_ex_mat(diag_h, upc_h, dwnc_h, ex_index)                 
  
    def calculate_eq_green_function(self, zp, sigma, ex=True):
        s, pk = self.s, self.pk
        self.G.reset_from_others(self.S[pk], self.H[s][pk], zp, -1, init=True)
        self.G.substract_sigma(sigma)
        #self.G.test_inv_eq()
        self.G.inv_eq()
        return self.G.recover(ex)

    def calculate_ne_green_function(self, zp, sigma, fermi_factors, ex=True):
        s, pk = self.s, self.pk        
        self.G.reset_from_others(self.S[pk], self.H[s][pk], zp, -1)
        self.G.substract_sigma(sigma)
        gamma = []
        for ff, tgt in zip(fermi_factors, sigma):
            full_tgt = tgt.recover()
            gamma.append(ff * 1.j * (full_tgt - full_tgt.T.conj()))
        self.G.calculate_less_green(gamma)
        return self.G.recover(ex)     

    def abstract_sub_green_matrix(self, zp, sigma, l1, l2, inv_mat=None):
        if inv_mat == None:
            s, pk = self.s, self.pk        
            self.G.reset_from_others(self.S[pk], self.H[s][pk], zp, -1)
            self.G.substract_sigma(sigma)            
            inv_mat = self.G.inv_ne()
            gr_sub = inv_mat[l2][l1][-1]
            return gr_sub, inv_mat
        else:
            return gr_sub
       
class Tp_Sparse_Matrix:
    def __init__(self, dtype, ll_index, mat=None, band_indices=None, ex=True):
    # ll_index : lead_layer_index
    # matrix stored here will be changed to inversion

        self.lead_num = len(ll_index)
        self.ll_index = ll_index
        self.ex_ll_index = copy.deepcopy(ll_index[:])
        self.extended = ex
        self.dtype = dtype
        self.initialize()
        self.band_indices = band_indices
        if self.band_indices == None:
            self.initialize_band_indices()
        if mat != None:
            self.reset(mat, True)
        
    def initialize_band_indices(self):
        self.band_indices = [None]
        for i in range(self.lead_num):
            self.band_indices.append([])
            for j in range(self.ex_lead_nlayer[i] - 1):
                self.band_indices[i + 1].append(None)
        
    def initialize(self):
    # diag_h : diagonal lead_hamiltonian
    # upc_h : superdiagonal lead hamiltonian
    # dwnc_h : subdiagonal lead hamiltonian 
        self.diag_h = []
        self.upc_h = []
        self.dwnc_h = []
        self.lead_nlayer = []
        self.ex_lead_nlayer = []
        self.mol_index = self.ll_index[0][0]
        self.nl = 1
        self.nb = len(self.mol_index)
        self.length = self.nb * self.nb
        self.mol_h = []

        for i in range(self.lead_num):
            self.diag_h.append([])
            self.upc_h.append([])
            self.dwnc_h.append([])
            self.lead_nlayer.append(len(self.ll_index[i]))
            if self.extended:
                self.ex_lead_nlayer.append(len(self.ll_index[i]) + 1)
            else:
                self.ex_lead_nlayer.append(len(self.ll_index[i]))
            
            assert (self.ll_index[i][0] == self.mol_index).all()
            self.nl += self.lead_nlayer[i] - 1       
            
            for j in range(self.lead_nlayer[i] - 1):
                self.diag_h[i].append([])
                self.upc_h[i].append([])
                self.dwnc_h[i].append([])
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                self.length += 2 * len1 * len2 + len2 * len2
                self.nb += len2
            
            if self.extended:                
                self.diag_h[i].append([])
                self.upc_h[i].append([])
                self.dwnc_h[i].append([])
        self.ex_nb = self.nb

    def append_ex_mat(self, diag_h, upc_h, dwnc_h, ex_index):
        assert self.extended
        for i in range(self.lead_num):
            self.diag_h[i][-1] = diag_h[i]
            self.upc_h[i][-1] = upc_h[i]
            self.dwnc_h[i][-1] = dwnc_h[i]
            self.ex_ll_index[i].append(ex_index[i])
            self.ex_nb += len(ex_index[i])
  
    def abstract_layer_info(self):
        self.basis_to_layer = np.empty([self.nb], int)
        self.neighbour_layers = np.zeros([self.nl, self.lead_num], int) - 1

        for i in self.mol_index:
            self.basis_to_layer[i] = 0
        nl = 1
        
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                for k in self.ll_index[i][j]:
                    self.basis_to_layer[k] = nl
                nl += 1

        nl = 1                 
        for i in range(self.lead_num):        
            self.neighbour_layers[0][i] = nl
            first = nl
            for j in range(self.lead_nlayer[i] - 1):
                if nl == first:
                    self.neighbour_layers[nl][0] = 0
                    if j != self.lead_nlayer[i] - 2:
                        self.neighbour_layers[nl][1] = nl + 1
                else:
                    self.neighbour_layers[nl][0] = nl - 1
                    if j != self.lead_nlayer[i] - 2:
                        self.neighbour_layers[nl][1] = nl + 1                    
                nl += 1
              
    def reset(self, mat, init=False):
        assert mat.dtype == self.dtype
        ind = get_matrix_index(self.mol_index)
        if init:
            self.mol_h = Banded_Sparse_Matrix(self.dtype, mat[ind.T, ind],
                                               self.band_indices[0])
            if self.band_indices[0] == None:
                self.band_indices[0] = self.mol_h.band_index            
        else:
            self.mol_h.reset(mat[ind.T, ind])

        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                ind = get_matrix_index(self.ll_index[i][j])
                ind1 = get_matrix_index(self.ll_index[i][j + 1])
                indr1, indc1 = get_matrix_index(self.ll_index[i][j],
                                                      self.ll_index[i][j + 1])
                indr2, indc2 = get_matrix_index(self.ll_index[i][j + 1],
                                                  self.ll_index[i][j])
                if init:
                    self.diag_h[i][j] = Banded_Sparse_Matrix(self.dtype,
                                                             mat[ind1.T, ind1],
                                                 self.band_indices[i + 1][j])
                    if self.band_indices[i + 1][j] == None:
                        self.band_indices[i + 1][j] = \
                                                  self.diag_h[i][j].band_index
                else:
                    self.diag_h[i][j].reset(mat[ind1.T, ind1])
                self.upc_h[i][j] = mat[indr1, indc1]
                self.dwnc_h[i][j] = mat[indr2, indc2]
       
    def reset_from_others(self, tps_mm1, tps_mm2, c1, c2, init=False):
        #self.mol_h = c1 * tps_mm1.mol_h + c2 * tps_mm2.mol_h
        if init:
            self.mol_h = Banded_Sparse_Matrix(complex)
        
        self.mol_h.spar = c1 * tps_mm1.mol_h.spar + c2 * tps_mm2.mol_h.spar
        self.mol_h.band_index = tps_mm1.mol_h.band_index
        self.ex_lead_nlayer = tps_mm1.ex_lead_nlayer
        self.ex_ll_index = tps_mm1.ex_ll_index
        self.ex_nb = tps_mm1.ex_nb
        
        for i in range(self.lead_num):
            for j in range(self.ex_lead_nlayer[i] - 1):
                assert (tps_mm1.ex_ll_index[i][j] == tps_mm2.ex_ll_index[i][j]).all()
                if init:
                    self.diag_h[i][j] = Banded_Sparse_Matrix(complex)
                    self.diag_h[i][j].band_index = \
                                             tps_mm1.diag_h[i][j].band_index
                
                self.diag_h[i][j].spar = c1 * tps_mm1.diag_h[i][j].spar + \
                                      c2 * tps_mm2.diag_h[i][j].spar
                self.upc_h[i][j] = c1 * tps_mm1.upc_h[i][j] + \
                                      c2 * tps_mm2.upc_h[i][j]
                self.dwnc_h[i][j] = c1 * tps_mm1.dwnc_h[i][j] + \
                                      c2 * tps_mm2.dwnc_h[i][j]
  
    def substract_sigma(self, sigma):
        if self.extended:
            n = -2
        else:
            n = -1
        for i in range(self.lead_num):
            self.diag_h[i][n].reset_minus(sigma[i])
        
    def recover(self, ex=False):
        if ex:
            nb = self.ex_nb
            lead_nlayer = self.ex_lead_nlayer
            ll_index = self.ex_ll_index
        else:
            nb = self.nb
            lead_nlayer = self.lead_nlayer
            ll_index = self.ll_index            
        
        mat = np.zeros([nb, nb], self.dtype)
        ind = get_matrix_index(ll_index[0][0])
        
        mat[ind.T, ind] = self.mol_h.recover()
        
        gmi = get_matrix_index
        for i in range(self.lead_num):
            for j in range(lead_nlayer[i] - 1):
                ind = gmi(ll_index[i][j])
                ind1 = gmi(ll_index[i][j + 1])
                indr1, indc1 = gmi(ll_index[i][j], ll_index[i][j + 1])
                indr2, indc2 = gmi(ll_index[i][j + 1], ll_index[i][j])                
                mat[ind1.T, ind1] = self.diag_h[i][j].recover()
                mat[indr1, indc1] = self.upc_h[i][j]
                mat[indr2, indc2] = self.dwnc_h[i][j]
        return mat        

    def test_inv_eq(self, tol=1e-12):
        tp_mat = copy.deepcopy(self)
        tp_mat.inv_eq()
        mol_h = dot(tp_mat.mol_h.recover(), self.mol_h.recover())
        for i in range(self.lead_num):
            mol_h += dot(tp_mat.upc_h[i][0], self.dwnc_h[i][0])
        diff = np.max(abs(mol_h - np.eye(mol_h.shape[0])))
        if diff > tol:
            print 'warning, mol_diff', diff
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 2):
                diag_h = dot(tp_mat.diag_h[i][j].recover(),
                                                  self.diag_h[i][j].recover())
                diag_h += dot(tp_mat.dwn_h[i][j], self.upc_h[i][j])
                diag_h += dot(tp_mat.upc_h[i][j + 1], self.dwnc_h[i][j + 1])                
                diff = np.max(abs(diag_h - np.eye(diag_h.shape[0])))
                if diff > tol:
                    print 'warning, diag_diff', i, j, diff
            j = self.lead_nlayer[i] - 2
            diag_h = dot(tp_mat.diag_h[i][j].recover(),
                                                  self.diag_h[i][j].recover())
            diag_h += dot(tp_mat.dwnc_h[i][j], self.upc_h[i][j])
            diff = np.max(abs(diag_h - np.eye(diag_h.shape[0])))
            if diff > tol:
                print 'warning, diag_diff', i, j, diff            
                                                
    def inv_eq(self):
        q_mat = []
        for i in range(self.lead_num):
            q_mat.append([])
            nll = self.lead_nlayer[i]
            for j in range(nll - 1):
                q_mat[i].append([])
            end = nll - 2
            q_mat[i][end] =  self.diag_h[i][end].inv()
          
            for j in range(end - 1, -1, -1):
                self.diag_h[i][j].reset_minus(self.dotdot(
                                                    self.upc_h[i][j + 1],
                                                         q_mat[i][j + 1],
                                            self.dwnc_h[i][j + 1]), full=True)
                q_mat[i][j] = self.diag_h[i][j].inv()
        h_mm = self.mol_h

        for i in range(self.lead_num):
            h_mm.reset_minus(self.dotdot(self.upc_h[i][0], q_mat[i][0],
                                                self.dwnc_h[i][0]), full=True)
        inv_h_mm = h_mm.inv()
        h_mm.reset(inv_h_mm)
        
        for i in range(self.lead_num):
            tmp_dc = self.dwnc_h[i][0].copy()
            self.dwnc_h[i][0] = -self.dotdot(q_mat[i][0], tmp_dc, inv_h_mm)
            self.upc_h[i][0] = -self.dotdot(inv_h_mm, self.upc_h[i][0],
                                                                    q_mat[i][0])
            dim = len(self.ll_index[i][1])
            self.diag_h[i][0].reset(dot(q_mat[i][0], np.eye(dim) -
                                                  dot(tmp_dc, self.upc_h[i][0])))
            for j in range(1, self.lead_nlayer[i] - 1):
                tmp_dc = self.dwnc_h[i][j].copy()
                self.dwnc_h[i][j] = -self.dotdot(q_mat[i][j], tmp_dc,
                                                self.diag_h[i][j - 1].recover())
                self.upc_h[i][j] = -self.dotdot(self.diag_h[i][j - 1].recover(),
                                                    self.upc_h[i][j],
                                                     q_mat[i][j])
                dim = len(self.ll_index[i][j + 1])
                self.diag_h[i][j].reset(dot(q_mat[i][j], np.eye(dim) -
                                           dot(tmp_dc, self.upc_h[i][j])))

    def inv_ne(self):
        q_mat = []
        qi_mat = []
        inv_mat = []
        #structure of inv_mat inv_cols_1, inv_cols_2, ..., inv_cols_n (n:lead_num)
        #structure of inv_cols_i   inv_cols_l1, inv_cols_l2,..., inv_cols_ln, inv_cols_mm(matrix)
        #structure of inv_cols_li  inv_cols_ll1, inv_cols_ll2,...,inv_cols_ll3
        for i in range(self.lead_num):
            q_mat.append([])
            qi_mat.append([])
            inv_mat.append([])
            
            nll = self.lead_nlayer[i]
            for j in range(nll - 1):
                q_mat[i].append([])
                qi_mat[i].append([])
                
            for j in range(self.lead_num):
                inv_mat[i].append([])
                nll_j = self.lead_nlayer[j]
                for k in range(nll_j - 1):
                    inv_mat[i][j].append([])
            inv_mat[i].append([])                
            
            end = nll - 2
            q_mat[i][end] =  self.diag_h[i][end].inv(keep_data=True)
            for j in range(end - 1, -1, -1):
                tmp_diag_h = copy.deepcopy(self.diag_h[i][j])
                tmp_diag_h.reset_minus(self.dotdot(self.upc_h[i][j + 1],
                                                     q_mat[i][j + 1],
                                                  self.dwnc_h[i][j + 1]),
                                        full=True)
                q_mat[i][j] = tmp_diag_h.inv()
        # above get all the q matrix, then if want to solve the cols
        # cooresponding to the lead i, the q_mat[i] will not be used

        #q_mm = self.mol_h.recover()
        q_mm = copy.deepcopy(self.mol_h)
        for i in range(self.lead_num):
            #q_mm -= dot(dot(self.upc_h[i][0], q_mat[i][0]),
            #                                 self.dwnc_h[i][0])
            q_mm.reset_minus(self.dotdot(self.upc_h[i][0],
                                  q_mat[i][0], self.dwnc_h[i][0]), full=True)
        
        for i in range(self.lead_num):
        # solve the corresponding cols to the lead i
            nll = self.lead_nlayer[i]
    
            #qi_mat[i][0] = q_mm + self.dotdot(self.upc_h[i][0],q_mat[i][0],
            #                                                self.dwnc_h[i][0])
            q_mm_tmp = copy.deepcopy(q_mm)
            q_mm_tmp.reset_plus(self.dotdot(self.upc_h[i][0],q_mat[i][0],
                                                self.dwnc_h[i][0]), full=True)
            
            #inv(qi_mat[i][0])
            qi_mat[i][0] = q_mm_tmp.inv()
            for j in range(1, nll - 1):
                tmp_diag_h = copy.deepcopy(self.diag_h[i][j - 1])
                tmp_diag_h.reset_minus(self.dotdot(self.dwnc_h[i][j -1],
                                                        qi_mat[i][j - 1],
                                                        self.upc_h[i][j - 1]),
                                                       full=True)
                qi_mat[i][j] = tmp_diag_h.inv()

            tmp_diag_h = copy.deepcopy(self.diag_h[i][nll - 2])
            tmp_diag_h.reset_minus(self.dotdot(self.dwnc_h[i][nll - 2],
                                                qi_mat[i][nll -2],
                                               self.upc_h[i][nll -2]),
                                                  full=True)
            inv_mat[i][i][nll - 2] = tmp_diag_h.inv()
            
            for j in range(nll - 3, -1, -1):
                inv_mat[i][i][j] = -self.dotdot(qi_mat[i][j + 1],
                                                  self.upc_h[i][j + 1],
                                                 inv_mat[i][i][j + 1])
            inv_mat[i][self.lead_num] = -self.dotdot(qi_mat[i][0],
                                                  self.upc_h[i][0],
                                                  inv_mat[i][i][0]) 
            
            for j in range(self.lead_num):
                if j != i:
                    nlj = self.lead_nlayer[j]
                    inv_mat[i][j][0] = -self.dotdot(q_mat[j][0], self.dwnc_h[j][0],
                                                inv_mat[i][self.lead_num])
                    for k in range(1, nlj - 1):
                        inv_mat[i][j][k] = -self.dotdot(q_mat[j][k], self.dwnc_h[j][k],
                                                inv_mat[i][j][k - 1])                         
        return inv_mat 
  
  
    def combine_inv_mat(self, inv_mat):
        nb = self.nb
        mat = np.zeros([nb, nb], complex)
        for i in range(self.lead_num):
            indr, indc = get_matrix_index(self.ll_index[i][0],
                                          self.ll_index[i][-1])
            mat[indr, indc] = inv_mat[i][self.lead_num]
            for j in range(self.lead_num):
                for k in range(1, self.lead_nlayer[j]):
                    indr, indc = get_matrix_index(self.ll_index[j][k],
                                                  self.ll_index[i][-1])
                    mat[indr, indc] = inv_mat[i][j][k - 1]
        return mat
  
    def dotdot(self, mat1, mat2, mat3):
        return dot(mat1, dot(mat2, mat3))
    
    def calculate_less_green(self, se_less):
        #se_less less selfenergy, structure  se_1, se_2, se_3,..., se_n
        #the lead sequence of se_less should be the same to self.ll_index
        inv_mat = self.inv_ne()
        self.mol_h.spar.fill(0.0)
        for i in range(self.lead_num):
            nll = self.lead_nlayer[i]
            for j in range(nll - 1):
                self.diag_h[i][j].spar.fill(0.0)
                self.upc_h[i][j].fill(0.0)
                self.dwnc_h[i][j].fill(0.0)
        
        for i in range(self.lead_num):
            # less selfenergy loop
            self.mol_h.reset_plus(self.dotdot(inv_mat[i][self.lead_num], se_less[i],
                                 inv_mat[i][self.lead_num].T.conj()), full=True)            
            for j in range(self.lead_num):
               # matrix operation loop    
                nlj = self.lead_nlayer[j]
                self.diag_h[j][0].reset_plus(self.dotdot(inv_mat[i][j][0], se_less[i],
                                                 inv_mat[i][j][0].T.conj()), full=True)            
            
                self.dwnc_h[j][0] += self.dotdot(inv_mat[i][j][0], se_less[i],
                                            inv_mat[i][self.lead_num].T.conj())
            
                self.upc_h[j][0] += self.dotdot(inv_mat[i][self.lead_num], se_less[i],
                                            inv_mat[i][j][0].T.conj())
            
                for k in range(1, nlj -1):
                    self.diag_h[j][k].reset_plus(self.dotdot(inv_mat[i][j][k - 1], se_less[i],
                                                 inv_mat[i][j][k - 1].T.conj()), full=True)
                    
                    self.dwnc_h[j][k] += self.dotdot(inv_mat[i][j][k], se_less[i],
                                                 inv_mat[i][j][k - 1].T.conj())
                        
                    self.upc_h[j][k] +=  self.dotdot(inv_mat[i][j][k - 1], se_less[i],
                                                    inv_mat[i][j][k].T.conj())

class CP_Sparse_HSD:
    def __init__(self, dtype, ns, npk, index=None):
        self.index = index
        self.dtype = dtype
        self.H = []
        self.S = []
        self.D = []
        self.ns = ns
        self.npk = npk
        self.s = 0
        self.pk = 0
        for s in range(ns):
            self.H.append([])
            self.D.append([])
            for k in range(npk):
                self.H[s].append([])
                self.D[s].append([])
        for k in range(npk):
            self.S.append([])        

    def reset(self, s, pk, mat, flag='S', init=False):
        assert mat.dtype == self.dtype
        if flag == 'S':
            spar = self.S
        elif flag == 'H':
            spar = self.H[s]
        elif flag == 'D':
            spar = self.D[s]
        if not init:
            spar[pk].reset(mat)
        elif self.index != None:
            spar[pk] = CP_Sparse_Matrix(self.dtype, mat, self.index)
        else:
            spar[pk] = CP_Sparse_Matrix(self.dtype, mat)
            self.index = spar[pk].index
     
class CP_Sparse_Matrix:
    def __init__(self, dtype, mat=None, index=None, flag=None, tol=1e-12):
        self.tol = tol
        self.index = index
        self.dtype = dtype
        self.flag = flag
        if mat != None:
            if self.index == None:
                self.initialize(mat)
            else:
                self.reset(mat)
        
    def initialize(self, mat):
        assert self.dtype == mat.dtype
        dim = mat.shape[-1]
        ud_array = np.empty([dim])
        dd_array = np.empty([dim])
        for i in range(dim):
            ud_array[i] = np.sum(abs(np.diag(mat, i)))
            dd_array[i] = np.sum(abs(np.diag(mat, -i)))
        spar_sum = 0
        mat_sum = np.sum(abs(mat))
        if np.sum(abs(ud_array)) >  np.sum(abs(dd_array)):
            self.flag = 'U'
            i = -1
            while abs(mat_sum - spar_sum) > self.tol * 10:
                i += 1
                while ud_array[i] < self.tol and  i < dim - 1:
                    i += 1
                self.index = (i, dim)
                ldab = dim - i
                self.spar = mat[:ldab, i:].copy()
                spar_sum = np.sum(abs(self.spar))
        else:
            self.flag = 'L'
            i = -1
            while abs(mat_sum - spar_sum) > self.tol * 10:
                i += 1
                while dd_array[i] < self.tol and  i < dim - 1:
                    i += 1
                self.index = (-i, dim)
                ldab = dim - i
                self.spar = mat[i:, :ldab].copy()
                spar_sum = np.sum(abs(self.spar))
        
    def reset(self, mat):
        assert mat.dtype == self.dtype and mat.shape[-1] == self.index[1]
        dim = mat.shape[-1]
        if self.index[0] > 0:
            ldab = dim - self.index[0]
            self.spar = mat[:ldab, self.index[0]:].copy()            
        else:
            ldab = dim + self.index[0]
            self.spar = mat[-self.index[0]:, :ldab].copy()               

    def recover(self, trans='n'):
        nb = self.index[1]
        mat = np.zeros([nb, nb], self.dtype)
        if self.index[0] > 0:
            ldab = nb - self.index[0]
            mat[:ldab, self.index[0]:] = self.spar
        else:
            ldab = nb + self.index[0]
            mat[-self.index[0]:, :ldab] = self.spar
        if trans == 'c':
            if self.dtype == float:
                mat = mat.T.copy()
            else:
                mat = mat.T.conj()
        return mat

class Se_Sparse_Matrix:
    def __init__(self, mat, tri_type, nn=None, tol=1e-12):
        # coupling sparse matrix A_ij!=0 if i>dim-nn and j>dim-nn (for right selfenergy)
        # or A_ij!=0 if i<nn and j<nn (for left selfenergy, dim is the shape of A)
        self.tri_type = tri_type
        self.tol = tol
        self.nb = mat.shape[-1]
        self.spar = []
        if nn == None:
            self.initialize(mat)
        else:
            self.reset(mat, nn)

    def initialize(self, mat):
        self.nn = 0
        nb = self.nb
        tol = self.tol
        if self.tri_type == 'L':
            while self.nn < nb and np.sum(abs(mat[self.nn])) > tol:
                self.nn += 1
            self.spar = mat[:self.nn, :self.nn].copy()
        else:
            while self.nn < nb and np.sum(abs(mat[nb - self.nn - 1])) > tol:
                self.nn += 1
            self.spar = mat[-self.nn:, -self.nn:].copy()                 
        
        diff = abs(np.sum(abs(mat)) - np.sum(abs(self.spar)))
        if diff > tol * 10:
            print 'Warning! Sparse Matrix Diff', diff
        
    def reset(self, mat, nn=None):
        if nn != None:
            self.nn = nn
        if self.tri_type == 'L':
            self.spar = mat[:self.nn, :self.nn].copy()
        else:
            self.spar = mat[-self.nn:, -self.nn:].copy()    
  
    def restore(self):
        mat = np.zeros([self.nb, self.nb], complex)
        if self.tri_type == 'L':
            mat[:self.nn, :self.nn] = self.spar
        else:
            mat[-self.nn:, -self.nn:] = self.spar
        return mat   

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
    b *= dim[d] / np.product(dim)
    return b
    
def aa2d(a, d=0):
    # array average in two dimensions
    b = np.sum(a, axis=d) / a.shape[d]
    return b

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
    
    tol = 1e-10
    position[d] = 2.0
    s_test = substract_pk(d, npk, ntk, kpts, sl_kmm, 's', position)
    
    matmax = np.max(abs(s_test))
    if matmax > tol:
        if txt != None:
            txt('Warning*: the principle layer should be lagger, \
                                                      matmax=%f' % matmax)
        else:
            print 'Warning*: the principle layer should be lagger, \
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
    ns = wfs.nspins
    kpts = wfs.ibzk_qc
    nq = len(kpts)
    nao = wfs.setups.nao
    d_skmm = np.empty([ns, nq, nao, nao], wfs.dtype)
    for kpt in wfs.kpt_u:
        wfs.calculate_density_matrix(kpt.f_n, kpt.C_nM, d_skmm[kpt.s, kpt.q])
    return d_skmm

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

