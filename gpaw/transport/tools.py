from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as np
from gpaw.mpi import world, rank
from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import inverse_symmetric, inverse_general
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
        
class Banded_Sparse_Matrix:
    def __init__(self, mat=None, band_index=None, tol=1e-12):
        self.tol = tol
        if mat == None:
            self.band_index = banded_index
        else:
            if band_index == None:
                self.initialize(mat)
            else:
                self.reset(mat)
        
    def initialize(self, mat):
        self.dtype = mat.dtype
        dim = mat.shape[-1]
        ku = 0
        kl = 0
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
            
        self.band_index = (kl, ku)
        assert self.dtype == complex
        
        #self.spar = np.zeros([2 * kl + ku + 1, dim], complex)
        
        #for i in range(kl, kl + ku + 1):
        #    ud = kl + ku - i
        #    self.spar[i][ud:] = np.diag(mat, ud)
        
        #for i in range(kl + ku + 1, 2 * kl + ku + 1):
        #    ud = kl + ku - i
        #    self.spar[i][:ud] = np.diag(mat, ud)
    
        # storage in the tranpose, bacause column major order for zgbsv_ function
        self.spar = np.zeros([dim, 2 * kl + ku + 1], complex)
        
        for i in range(kl, kl + ku + 1):
            ud = kl + ku - i
            self.spar[ud:][i] = np.diag(mat, ud)
        for i in range(kl + ku + 1, 2 * kl + ku + 1):
            ud = kl + ku - i
            self.spar[:ud][i] + np.diag(mat, ud)
  
    def reset(self, mat):
        kl, ku = self.band_index
        assert self.dtype == complex
        #for i in range(kl + 1, kl + ku + 2):
        #    ud = kl + ku + 1 - i
        #    self.spar[i][ud:] = np.diag(mat, ud)
        #for i in range(kl + ku + 2, 2 * kl + ku + 1):
        #    ud = kl + ku + 1 - i
        #    self.spar[i][:ud] = np.diag(mat, ud)        

        for i in range(kl, kl + ku + 1):
            ud = kl + ku - i
            self.spar[ud:][i] = np.diag(mat, ud)
        for i in range(kl + ku + 1, 2 * kl + ku + 1):
            ud = kl + ku - i
            self.spar[:ud][i] = np.diag(mat, ud)
    
    def reset_from_others(self, bds_mm1, bds_mm2, c1, c2):
        self.spar = c1 * bds_mm1.spar + c2 * bds_mm2.spar 
            
    def reset_minus(self, mat):
        kl, ku = self.band_index
        assert self.dtype == complex
        for i in range(kl, kl + ku + 1):
            ud = kl + ku - i
            self.spar[ud:][i] -= np.diag(mat, ud)
        for i in range(kl + ku + 1, 2 * kl + ku + 1):
            ud = kl + ku - i
            self.spar[:ud][i] -= np.diag(mat, ud)
    
    def reset_plus(self, mat):
        kl, ku = self.band_index
        assert self.dtype == complex
        for i in range(kl, kl + ku + 1):
            ud = kl + ku - i
            self.spar[ud:][i] -= np.diag(mat, ud)
        for i in range(kl + ku + 1, 2 * kl + ku + 1):
            ud = kl + ku - i
            self.spar[:ud][i] -= np.diag(mat, ud)

    def inv(self, keep_data=False):
        kl, ku = self.band_index
        dim = self.spar.shape[1]
        inv_mat = np.eye(dim, dtype=complex)
        ldab = 2*kl + ku + 1
       
        if keep_data:
            source_mat = self.spar
        else:
            source_mat = self.spar.copy()
        info = _gpaw.linear_solve_band(source_mat, inv_mat,
                                                    kl, ku, dim, ldab, dim, dim)            
        return inv_mat
       
class Tp_Sparse_HSD:
    def __init__(self, ns, npk, ll_index):
        self.ll_index = ll_index
        self.H = []
        self.S = []
        self.D = []
        self.G = []
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
        self.G = Tp_Sparse_Matrix(self.ll_index)
    
    def reset(self, s, pk, mat, flag='S', init=False):
        if flag == 'S':
            spar = self.S
        elif flag == 'H':
            spar = self.H[s]
        elif flag == 'D':
            spar = self.D[s]
        if init:
            spar[pk] = Tp_Sparse_Matrix(self.ll_index, mat)
        else:
            spar[pk].reset(mat)
        
    def calculate_eq_green_function(self, zp, sigma):
        s, pk = self.s, self.pk
        self.G.reset_from_others(self.S[pk], self.H[s][pk], zp, -1)
        self.G.substract_sigma(sigma)
        self.G.inv_eq()
        return self.G.recover()

    def calculate_ne_green_function(self, zp, sigma, fermi_factors):
        s, pk = self.s, self.pk        
        self.G.reset_from_others(self.S[pk], self.H[s][pk], zp, -1)
        self.G.substract_sigma(sigma)
        gamma = []
        for ff, tgt in zip(fermi_factors, sigma):
            gamma.append(ff * 1.j * (tgt- tgt.T.conj()))
        self.G.calculate_less_green(gamma)
        return self.G.recover()     
       
class Tp_Sparse_Matrix:
    def __init__(self, ll_index, mat=None):
    # ll_index : lead_layer_index
    # matrix stored here will be changed to inversion

        self.lead_num = len(ll_index)
        self.ll_index = ll_index
        self.initialize()
        if mat != None:
            self.dtype = mat.dtype            
            self.reset(mat)
        
    def initialize(self):
    # diag_h : diagonal lead_hamiltonian
    # upc_h : superdiagonal lead hamiltonian
    # dwnc_h : subdiagonal lead hamiltonian 
        self.diag_h = []
        self.upc_h = []
        self.dwnc_h = []
        self.lead_nlayer = []
        self.mol_index = self.ll_index[0][0]
        self.nl = 1
        self.nb = len(self.mol_index)
        self.length = self.nb * self.nb 
        for i in range(self.lead_num):
            self.diag_h.append([])
            self.upc_h.append([])
            self.dwnc_h.append([])
            self.lead_nlayer.append(len(self.ll_index[i]))
            assert (self.ll_index[i][0] == self.mol_index).all()
            self.nl += self.lead_nlayer[i]        
            for j in range(self.lead_nlayer[i] - 1):
                self.diag_h[i].append([])
                self.upc_h[i].append([])
                self.dwnc_h[i].append([])
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                self.length += 2 * len1 * len2 + len2 * len2
                self.nb += len2
    
    def reset(self, mat):
        ind = get_matrix_index(self.mol_index)
        #self.mol_h = mat[ind.T, ind]
        self.mol_h.reset(mat[ind.T, ind])
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                ind = get_matrix_index(self.ll_index[i][j])
                ind1 = get_matrix_index(self.ll_index[i][j + 1])
                #self.diag_h[i][j] = mat[ind1.T, ind1]
                self.diag_h[i][j].reset(mat[ind1.T, ind1])
                self.upc_h[i][j] = mat[ind.T, ind1]
                self.dwnc_h[i][j] = mat[ind1.T, ind]
       
    def reset_from_others(self, tps_mm1, tps_mm2, c1, c2):
        #self.mol_h = c1 * tps_mm1.mol_h + c2 * tps_mm2.mol_h
        self.mol_h.spar = c1 * tps_mm1.mol_h.spar + c2 * tps_mm2.mol_h.spar
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                assert (tps_mm1.ll_index[i][j] == tps_mm2.ll_index[i][j]).all()
                #self.diag_h[i][j] = c1 * tps_mm1.diag_h[i][j] + \
                #                     c2 * tps_mm2.diag_h[i][j]
                self.diag_h[i][j].spar = c1 * tps_mm1.diag_h[i][j].spar + \
                                      c2 * tps_mm2.diag_h[i][j].spar
                self.upc_h[i][j] = c1 * tps_mm1.upc_h[i][j] + \
                                      c2 * tps_mm2.upc_h[i][j]
                self.dwnc_h[i][j] = c1 * tps_mm1.dwnc_h[i][j] + \
                                      c2 * tps_mm2.dwnc_h[i][j]
  
    def substract_sigma(self, sigma):
        for i in range(self.lead_num):
            self.diag_h[i][-1] -= sigma[i]
        
    def recover(self):
        nb = self.nb
        mat = np.zeros([nb, nb], complex)
        ind = get_matrix_index(self.mol_index)
        mat[ind.T, ind] = self.mol_h.recover()
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                ind = get_matrix_index(self.ll_index[i][j])
                ind1 = get_matrix_index(self.ll_index[i][j + 1])
                mat[ind1.T, ind1] = self.diag_h[i][j].recover()
                mat[ind.T, ind1] = self.upc_h[i][j]
                mat[ind1.T, ind] = self.dwnc_h[i][j]
        return mat        

    def storage(self):
        begin = 0 
        mem = np.empty([self.length], complex)
        nb = len(self.mol_index)
        mem[: nb ** 2] = np.resize(self.mol_h.recover(), [nb ** 2])
        begin += nb ** 2
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                mem[begin: begin + len2 ** 2] = np.resize(self.diag_h[i][j].recover(),
                                                                    [len2 ** 2])
                begin += len2 * len2
                mem[begin: begin + len1 * len2] = np.resize(self.upc_h[i][j],
                                                                  [len1 * len2])
                begin += len1 * len2
                mem[begin: begin + len1 * len2] = np.resize(
                                                         self.dwnc_h[i][j + 1],
                                                         [len1 * len2])
                begin += len1 * len2
        return mem                                                   

    def read(self, mem):
        begin = 0 
        nb = len(self.mol_index)
        self.mol_h.reset(np.resize(mem[: nb ** 2], [nb, nb]))
        begin += nb ** 2
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                self.diag_h[i][j].reset(np.resize(mem[begin: begin + len2 ** 2],
                                                                [len2, len2]))
                begin += len2 * len2
                self.upc_h[i][j] = np.resize(mem[begin: begin + len1 * len2],
                                             [len1, len2])
                begin += len1 * len2
                self.dwnc_h[i][j] = np.resize(mem[begin: begin + len1 * len2],
                                              [len2, len1])
                begin += len1 * len2
    
    def inv_eq(self):
        inv = inverse_general
        q_mat = []
        for i in range(self.lead_num):
            q_mat.append([])
            nll = self.lead_nlayer[i]
            for j in range(nll - 1):
                q_mat[i].append([])
            end = nll - 2
            q_mat[i][end] =  self.diag_h[i][end].inv()
            #inv(q_mat[i][end])
            
            for j in range(end - 1, -1, -1):
                self.diag_h[i][j].reset_minus(self.dotdot(
                                                    self.upc_h[i][j + 1],
                                                         q_mat[i][j + 1],
                                                  self.dwnc_h[i][j + 1]))
                q_mat[i][j] = self.diag_h[i][j].inv()
                #inv(q_mat[i][j])
        h_mm = self.mol_h
        
        for i in range(self.lead_num):
            h_mm.reset_minus(self.dotdot(self.upc_h[i][0], q_mat[i][0],
                                                             self.dwnc_h[i][0]))
        inv_hmm = h_mm.inv()
        h_mm.reset(inv_hmm)
        
        #inv(h_mm)
        
        for i in range(self.lead_num):
            tmp_dc = self.dwnc_h[i][0].copy()
            #tmp_uc = self.upc_h[i][0].copy()
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
        inv = inverse_general
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
            #q_mat[i][end] =  self.diag_h[i][end].copy()
            q_mat[i][end] =  self.diag_h[i][end].inv(keep_data=True)
            #inv(q_mat[i][end])
            for j in range(end - 1, -1, -1):
                tmp_diag_h = copy.deepcopy(self.diag_h[i][j])
                tmp_diag_h.reset_minus(self.dotdot(self.upc_h[i][j + 1],
                                                     q_mat[i][j + 1],
                                                  self.dwnc_h[i][j + 1]))
                #q_mat[i][j] = self.diag_h[i][j] - dot(
                #                                  dot(self.upc_h[i][j + 1],
                #                                     q_mat[i][j + 1]),
                #                                  self.dwnc_h[i][j + 1])
                #inv(q_mat[i][j])
                q_mat[i][j] = tmp_diag_h.inv()
        # above get all the q matrix, then if want to solve the cols
        # cooresponding to the lead i, the q_mat[i] will not be used

        q_mm = self.mol_h.recover()
        for i in range(self.lead_num):
            q_mm -= dot(dot(self.upc_h[i][0], q_mat[i][0]),
                                             self.dwnc_h[i][0])        
        
        for i in range(self.lead_num):
        # solve the corresponding cols to the lead i
            nll = self.lead_nlayer[i]
            qi_mat[i][0] = q_mm + self.dotdot(self.upc_h[i][0],q_mat[i][0],
                                                            self.dwnc_h[i][0])
            inv(qi_mat[i][0])
            for j in range(1, nll - 1):
                tmp_diag_h = copy.deepcopy(self.diag_h[i][j - 1])
                tmp_diag_h.reset_minus(self.dotdot(self.dwnc_h[i][j -1],
                                                        qi_mat[i][j - 1],
                                                        self.upc_h[i][j - 1]))
                qi_mat[i][j] = tmp_diag_h.inv()
                
                #qi_mat[i][j] = self.diag_h[i][j - 1] - dot(self.dwnc_h[i][j -1],
                #                                        dot(qi_mat[i][j - 1],
                #                                        self.upc_h[i][j - 1]))
                #inv(qi_mat[i][j])
            
            
            #inv_mat[i][i][nll - 2] = self.diag_h[i][nll - 2] - \
            #                            dot(self.dwnc_h[i][nll - 2],
            #                            dot(qi_mat[i][nll -2],
            #                                   self.upc_h[i][nll -2]))
            #inv(inv_mat[i][i][nll - 2])
            tmp_diag_h = copy.deepcopy(self.diag_h[i][nll - 2])
            tmp_diag_h.reset_minus(self.dotdot(self.dwnc_h[i][nll - 2],
                                                qi_mat[i][nll -2],
                                               self.upc_h[i][nll -2]))
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
            ind = get_matrix_index(self.ll_index[i][-1])
            ind1 = get_matrix_index(self.ll_index[i][0])
            mat[ind1.T, ind] = inv_mat[i][self.lead_num]
            for j in range(self.lead_num):
                for k in range(1, self.lead_nlayer[j]):
                    ind1 = get_matrix_index(self.ll_index[j][k])
                    mat[ind1.T, ind] = inv_mat[i][j][k - 1]
        return mat
                    
  
    def dotdot(self, mat1, mat2, mat3):
        return dot(mat1, dot(mat2, mat3))
    
    def calculate_less_green(self, se_less):
        #se_less less selfenergy, structure  se_1, se_2, se_3,..., se_n
        #the lead sequence of se_less should be the same to self.ll_index
        inv_mat = self.inv_ne()
        self.mol_h.fill(0.0)
        for i in range(self.lead_num):
            nll = self.lead_nlayer[i]
            for j in range(nll - 1):
                self.diag_h[i][j].fill(0.0)
                self.upc_h[i][j].fill(0.0)
                self.dwnc_h[i][j].fill(0.0)
        
        for i in range(self.lead_num):
            # less selfenergy loop
            self.mol_h.reset_plus(self.dotdot(inv_mat[i][self.lead_num], se_less[i],
                                      inv_mat[i][self.lead_num].T.conj()))            
            for j in range(self.lead_num):
               # matrix operation loop    
                nlj = self.lead_nlayer[j]
                self.diag_h[j][0].reset_plus(self.dotdot(inv_mat[i][j][0], se_less[i],
                                                 inv_mat[i][j][0].T.conj()))            
            
                self.dwnc_h[j][0] += self.dotdot(inv_mat[i][j][0], se_less[i],
                                            inv_mat[i][self.lead_num].T.conj())
            
                self.upc_h[j][0] += self.dotdot(inv_mat[i][self.lead_num], se_less[i],
                                            inv_mat[i][j][0].T.conj())
            
                for k in range(1, nlj -1):
                    self.diag_h[j][k].reset_plus(self.dotdot(inv_mat[i][j][k - 1], se_less[i],
                                                 inv_mat[i][j][k - 1].T.conj()))
                    
                    self.dwnc_h[j][k] += self.dotdot(inv_mat[i][j][k], se_less[i],
                                                 inv_mat[i][j][k - 1].T.conj())
                        
                    self.upc_h[j][k] +=  self.dotdot(inv_mat[i][j][k - 1], se_less[i],
                                                    inv_mat[i][j][k].T.conj())
           
class CP_Sparse_Matrix:
    def __init__(self, mat, tri_type, nn=None, tol=1e-16):
        # coupling sparse matrix A_ij!=0 if i>dim -nn and j>nn (for lower triangle
        # matrix) or A_ij!=0 if i<nn and j>dim-nn (for upper triangle matrix,
        #dim is the shape of A)

        self.tri_type = tri_type
        self.spar = []
        if nn == None:
            self.initialize(mat)
        else:
            self.reset(mat, nn)
        
    def initialize(self, mat):
        dim = mat.shape[-1]
        flag = 0        
        for i in range(dim):
            for j in range(i + 1):
                if stri_type == 'L':
                    if abs(mat[i, j]) > tol:
                        if flag == 0:
                            self.nn = dim - i
                            flag = 1
                        if flag == 1 and j <= self.nn:
                            self.spar.append(mat[i ,j])
                else:
                    if abs(mat[j, i]) > tol:
                        if flag == 0:
                            self.nn = dim - i
                            flag = 1
                        if flag == 1 and i <= self.nn:
                            self.spar.append(mat[j, i])
       
        self.spar = np.array(self.spar)
        self.spar.shape = (self.nn, self.nn)

        assert abs(np.sum(abs(mat)) - np.sum(abs(self.spar))) < tol
        
    def reset(self, mat, nn=None):
        if nn != None:
            self.nn = nn
        if self.tri_type == 'L':
            self.spar = mat[-self.nn:, :self.nn]
        else:
            self.spar = mat[:self.nn, -self.nn:]

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

def get_matrix_index(ind):
    dim = len(ind)
    return np.resize(ind, (dim, dim))
    
def aa1d(self, a, d=2):
    # array average in one dimension
    dim = a.shape
    b = [np.sum(np.take(a, i, axis=d)) for i in range(dim[d])]
    b *= dim[d] / np.product(dim)
    return b
    
def aa2d(self, a, d=0):
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

