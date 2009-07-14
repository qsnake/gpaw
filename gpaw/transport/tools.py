from gpaw.utilities import unpack
from ase import Hartree
import pickle
import numpy as np
from gpaw.mpi import world, rank
from gpaw.utilities.blas import gemm
from gpaw.utilities.lapack import inverse_symmetric, inverse_general
from gpaw.utilities.timing import Timer
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
        
class Fb_Sparse_Matrix:
    # For matrix with the sparse property like A_ij != 0,  q<=j<=q+l (*)
    # q(i) denpends on i and now only consider the symmetric sparse matrix,
    # which means the non_zero elements' position is symmetric, not for
    # the element value.
    
    def __init__(self, mat, symm=False, index=None, tol=1e-16):
        self.tol = tol
        self.dtype = mat.dtype
        self.symm = symm
        if index == None:
            self.initialize(mat)
        else:
            self.reset(mat, index)
        self.timer = Timer()
        
    def initialize(self, mat):
        # index[0][0] = dim, index[0][1:] is the q in (*) for each line
        # index[1][0] = dim, index[1][1:] is the q + l in (*) for each line
        # index[2][-1] = nele, index[2][:-1] is the start pointer for each line
        # in the data array.
        dim = mat.shape[-1]  
        self.index = [[dim], [dim], []]
        self.spar = []
        cur_row = -1
        
        for i in range(dim):
            for j in range(dim):
                 if abs(mat[i, j]) > self.tol:
                    if not self.symm or (self.symm and i>=j):
                        self.spar.append(mat[i, j])
                    k = j
                    if i != cur_row:
                        self.index[0].append(k)
                        self.index[2].append(len(self.spar) - 1)
                        cur_row += 1
            self.index[1].append(k)
        self.index[2].append(len(self.spar))
        self.index = np.array(self.index)
        self.spar = np.array(self.spar)                    

    def reset(self, mat, index=None):
        if index != None:
            self.index = index
        dim = self.index[0][0]
        self.spar = []
        for i in range(dim):
            begin = self.index[0][i + 1]
            end = self.index[1][i + 1]
            for j in range(begin, end + 1):
                self.spar.append(mat[i, j])
        self.spar = np.array(self.spar)
    
    def inv(self, data=None, partly=False, row_b=None, row_e=None,
                                                      col_b=None, col_e=None):
        # get the sparse matrix inversion in the same positions of
        # non_zero elements, also can only get some rows:  [row_b:row_e] and
        # some columns: [col_b:col_e]
        # self.spar is lost after inversion
        
        assert self.dtype == complex
        dim = self.index[0][0]
        length = dim * (dim + 1) / 2

        if partly:
            assert row_b != None and row_e != None
            assert col_b != None and col_e != None
        else:
            row_b = 0
            col_b = 0
            row_e = dim - 1
            col_e = dim - 1

        inv_mat = np.zeros([dim, dim], complex)
           
        if data == None:
            data = self.spar

        #a0 = self.spar2full(self.spar, self.index)
        if self.symm:
            #if not _gpaw.csspar_ll(self.spar, self.index):
            #    raise RuntimeError('lu decompostion for sparse matrix fails')
            #for i in range(dim):
            #    inv_mat[i][i] = 1.0 
            #    if not _gpaw.cspar_lx(self.spar, self.index, inv_mat[i],
            #                                              dim, row_b, row_e):
            #        raise RuntimeError('row recus for sparse matrix fails')
            #    if not _gpaw.cspar_uy(self.spar, self.index, inv_mat[i],
            #                                               dim, col_b, col_e):
            #        raise RuntimeError('col recus for sparse matrix fails')
             _gpaw.cspar_inv(self.spar, self.index, inv_mat, 'S', dim, 0, dim-1)   
        else:
            #print 'debug0'
            #print self.spar, self.index
            #if not _gpaw.cgspar_lu(self.spar, self.index):
            #    print 'debug1'
            #    raise RuntimeError('lu decompostion for sparse matrix fails')
            #print 'debug2'
            #inv_u, index_u = self.split_u(self.spar)
            #inv_l, index_l = self.split_l(self.spar)
            #l0 = self.spar2full(inv_l, index_l)            
            #u0 = self.spar2full(inv_u, index_u)            
            #print np.max(abs(np.dot(l0, u0) - a0)), 'hahsd'
            #seq = np.array([0, 1,3,2,4,5])
            #for i in range(dim):
            #    inv_mat[i][i] = 1.0 
            #    if not _gpaw.cspar_lx(inv_u[seq], index_l, inv_mat[i],
            #                                              dim, row_b, row_e):
            #        raise RuntimeError('row recus for sparse matrix fails')
            #    if not _gpaw.cspar_uy(inv_l, index_l, inv_mat[i],
            #                                               dim, col_b, col_e):
            #        raise RuntimeError('col recus for sparse matrix fails')
             _gpaw.cspar_inv(self.spar, self.index, inv_mat, 'G', dim, 0, dim-1)              
            
        return inv_mat  

    def split_u(self, data):
        dim = self.index[0][0]
        index = self.index.copy()
        data_u = []
        for i in range(dim):
            index[2][i] = len(data_u)
            data_u.append(1.0)
            for j in range(i + 1, index[1][i + 1] + 1):
                data_u.append(data[self.index[2][i] + j - index[0][i + 1]])
        index[2][-1] = len(data_u)
        return np.array(data_u), np.array(index)
    
    def split_l(self, data):
        dim = self.index[0][0]
        index = self.index.copy()
        data_l = []
        for i in range(dim):
            index[2][i] = len(data_l)
            for j in range(index[0][i + 1], i + 1):
                data_l.append(data[self.index[2][i] + j - index[0][i + 1]])
        index[2][-1] = len(data_l)
        return np.array(data_l), np.array(index)
    
    def spar2full(self, data, index):
        dim = index[0][0]
        a = np.zeros((dim, dim), complex)
        for i in range(dim):
            n = 0
            for j in range(index[0][i + 1], index[1][i + 1] + 1):
                if not self.symm or (self.symm and j <= i):
                    a[i, j] = data[index[2][i] + n]
                    n += 1
        return a
   
class Banded_Sparse_Matrix:
    def __init__(self, mat, band_index=None, tol=1e-12):
        self.dtype = mat.dtype
        self.tol = tol
        if band_index == None:
            self.initialize(mat)
        else:
            self.reset(mat)
        
    def initialize(self, mat):
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
        self.spar = np.zeros([2 * kl + ku + 1, dim], complex)
        
        for i in range(kl, kl + ku + 1):
            ud = kl + ku - i
            self.spar[i][ud:] = np.diag(mat, ud)
        
        for i in range(kl + ku + 1, 2 * kl + ku + 1):
            ud = kl + ku - i
            self.spar[i][:ud] = np.diag(mat, ud)
    
    def reset(self, mat):
        kl, ku = self.band_index
        assert self.dtype == complex
        for i in range(kl + 1, kl + ku + 2):
            ud = kl + ku + 1 - i
            self.spar[i][ud:] = np.diag(mat, ud)
        for i in range(kl + ku + 2, 2 * kl + ku + 1):
            ud = kl + ku + 1 - i
            self.spar[i][:ud] = np.diag(mat, ud)        

    def inv(self):
        kl, ku = self.band_index
        dim = self.spar.shape[1]
        inv_mat = np.eye(dim, dtype=complex)
        #inv_mat=np.zeros([dim],complex)
        #inv_mat[0]=1.0
        spar = self.spar.copy()
        ldab = 2*kl + ku + 1
        info = _gpaw.linear_solve_band(self.spar, inv_mat, kl, ku, dim, ldab, dim, dim)
        return inv_mat
       
class Tp_Sparse_Matrix:
    def __init__(self, mat, ll_index):
    # ll_index : lead_layer_index
    # matrix stored here will be changed to inversion
        self.dtype = mat.dtype
        self.lead_num = len(ll_index)
        self.ll_index = ll_index
        self.initialize(mat)
        
    def initialize(self, mat):
    # diag_h : diagonal lead_hamiltonian
    # upc_h : superdiagonal lead hamiltonian
    # dwnc_h : subdiagonal lead hamiltonian 
        self.diag_h = []
        self.upc_h = []
        self.dwnc_h = []
        self.lead_nlayer = []
       
        self.mol_index = self.ll_index[0][0]
        ind = get_matrix_index(self.mol_index)
        self.mol_h = mat[ind.T, ind]

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
                ind = get_matrix_index(self.ll_index[i][j])
                ind1 = get_matrix_index(self.ll_index[i][j + 1])
                
                self.diag_h[i].append(mat[ind1.T, ind1])
                self.upc_h[i].append(mat[ind.T, ind1])
                self.dwnc_h[i].append(mat[ind1.T, ind])
                
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                
                self.length += 2 * len1 * len2 + len2 * len2
                self.nb += len2
    
    def recover(self):
        nb = self.nb
        mat = np.zeros([nb, nb], complex)
        ind = get_matrix_index(self.mol_index)
        mat[ind.T, ind] = self.mol_h
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                ind = get_matrix_index(self.ll_index[i][j])
                ind1 = get_matrix_index(self.ll_index[i][j + 1])
                mat[ind1.T, ind1] = self.diag_h[i][j]
                mat[ind.T, ind1] = self.upc_h[i][j]
                mat[ind1.T, ind] = self.dwnc_h[i][j]
        return mat        

    def storage(self):
        begin = 0 
        mem = np.empty([self.length], complex)
        nb = len(self.mol_index)
        mem[: nb ** 2] = np.resize(self.mol_h, [nb ** 2])
        begin += nb ** 2
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                mem[begin: begin + len2 ** 2] = np.resize(self.diag_h[i][j],
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
        self.mol_h = np.resize(mem[: nb ** 2], [nb, nb])
        begin += nb ** 2
        for i in range(self.lead_num):
            for j in range(self.lead_nlayer[i] - 1):
                len1 = len(self.ll_index[i][j])
                len2 = len(self.ll_index[i][j + 1])
                self.diag_h[i][j] = np.resize(mem[begin: begin + len2 ** 2],
                                                                [len2, len2])
                begin += len2 * len2
                self.upc_h[i][j] = np.resize(mem[begin: begin + len1 * len2],
                                             [len1, len2])
                begin += len1 * len2
                self.dwnc_h[i][j] = np.resize(mem[begin: begin + len1 * len2],
                                              [len2, len1])
                begin += len1 * len2
    
    def inv_eq(self):
        inv = inverse_symmetric
        q_mat = []
        for i in range(self.lead_num):
            q_mat.append([])
            nll = self.lead_nlayer[i]
            for j in range(nll - 1):
                q_mat[i].append([])
            end = nll - 2
            q_mat[i][end] =  self.diag_h[i][end]
            inv(q_mat[i][end])
            
            for j in range(end - 1, -1):
                q_mat[i][j] = self.diag_h[i][j] - np.dot(
                                                  np.dot(self.upc_h[i][j + 1],
                                                         q_mat[i][j + 1]),
                                                  self.dwnc_h[i][j + 1])
                inv(q_mat[i][j])
        h_mm = self.mol_h
        
        for i in range(self.lead_num):
            h_mm -= np.dot(np.dot(self.upc_h[i][0], q_mat[i][0]),
                                             self.dwnc_h[i][0])
        inv(h_mm)
        
        for i in range(self.lead_num):
            tmp_dc = self.dwnc_h[i][0].copy()
            #tmp_uc = self.upc_h[i][0].copy()
            self.dwnc_h[i][0] = -np.dot(np.dot(q_mat[i][0], tmp_dc), h_mm)
            self.upc_h[i][0] = -np.dot(np.dot(h_mm, self.upc_h[i][0]),
                                                           q_mat[i][0])
            dim = len(self.ll_index[i][1])
            self.diag_h[i][0] = np.dot(q_mat[i][0], np.eye(dim) -
                                            np.dot(tmp_dc, self.upc_h[i][0]))

            for j in range(1, self.lead_nlayer[i] - 1):
                tmp_dc = self.dwnc_h[i][j].copy()
                self.dwnc_h[i][j] = -np.dot(np.dot(q_mat[i][j], tmp_dc),
                                                       self.diag_h[i][j - 1])
                self.upc_h[i][j] = -np.dot(np.dot(self.diag_h[i][j - 1],
                                                    self.upc_h[i][j]),
                                                     q_mat[i][j])
                dim = len(self.ll_index[i][j + 1])
                self.diag_h[i][j] = np.dot(q_mat[i][j], np.eye(dim) -
                                           np.dot(tmp_dc, self.upc_h[i][j]))

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
                    inv_mat[i][k].append([])
                inv_mat[i].append([])                
            
            end = nll - 2
            q_mat[i][end] =  self.diag_h[i][end]
            inv(q_mat[i][end])
            for j in range(end - 1, -1):
                q_mat[i][j] = self.diag_h[i][j] - np.dot(
                                                  np.dot(self.upc_h[i][j + 1],
                                                     q_mat[i][j + 1]),
                                                  self.dwnc_h[i][j + 1])
                inv(q_mat[i][j])
        # above get all the q matrix, then if want to solve the cols
        # cooresponding to the lead i, the q_mat[i] will not be used

        q_mm = self.mol_h.copy()
        for i in range(self.lead_num):
            q_mm -= np.dot(np.dot(self.upc_h[i][0], q_mat[i][0]),
                                             self.dwnc_h[i][0])        
        
        for i in range(self.lead_num):
        # solve the corresponding cols to the lead i
            nll = self.lead_nlayer[i]
            qi_mat[i][0] = q_mm + np.dot(self.upc_h[i][0],
                                  np.dot(q_mat[i][0], self.dwnc_h[i][0]))
            inv(qi_mat[i][0])
            for j in range(1, nll - 1):
                qi_mat[i][j] = self.diag_h[i][j - 1] - np.dot(self.dwnc_h[i][j -1],
                                                        np.dot(qi_mat[i][j - 1],
                                                        self.upc_h[i][j - 1]))
                inv(qi_mat[i][j])
            
            
            inv_mat[i][i][nll - 2] = self.diag_h[i][nll - 2] - \
                                        np.dot(self.dwnc_h[i][nll - 2],
                                        np.dot(qi_mat[i][nll -2],
                                               self.upc_h[i][nll -2]))
            inv(inv_mat[i][i][nll - 2])
            
            for j in range(nll - 3, -1):
                inv_mat[i][i][j] = -np.dot(np.dot(qi_mat[i][j + 1],
                                                  self.upc_h[i][j + 1]),
                                            inv_mat[i][i][j + 1])
            inv_mat[i][self.lead_num] = -np.dot(np.dot(qi_mat[i][0],
                                                  self.upc_h[i][0]),
                                            inv_mat[i][i][0]) 
            
            for j in range(self.lead_num):
                if j != i:
                    nlj = self.lead_nlayer[j]
                    inv_mat[i][j][0] = -np.dot(np.dot(q_mat[j][0], self.dwnc_h[j][0]),
                                                inv_mat[i][self.lead_num])
                    for k in range(1, nlj - 1):
                        inv_mat[i][j][k] = -np.dot(np.dot(q_mat[j][k], self.dwnc_h[j][k]),
                                                inv_mat[i][j][k - 1])                         
        return inv_mat 
  
    def dotdot(self, mat1, mat2, mat3):
        return np.dot(mat1, np.dot(mat2, mat3))
    
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
            self.mol_h += self.dotdot(inv_mat[i][self.lead_num], se_less[i],
                                      inv_mat[i][self.lead_num].T.conj())            
            for j in range(self.lead_num):
               # matrix operation loop    
                nlj = self.lead_nlayer[j]
                self.diag_h[j][0] += self.dotdot(inv_mat[i][j][0], se_less[i],
                                                 inv_mat[i][j][0].T.conj())            
            
                self.dwnc_h[j][0] += self.dotdot(inv_mat[i][j][0], se_less[i],
                                            inv_mat[i][lead_num].T.conj())
            
                self.upc_h[j][0] += self.dotdot(inv_mat[i][lead_num], se_less[i],
                                            inv_mat[i][j][0].T.conj())
            
                for k in range(1, nlj -1):
                    self.diag_h[j][k] += self.dotdot(inv_mat[i][j][k - 1], se_less[i],
                                                 inv_mat[i][j][k - 1].T.conj())
                    
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

