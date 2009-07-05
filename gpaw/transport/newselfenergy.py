from gpaw import GPAW
from gpaw.transport.tools import k2r_hs, get_hs, dagger, dot, get_tri_type
from gpaw.transport.tools import Fb_Sparse_Matrix
from gpaw.utilities.lapack import inverse_general, inverse_symmetric
from gpaw.lcao.tools import get_realspace_hs

class LeadSelfEnergy:
    conv = 1e-8 # Convergence criteria for surface Green function
    
    def __init__(self, hs_ii, hs_ij, hs_im, eta=1e-4):
        # use sparse matrix method to deal with the lead selfenergy
        # define two kinds of sparse matrix for lead
        # h_ii and s_ii, for the onsite matrix use A_ij !=0 if j in [q:q+l]
        # where q and l depends on i
        # h_ij and s_ij, for the lower triangular coupling matrix
        # use A_ij !=0 if i >= nb - nn and j < nn
        # for upper triangular coupling matrix
        # use A_ij !=0 if i < nn and j >= nb - nn
        
        self.initialize(hs_ii, hs_ij)
        self.nb = hs_dii.shape[-1]
        self.eta = eta
        self.energy = None
        self.bias = 0
   
    def initialize(self, hs_ii, hs_ij):
        #define the sparse matrix indices and store them
        h_ii, s_ii = hs_ii
        h_ij, s_ij = hs_ij
        
        self.dtype = h_ii.dtype
        self.tri_type = get_tri_type(h_ij)
        
        #initialize sparse matrix
        
        self.symm = self.dtype == float
        self.s_ii_spar = Fb_Sparse_Matrix(s_ii, self.symm)
        index = self.s_ii_spar.index
        self.h_ii_spar = Fb_sparse_Matrix(h_ii, self.symm, index)
        self.spar_init(h_ij, s_ij)

    def reset(self, hs_ii, hs_ij):
        h_ii, s_ii = hs_ii
        h_ij, s_ij = hs_ij
        index = self.s_ii_spar.index
        self.s_ii_spar = Fb_Sparse_Matrix(s_ii, self.symm, index)
        self.h_ii_spar = Fb_Sparse_Matrix(h_ii, self.symm, index)
        if self.tri_type == 'L':
            self.s_ij_spar = s_ij[-self.nn:, :self.nn]
            self.h_ij_spar = h_ij[-self.nn:, :self.nn]
        else:
            self.s_ij_spar = s_ij[:self.nn, -self.nn:]
            self.h_ij_spar = h_ij[:self.nn, -self.nn:]
  
    def spar_init(self, h_ij, s_ij, tol=1e-16):
        # determine the sparse property accorrding to overlap and apply to
        # hamiltonian matrix

        dim = s_ij.shape[-1]
        flag = 0
        self.s_ij_spar = []
        for i in range(dim):
            for j in range(i + 1):
                if self.tri_type == 'L':
                    if abs(s_ij[i, j]) > tol:
                        if flag == 0:
                            self.nn = dim - i
                            flag = 1
                        if flag == 1 and j < self.nn:
                            self.s_ij_spar.append(s_ij[i ,j])
                else:
                    if abs(s_ij[j, i]) > tol:
                        if flag == 0:
                            self.nn = dim - i
                            flag = 1
                        if flag == 1 and i < self.nn:
                            self.s_ij_spar.append(s_ij[j, i])
       
        self.s_ij_spar = np.array(self.s_ij_spar)
        self.s_ij_spar.shape = (self.nn, self.nn)

        assert abs(np.sum(abs(s_ij)) - np.sum(abs(self.s_ij_spar))) < tol

        if self.tri_type == 'L':                        
            self.h_ij_spar = h_ij[-self.nn:, :self.nn]
        else:
            self.s_ij_spar = self.s_ij_spar.T
            self.h_ij_spar = h_ij[:self.nn, -self.nn:]
 
    def __call__(self, energy):
        self.energy = energy
        z = energy - self.bias + self.eta * 1.j           
        tau_im = z * self.s_ij_spar - self.h_ij_spar


        ginv = self.get_sgfinv(energy)
        inv(ginv, gsub, pb, pe)
        
        gsub2 = np.take(gsub, )
        a_im = dot(gsub, tau_im)
        
        tau_mi = z * dagger(self.s_im) - dagger(self.h_im)
        self.sigma_mm[:] = dot(tau_mi, a_im)
        return self.sigma_mm
    
    
        # sigma = _gpaw.calculate_selfenergy(self.h_ii, self.h_ij, self.s_ii,
        #                                     self.s_ij, energy)
        raise NotImplementError
    
    def set_bias(self, bias):
        self.bias = bias
        
    def get_lambda(self, energy):
        sigma_mm = self(energy)
        return 1.j * (sigma_mm - dagger(sigma_mm))
    
    def get_sgfinv(self, energy):
        """The inverse of the retarded surface Green function"""
        if self.symm:
            inv = inverse_symmetric
        else:
            inv = inverse_general
        z = energy - self.bias + self.eta * 1.0j
        
        v_00 = z * dagger(self.s_ii) - dagger(self.h_ii)
        
        v_11 = v_00.copy()
        
        v_10 = z * self.s_ij - self.h_ij
        v_01 = z * dagger(self.s_ij) - dagger(self.h_ij)
        delta = self.conv + 1
        n = 0
        while delta > self.conv:
            inv_v_11 = np.copy(v_11)
            inv(inv_v_11)
            a = dot(inv_v_11, v_01)
            b = dot(inv_v_11, v_10)
            v_01_dot_b = dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -dot(v_01, a)
            v_10 = -dot(v_10, b)
            delta = np.abs(v_01).max()
            n += 1
        return v_00    
    