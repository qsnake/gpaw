from gpaw import GPAW
from gpaw.transport.tools import k2r_hs, get_hs, dagger, dot, get_tri_type
from gpaw.transport.tools import Banded_Sparse_Matrix, get_matrix_index
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
        self.s_ii_spar = Banded_Sparse_Matrix(s_ii)
        index = self.s_ii_spar.banded_index
        self.h_ii_spar = Banded_sparse_Matrix(h_ii, index)
        
        self.s_ij = s_ij
        self.h_ij = h_ij
        
        #self.spar_init(h_ij, s_ij)
        
        #if self.tri_type == 'L':
        #    self.ind0 = get_matrix_index(np.arange(self.nn))  #the index of non-zero selfenergy elements
        #    self.ind1 = get_matrix_index(np.arange(self.nn, 0)) #the index of non-zero selfenergy elements
                                                           # in the other direction    
        #else:
        #    self.ind0 = get_matrix_index(np.arange(-self.nn, 0))
        #    self.ind1 = get_matrix_index(np.arange(self.nn))


    def reset(self, hs_ii, hs_ij):
        h_ii, s_ii = hs_ii
        h_ij, s_ij = hs_ij
        index = self.s_ii_spar.banded_index
        self.s_ii_spar.reset(s_ii)
        self.h_ii_spar.reset(h_ii)
        #if self.tri_type == 'L':
        #    self.s_ij_spar = s_ij[-self.nn:, :self.nn]
        #    self.h_ij_spar = h_ij[-self.nn:, :self.nn]
        #else:
        #    self.s_ij_spar = s_ij[:self.nn, -self.nn:]
        #    self.h_ij_spar = h_ij[:self.nn, -self.nn:]
  
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
        tau_ij = z * self.s_ij - self.h_ij
        ginv = self.get_sgfinv(energy)
        a_ij = dot(ginv, tau_ij)        
        sigma = dot(tau_ij.T.conj(), a_ij)
        return sigma
       
    def set_bias(self, bias):
        self.bias = bias
        
    def get_lambda(self, energy):
        sigma_mm = self(energy)
        return 1.j * (sigma_mm - dagger(sigma_mm))
    
    def get_sgfinv(self, energy):
        """The inverse of the retarded surface Green function"""
        z = energy - self.bias + self.eta * 1.0j
        
        g_spar = Banded_Sparse_Matrix(None, self.s_ii.spar.banded_index)
        g_spar.reset_from_others(self.s_ii_spar, self.h_ii_spar, z, -1.0)
        
        g_spar0 = copy.deepcopy(g_spar)
        
        #v_00 = z * self.s_ii- self.h_ii
        #v_11 = v_00.copy()
        v_10 = z * self.s_ij- self.h_ij
        #v_01 = z * dagger(self.s_ij_spar) - dagger(self.h_ij_spar)
        v_01 = dagger(v_10)
        
        delta = self.conv + 1
        while delta > self.conv:
            inv_v_11 = g_spar.inv()
            a = dot(inv_v_11, v_01)
            b = dot(inv_v_11, v_10)
            v_01_dot_b = dot(v_01, b)
            v_00.reset_minus(v_01_dot_b)
            v_11.reset_minus(dot(v_10, a))
            v_11.reset_minus(v_01_dot_b)
            v_01 = -dot(v_01, a)
            v_10 = -dot(v_10, b)
            delta = np.abs(v_01).max()
    