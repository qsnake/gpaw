from gpaw.transport.tools import dagger, dot
from gpaw.transport.tools import Banded_Sparse_Matrix
import copy
import numpy as np

class LeadSelfEnergy:
    conv = 1e-8 # Convergence criteria for surface Green function
    
    def __init__(self, hsd_ii, hsd_ij, eta=1e-4):
        self.hsd_ii = hsd_ii
        self.hsd_ij = hsd_ij
        self.eta = eta
        self.energy = None
        self.bias = 0
        self.s = 0
        self.pk = 0

    def __call__(self, energy):
        self.energy = energy
        z = energy - self.bias + self.eta * 1.j           
        tau_ij = z * self.hsd_ij.S[self.pk].recover() - \
                                     self.hsd_ij.H[self.s][self.pk].recover()
        tau_ji = z * dagger(self.hsd_ij.S[self.pk].recover()) - \
                             dagger(self.hsd_ij.H[self.s][self.pk].recover())
        ginv = self.get_sgfinv(energy)
        a_ij = dot(ginv, tau_ij)        
        return Banded_Sparse_Matrix(complex, dot(tau_ji, a_ij),
                                    self.hsd_ii.S[self.pk].band_index)
       
    def set_bias(self, bias):
        self.bias = bias
        
    def get_lambda(self, energy):
        sigma = self(energy)
        sigma_mm = sigma.recover()
        return 1.j * (sigma_mm - dagger(sigma_mm))
    
    def get_sgfinv(self, energy):
        """The inverse of the retarded surface Green function"""
        z = energy - self.bias + self.eta * 1.0j
        v_00 = Banded_Sparse_Matrix(complex,
                                     None,
                                     self.hsd_ii.S[self.pk].band_index)
        
        v_00.reset_from_others(self.hsd_ii.S[self.pk],
                                self.hsd_ii.H[self.s][self.pk],
                                z, -1.0)
        
        v_11 = copy.deepcopy(v_00)
        
        v_10 = z * self.hsd_ij.S[self.pk].recover()- \
                                     self.hsd_ij.H[self.s][self.pk].recover()
        v_01 = z * dagger(self.hsd_ij.S[self.pk].recover()) - \
                              dagger(self.hsd_ij.H[self.s][self.pk].recover())
        delta = self.conv + 1
        while delta > self.conv:
            inv_v_11 = v_11.inv(keep_data=True)
            a = dot(inv_v_11, v_01)
            b = dot(inv_v_11, v_10)
            v_01_dot_b = dot(v_01, b)
            v_00.reset_minus(v_01_dot_b, full=True)
            v_11.reset_minus(dot(v_10, a), full=True)
            v_11.reset_minus(v_01_dot_b, full=True)
            v_01 = -dot(v_01, a)
            v_10 = -dot(v_10, b)
            delta = np.abs(v_01).max()
        return v_00.inv()