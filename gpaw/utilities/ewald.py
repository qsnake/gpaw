from gpaw.utilities import erf
from gpaw.utilities.tools import construct_reciprocal
import numpy as np

def erfc(x):
    """
    The complimentary error function.
    """
    return 1.-erf(x)

class Ewald:
    """
    Class for calculating Ewald summations.
    'cell' is the unit cell in the cartesian basis.
    'G' is a normalized Ewald parameter. Large G results in fast real-space
    convergence but slow convergence in reciprocal space,
    Ng and Nl are lists specifying the number or nearest neighbors in sums over
    the reciprocal lattice and the real space lattice respectively.
    """
    def __init__(self, cell, G=5, Ng=[9,9,9], Nl=[3,3,3]):
        self.cell = cell
        self.G = G / np.sqrt(max(np.sum(cell**2,axis=1))) #G is renormalized to the longest lattice vector
        self.Ng = Ng 
        self.Nl = Nl

    def get_volume(self):
        return np.linalg.det(self.cell)

    def get_wigner_seitz_radius(self, N):
        """
        Wigner-Seitz radius for N electrons in the unit cell.
        """
        vc = self.get_volume()
        return ( 3. * vc / (4. * np.pi * N))**(1./3.)

    def get_recip_basis(self):
        """
        Returns the reciprocal triad of the basic vectors.
        """
        cv = self.cell
        Vcell = np.dot(cv[0],np.cross(cv[1],cv[2]))
        K_v = np.zeros((3,3))
        K_v[0] = np.cross(cv[1],cv[2])/Vcell
        K_v[1] = np.cross(cv[2],cv[0])/Vcell
        K_v[2] = np.cross(cv[0],cv[1])/Vcell
        return K_v

    def get_sum_recip_i(self, eps=1e-10):
        """           
            -----  -x
        pi  \     e
        ---2 |   -------   , with x = g^2/(4 G^2)
        V G /       x
            -----
           g not 0
        """
        N = self.Ng
        E_g = 0.
        vcell = np.linalg.det(self.cell)
        a = np.pi / (vcell * self.G**2)
        Kv = self.get_recip_basis()
        for i in np.arange(-N[0],N[0]+1.):
            for j in np.arange(-N[1],N[1]+1):
                for k in np.arange(-N[2],N[2]+1):
                    g_v = 2 * np.pi * (i*Kv[0] + j*Kv[1] + k*Kv[2])
                    g2 = np.dot(g_v,g_v)
                    x = g2 / 4. / self.G**2
                    #print g_v, g2, x
                    if g2 > eps:## exclude g=0
                        E_g += a*np.exp(-x)/x
        return E_g

    def get_sum_real_i(self, eps=1e-10):
        """
            -----  
            \     erfc( G [l| )
             |   -------------- 
            /         |l|
            -----
           l not 0
        """
        N = self.Nl
        E_r = 0.
        for i in np.arange(-N[0],N[0]+1.):
            for j in np.arange(-N[1],N[1]+1):
                for k in np.arange(-N[2],N[2]+1):       
                    l_v = i*self.cell[0] + j*self.cell[1] + k*self.cell[2]
                    l = np.sqrt(np.dot(l_v,l_v))
                    if l > eps:## exclude l=0
                        E_r += erfc(self.G*l)/l
        return E_r

    def get_sum_recip_ij(self,r_v,eps=1e-10):
        """           
            -----  -x  i g.r
         pi \     e   e
        ---2 |   -------   , with x = g^2/(4 G^2)
        V G /       x
            -----
           g not 0
        """
        N = self.Ng
        E_g = 0.
        vcell = np.linalg.det(self.cell)
        a = np.pi / (vcell * self.G**2)
        Kv = self.get_recip_basis()
        for i in np.arange(-N[0],N[0]+1.):
            for j in np.arange(-N[1],N[1]+1):
                for k in np.arange(-N[2],N[2]+1):
                    g_v = 2 * np.pi * (i*Kv[0] + j*Kv[1] + k*Kv[2])
                    g2 = np.dot(g_v,g_v)
                    x = g2 / 4. / self.G**2
                    if g2 > eps: ## exclude g=0
                        E_g += a*np.exp(1j*np.dot(r_v,g_v))*np.exp(-x)/x
        return E_g.real   

    def get_sum_real_ij(self, r_v, eps=1e-5):
        """
            -----   
            \     erfc( G [l-r| )
             |   -------------- 
            /         |l-r|
            -----
           l not 0
           
        Note: Add the l=0 term with self.get_erfc.
        """
        N = self.Nl
        E_r = 0.
        for i in np.arange(-N[0],N[0]+1.):
            for j in np.arange(-N[1],N[1]+1):
                for k in np.arange(-N[2],N[2]+1):       
                    l_v = i*self.cell[0] + j*self.cell[1] + k*self.cell[2]
                    l = np.sqrt(np.dot(l_v,l_v))
                    lr_v = l_v-r_v
                    lr = np.sqrt(np.dot(lr_v,lr_v))
                    if l > eps: ## exclude l=0
                        E_r += erfc(self.G*lr)/lr
        return E_r

    def get_erfr_limit(self):
        """
          lim erf(r G) / r
          r->0
        """
        return 2.*self.G/np.sqrt(np.pi)

    def get_erfr(self, r_v, eps = 1e-14):
        """
           erf(r G) / r 
        """
        r = np.sqrt(np.dot(r_v,r_v))
        if r > eps:
            y = erf(r*self.G) / r
        else:
            y = 2.*self.G/np.sqrt(np.pi)
        return y
    
    def get_erfc(self,r_v):
        r = np.sqrt(np.dot(r_v,r_v))  
        return erfc(r*self.G)/r

    def get_hom_correction(self):
        """
        A contribution from the homogenous background in the Ewald summation.
        """
        vc = self.get_volume()
        return np.pi / (self.G**2 * vc)
    

    def get_madelung_constant(self, r_M, q_v):
        """
        Returns the 'Madelung' constant, assuming the g=0 terms cancel.
        It reflects the coulomb energy of the ion 'i' due to all other charges
        in the ionic crystal.
        
        ---- ----
        \    \    (+/-) 1
         |    |  ---------     
        /    /   |r_ij + l|
        ---- ----
          j   l
        
        where the sum over j runs over all ions in the unit cell,
        and l runs over all lattice vectors, unless j=i where the l=0 term is
        left out (this is the diverging self energy of ion i).
        The sum is identical to the r.h.s. of eq. (3.25) in Kittel (1996).
        r_M is a matrix with the ion basis in cartesian coordinates.
        q_v is a vector with the charges of the ions.
        The Madelung constant scales with inter-ionic distance and is usually
        referred to the nearest-neigbor distance (see Kittel 1996).
        """
        E0 = 0.
        r_M = r_M-r_M[0]
        E0 -= self.get_sum_recip_i()
        E0 -= self.get_sum_real_i()
        E0 += self.get_erfr_limit() 
        for i in range(1,len(q_v)):
            Et = 0.
            s = 1-int(q_v[0]==q_v[i])*2 # sign of interaction
            Et += s*self.get_sum_recip_ij(r_M[i])
            Et += s*self.get_sum_real_ij(r_M[i])
            Et += s*self.get_erfc(r_M[i])
            E0 += Et
        return E0
    
    def get_electrostatic_potential(self, r, r_B, q_B, excludefroml0=None):
        """
        Calculates the electrostatic potential at point r_i from point charges 
        at {r_B} in a lattice using the Ewald summation.
        Charge neutrality is obtained by adding the homogenous 
        density q_hom/V.
        
                      ---- ----'                             -
                      \    \         q_j            q_hom   /      1
        phi(r_i)  =    |    |   ---------------  +  -----   |dr ---------
                      /    /    |r_i - r_j + l|       V     |   |r - r_i|
                      ---- ----                             / 
                       j    l
        
        r_B : matrix with the lattice basis (in cartesian coordinates).
        q_B : point charges (in units of e).
        excludefroml0 : integer specifying if a point charge is not to be
        included in the central (l=0) unit cell. Used for Madelung constants.
        """
        E0 = 0.
        if excludefroml0 is None:
            excludefroml0 = np.zeros([len(q_B)], dtype=int)
        if excludefroml0 in range(len(q_B)):
            i = excludefroml0
            excludefroml0 = np.zeros([len(q_B)], dtype=int)
            excludefroml0[i] = 1
        assert(sum(excludefroml0) <= 1)
        for i, q in enumerate(q_B): #potential from point charges
            rprime = r-r_B[i]
            E0 += q * self.get_sum_real_ij(rprime)
            E0 += q * self.get_sum_recip_ij(rprime)
            if excludefroml0[i]: # if sum over l not 0
                E0 -= q * self.get_erfr(rprime)
            else: # if sum over all l
                E0 += q * self.get_erfc(rprime)
        q_hom = -sum(q_B)
        E0 += q_hom * self.get_hom_correction() # compensating hom. background
        return E0
