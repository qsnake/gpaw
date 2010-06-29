import numpy as np

from ase.dft.kpoints import get_monkhorst_shape

class KPointDescriptor:
    """Class for keeping track of k-points."""

    def __init__(self, bzk_kc, ibzk_kc):
        """Init with k-point coordinates for full and irreducible BZ.

        Parameters
        ----------
        bzk_kc: ndarray
            K-points in the BZ.
        ibzk_kc: ndarray
            K-points in the irreducible part of the BZ.

        """

        self.bzk_kc = bzk_kc
        self.ibzk_kc = ibzk_kc
        self.nkpts = len(bzk_kc)

    def find_q_grid(self):
        """Find q-points as difference between k-points."""

        raise NotImplementedError
    
    def find_k_plus_q(self, q_c):
        """Find the index of k+q for all kpoints in the BZ.

        In case that k+q is outside the BZ, the k-point inside the BZ
        corresponding to k+q is given.
        
        Parameters
        ----------
        q_c: ndarray
            Scaled coordinates the q-vector in units of the reciprocal lattice
            vectors.

        """

        kplusq_k = []
        
        nkptxyz = get_monkhorst_shape(self.bzk_kc)
        
        dk = 1. / nkptxyz 
        kmax = (nkptxyz - 1) * dk / 2.
        N = np.zeros(3, dtype=int)

        for k in range(self.nkpts):
            
            kplusq_c = self.bzk_kc[k] + q_c
            
            for dim in range(3):
                if kplusq_c[dim] > 0.5:
                    kplusq_c[dim] -= 1.
                elif kplusq_c[dim] < -0.5:
                    kplusq_c[dim] += 1.
    
                N[dim] = int(np.round((kplusq_c[dim] + kmax[dim])/dk[dim]))
    
            kplusq_k.append(N[2] + N[1]*nkptxyz[2] + N[0]*nkptxyz[2]*nkptxyz[1])

            # Check the k+q vector index
            k_c = self.bzk_kc[kplusq_k[k]]

            assert abs(kplusq_c - k_c).sum() < 1e-8, 'k+q index not correct!'
    
        return kplusq_k


##     def find_ibzkpt(symrel, ibzk_kv, bzk_v):
##         """Given a certain kpoint, find its index in IBZ and related symmetry operations."""
##         find = False
##         ibzkpt = 0
##         iop = 0
##         timerev = False
    
##         for ioptmp, op in enumerate(symrel):
##             for i, ibzk in enumerate(ibzk_kv):
##                 diff_c = np.dot(ibzk, op.T) - bzk_v
##                 if (np.abs(diff_c - diff_c.round()) < 1e-8).all():
##                     ibzkpt = i
##                     iop = ioptmp
##                     find = True
##                     break
    
##                 diff_c = np.dot(ibzk, op.T) + bzk_v
##                 if (np.abs(diff_c - diff_c.round()) < 1e-8).all():
##                     ibzkpt = i
##                     iop = ioptmp
##                     find = True
##                     timerev = True
##                     break
                
##             if find == True:
##                 break
            
##         if find == False:        
##             print bzk_v
##             print ibzk_kv
##             raise ValueError('Cant find corresponding IBZ kpoint!')
    
##         return ibzkpt, iop, timerev
