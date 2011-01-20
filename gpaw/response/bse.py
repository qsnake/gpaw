from time import time, ctime
import numpy as np
from math import pi
from ase.units import Hartree
from gpaw.mpi import world, size, rank
from gpaw.response.symmetrize import find_ibzkpt
from gpaw.response.base import BASECHI
from gpaw.response.parallel import parallel_partition


class BSE(BASECHI):
    """This class defines Belth-Selpether equations."""

    def __init__(self,
                 calc=None,
                 nbands=None,
                 w=None,
                 q=None,
                 ecut=10.,
                 eta=0.2,
                 ftol=1e-5,
                 txt=None,
                 optical_limit=False):

        BASECHI.__init__(self, calc, nbands, w, q, ecut,
                     eta, ftol, txt, optical_limit)

        self.epsilon_w = None

    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------------')
        self.printtxt('Belth Selpeter Equation calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        BASECHI.initialize(self)
        
        calc = self.calc

        # frequency points init
        self.dw = self.w_w[1] - self.w_w[0]
        assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
        assert self.w_w.max() == self.w_w[-1]

        self.dw /= Hartree
        self.w_w  /= Hartree
        self.wmax = self.w_w[-1] 
        self.Nw  = int(self.wmax / self.dw) + 1

        # find the pair index and initialized pair energy (e_i - e_j) and occupation(f_i-f_j)
        self.e_S = {}
        focc_s = {}
        self.Sindex_S3 = {}
        iS = 0
        kq_k = self.kq_k
        for k1 in range(self.nkpt):
            ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op_scc, self.ibzk_kc, self.bzk_kc[k1])
            ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op_scc, self.ibzk_kc, self.bzk_kc[kq_k[k1]])
            for n1 in range(self.nbands):
                for m1 in range(self.nbands):
                    focc = self.f_kn[ibzkpt1,n1] - self.f_kn[ibzkpt2,m1]
                    if np.abs(focc) > self.ftol:
                        self.e_S[iS] =self.e_kn[ibzkpt2,m1] - self.e_kn[ibzkpt1,n1]
                        focc_s[iS] = focc
                        self.Sindex_S3[iS] = (k1, n1, m1)
                        iS += 1
        self.nS = iS
        self.focc_S = np.zeros(self.nS)
        for iS in range(self.nS):
            self.focc_S[iS] = focc_s[iS]

        # parallel init
        self.Scomm = world
        self.nS, self.nS_local, self.nS_start, self.nS_end = parallel_partition(
                               self.nS, self.Scomm.rank, self.Scomm.size, reshape=False)
        self.print_bse()

        self.get_phi_aGp()

        # Coulomb kernel init
        self.kc_G = np.zeros(self.npw)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            self.kc_G[iG] = 1. / np.inner(qG, qG)
        if self.optical_limit:
            self.kc_G[0] = 0.
        self.printtxt('')
        
        return


    def calculate(self):

        calc = self.calc
        f_kn = self.f_kn
        e_kn = self.e_kn
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        focc_S = self.focc_S
        e_S = self.e_S

        # calculate kernel
        K_SS = np.zeros((self.nS, self.nS), dtype=complex)
        rhoG0_S = np.zeros((self.nS), dtype=complex)

        for iS in range(self.nS_start, self.nS_end):
            print 'calculating kernel', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            rho1_G = self.density_matrix_Gspace(n1,m1,k1)
            rhoG0_S[iS] = rho1_G[0]
            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                rho2_G = self.density_matrix_Gspace(n2,m2,k2)
                K_SS[iS, jS] = np.sum(rho1_G.conj() * rho2_G * self.kc_G)
        K_SS *= 4 * pi / self.vol
        self.Scomm.sum(K_SS)
        self.Scomm.sum(rhoG0_S)

        # get and solve hamiltonian
        H_SS = np.zeros_like(K_SS)
        for iS in range(self.nS):
            H_SS[iS,iS] = e_S[iS]
            for jS in range(self.nS):
                H_SS[iS,jS] += focc_S[iS] * K_SS[iS,jS]

        w_S, v_SS = np.linalg.eig(H_SS)

        # get overlap matrix
        tmp = np.zeros((self.nS, self.nS), dtype=complex)
        for iS in range(self.nS):
            for jS in range(self.nS):
                tmp[iS, jS] = (v_SS[:, iS].conj() * v_SS[:, jS]).sum()
        overlap_SS = np.linalg.inv(tmp)

        # get chi
        epsilon_w = np.zeros(self.Nw, dtype=complex)
        tmp_w = np.zeros(self.Nw, dtype=complex)
        for iS in range(self.nS_start, self.nS_end):
            tmp_iS = v_SS[:,iS] * rhoG0_S 
            for iw in range(self.Nw):
                tmp_w[iw] = 1. / (iw*self.dw - w_S[iS] + 1j * self.eta)
            print 'calculating epsilon', iS
            for jS in range(self.nS):
                tmp_jS = v_SS[:,jS] * rhoG0_S * focc_S
                tmp = np.outer(tmp_iS, tmp_jS.conj()).sum() * overlap_SS[iS, jS]
                epsilon_w += tmp * tmp_w
        self.Scomm.sum(epsilon_w)

        epsilon_w *=  - 4 * pi / np.inner(self.qq_v, self.qq_v) / self.vol
        epsilon_w += 1        

        return w_S, epsilon_w

    
    def print_bse(self):

        printtxt = self.printtxt

        printtxt('Number of frequency points   : %d' %(self.Nw) )
        printtxt('Number of pair orbitals      : %d' %(self.nS) )
        printtxt('Parallelization scheme:')
        printtxt('   Total cpus         : %d' %(world.size))
        printtxt('   pair orb parsize   : %d' %(self.Scomm.size))        
        
        return


    def get_dielectric_function(self, filename='df.dat'):

        if self.epsilon_w is None:
            self.initialize()
            w_S, epsilon_w = self.calculate()
            self.epsilon_w = epsilon_w

        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.real(epsilon_w[iw]), np.imag(epsilon_w[iw])
            f.close()

        # Wait for I/O to finish
        world.barrier()

