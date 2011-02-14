from time import time, ctime
import numpy as np
import pickle
from math import pi
from ase.units import Hartree
from ase.io import write
from gpaw.mpi import world, size, rank, serial_comm
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
        self.printtxt(ctime())

        BASECHI.initialize(self)
        
        calc = self.calc
        self.kd = kd = calc.wfs.kd

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
            ibzkpt1 = kd.kibz_k[k1]
            ibzkpt2 = kd.kibz_k[kq_k[k1]]
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
        # kcomm and wScomm is only to be used when wavefunctions r parallelly distributed.
        self.kcomm = world
        self.wScomm = serial_comm
        
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
        self.rhoG0_S = np.zeros((self.nS), dtype=complex)

        t0 = time()
        for iS in range(self.nS_start, self.nS_end):
            print 'calculating kernel', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            rho1_G = self.density_matrix(n1,m1,k1)
            self.rhoG0_S[iS] = rho1_G[0]
            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                rho2_G = self.density_matrix(n2,m2,k2)
                K_SS[iS, jS] = np.sum(rho1_G.conj() * rho2_G * self.kc_G)

            if iS == 0:
                dt = time() - t0
                totaltime = dt * self.nS_local
                self.printtxt('Finished pair orbital 0 in %f seconds, estimated %f seconds left.' %(dt, totaltime))
                
            if rank == 0 and self.nS_local // 5 > 0:            
                if iS > 0 and iS % (self.nS_local // 5) == 0:
                    dt =  time() - t0
                    self.printtxt('Finished pair orbital %d in %f seconds, estimated %f seconds left.  '%(k, dt, totaltime - dt) )
                    
        K_SS *= 4 * pi / self.vol
        self.Scomm.sum(K_SS)
        self.Scomm.sum(self.rhoG0_S)

        # get and solve hamiltonian
        H_SS = np.zeros_like(K_SS)
        for iS in range(self.nS):
            H_SS[iS,iS] = e_S[iS]
            for jS in range(self.nS):
                H_SS[iS,jS] += focc_S[iS] * K_SS[iS,jS]

        self.w_S, self.v_SS = np.linalg.eig(H_SS)
        
        return 

    
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
            self.calculate()

            w_S = self.w_S
            v_SS = self.v_SS
            rhoG0_S = self.rhoG0_S
            focc_S = self.focc_S
            
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

            self.epsilon_w = epsilon_w
    
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.real(epsilon_w[iw]), np.imag(epsilon_w[iw])
            f.close()
    
        # Wait for I/O to finish
        world.barrier()

        return

    def get_e_h_density(self, lamda=None, filename=None):

        if filename is not None:
            self.load(filename)
            self.initialize()
            
        gd = self.gd
        w_S = self.w_S
        v_SS = self.v_SS
        A_S = v_SS[:, lamda]
        kq_k = self.kq_k
        kd = self.kd

        # Electron density
        nte_R = gd.zeros()
        
        for iS in range(self.nS_start, self.nS_end):
            print 'electron density:', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            ibzkpt1 = kd.kibz_k[k1]
            psitold_g = self.get_wavefunction(ibzkpt1, n1)
            psit1_g = kd.transform_wave_function(psitold_g, k1)

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                if m1 == m2 and k1 == k2:
                    psitold_g = self.get_wavefunction(ibzkpt1, n2)
                    psit2_g = kd.transform_wave_function(psitold_g, k1)

                    nte_R += A_S[iS] * A_S[jS].conj() * psit1_g.conj() * psit2_g

        # Hole density
        nth_R = gd.zeros()
        
        for iS in range(self.nS_start, self.nS_end):
            print 'hole density:', iS
            k1, n1, m1 = self.Sindex_S3[iS]
            ibzkpt1 = kd.kibz_k[kq_k[k1]]
            psitold_g = self.get_wavefunction(ibzkpt1, m1)
            psit1_g = kd.transform_wave_function(psitold_g, kq_k[k1])

            for jS in range(self.nS):
                k2, n2, m2 = self.Sindex_S3[jS]
                if n1 == n2 and k1 == k2:
                    psitold_g = self.get_wavefunction(ibzkpt1, m2)
                    psit2_g = kd.transform_wave_function(psitold_g, kq_k[k1])

                    nth_R += A_S[iS] * A_S[jS].conj() * psit1_g * psit2_g.conj()
                    
        self.Scomm.sum(nte_R)
        self.Scomm.sum(nth_R)


        if rank == 0:
            write('rho_e.cube',self.calc.atoms, format='cube', data=nte_R)
            write('rho_h.cube',self.calc.atoms, format='cube', data=nth_R)
            
        world.barrier()
        
        return 

    def get_excitation_wavefunction(self, lamda=None,filename=None, re_c=None, rh_c=None):

        if filename is not None:
            self.load(filename)
            self.initialize()
            
        gd = self.gd
        w_S = self.w_S
        v_SS = self.v_SS
        A_S = v_SS[:, lamda]
        kq_k = self.kq_k
        kd = self.kd

        if re_c is not None:
            psith_R = gd.zeros(dtype=complex)
        elif rh_c is not None:
            psite_R = gd.zeros(dtype=complex)
        else:
            self.printtxt('No wavefunction output !')
            return
            
        for iS in range(self.nS_start, self.nS_end):
            print 'hole wavefunction', iS
            k, n, m = self.Sindex_S3[iS]
            ibzkpt1 = kd.kibz_k[k]
            ibzkpt2 = kd.kibz_k[kq_k[k]]

            psitold_g = self.get_wavefunction(ibzkpt1, n)
            psit1_g = kd.transform_wave_function(psitold_g, k)

            psitold_g = self.get_wavefunction(ibzkpt2, m)
            psit2_g = kd.transform_wave_function(psitold_g, kq_k[k])

            if re_c is not None:
                # given electron position, plot hole wavefunction
                psith_R += A_S[iS] * psit1_g[re_c].conj() * psit2_g
            elif rh_c is not None:
                # given hole position, plot electron wavefunction
                psite_R += A_S[iS] * psit1_g.conj() * psit2_g[rh_c]
            else:
                pass

        if re_c is not None:
            self.Scomm.sum(psith_R)
            if rank == 0:
                write('psit_h.cube',self.calc.atoms, format='cube', data=psith_R)
        elif rh_c is not None:
            self.Scomm.sum(psite_R)
            if rank == 0:
                write('psit_e.cube',self.calc.atoms, format='cube', data=psite_R)
        else:
            pass

        world.barrier()
            
        return
    

    def load(self, filename):

        data = pickle.load(open(filename))
        self.w_S  = data['w_S']
        self.v_SS = data['v_SS']

        self.printtxt('Read succesfully !')
        

    def save(self, filename):
        """Dump essential data"""

        data = {'w_S'  : self.w_S,
                'v_SS' : self.v_SS}
        
        if rank == 0:
            pickle.dump(data, open(filename, 'w'), -1)

        world.barrier()

