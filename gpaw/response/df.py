import numpy as np
from math import sqrt, pi
import pickle
from ase.units import Hartree, Bohr
from gpaw.mpi import rank
from gpaw.response.math_func import delta_function
from gpaw.response.chi import CHI

class DF(CHI):
    """This class defines dielectric function related physical quantities."""

    def __init__(self,
                 calc=None,
                 nbands=None,
                 w=None,
                 q=None,
                 ecut=10.,
                 eta=0.2,
                 ftol=1e-7,
                 txt=None,
                 hilbert_trans=True,
                 optical_limit=False,
                 kcommsize=None):

        CHI.__init__(self, calc, nbands, w, q, ecut,
                     eta, ftol, txt, hilbert_trans, optical_limit, kcommsize)

        self.df1_w = None
        self.df2_w = None


    def get_RPA_dielectric_matrix(self):

	if self.chi0_wGG is None:
            self.initialize()
            self.calculate()
        else:
            pass # read from file and re-initializing .... need to be implemented
                       
        tmp_GG = np.eye(self.npw, self.npw)
        dm_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype = complex)

        for iw in range(self.Nw_local):
            for iG in range(self.npw):
                qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
                dm_wGG[iw,iG] = tmp_GG[iG] - 4 * pi / np.dot(qG, qG) * self.chi0_wGG[iw,iG]

        if self.nspins == 2:
            nibzkpt = self.ibzk_kc.shape[0]
            kweight_k = self.calc.get_k_point_weights()
            self.e_kn = np.array([self.calc.get_eigenvalues(kpt=k, spin=1)
                                  for k in range(nibzkpt)]) / Hartree
            self.f_kn = np.array([self.calc.get_occupation_numbers(kpt=k, spin=1) /
                                  kweight_k[k]
                                  for k in range(nibzkpt)]) / self.nkpt
            self.calculate(spin=1)

            for iw in range(self.Nw_local):
                for iG in range(self.npw):
                    qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
                    dm_wGG[iw,iG] -=  4 * pi / np.dot(qG, qG) * self.chi0_wGG[iw,iG]
        
        return dm_wGG


    def get_chi(self, xc='RPA'):
        """Solve Dyson's equation."""

	if self.chi0_wGG is None:
            self.initialize()
            self.calculate()
        else:
            pass # read from file and re-initializing .... need to be implemented

        kernel_GG = np.zeros((self.npw, self.npw), dtype=complex)
        chi_wGG = np.zeros_like(self.chi0_wGG)

        # Coulomb kernel
        for iG in range(self.npw):
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            kernel_GG[iG,iG] = 4 * pi / np.dot(qG, qG)
            
        if xc == 'ALDA':
            kernel_GG += self.Kxc_GG

        for iw in range(self.Nw_local):
            tmp_GG = np.eye(self.npw, self.npw) - np.dot(self.chi0_wGG[iw], kernel_GG)
            chi_wGG[iw] = np.dot(np.linalg.inv(tmp_GG) , self.chi0_wGG[iw])

        return chi_wGG


    def get_dielectric_function(self):
        """Calculate the dielectric function. Returns df1_w and df2_w.

        Parameters:

        df1_w: ndarray
            Dielectric function without local field correction.
        df2_w: ndarray
            Dielectric function with local field correction.
        """

        if self.df1_w is None:
            dm_wGG = self.get_RPA_dielectric_matrix()

            Nw_local = dm_wGG.shape[0]
            dfNLF_w = np.zeros(Nw_local, dtype = complex)
            dfLFC_w = np.zeros(Nw_local, dtype = complex)
            df1_w = np.zeros(self.Nw, dtype = complex)
            df2_w = np.zeros(self.Nw, dtype = complex)

            for iw in range(Nw_local):
                tmp_GG = dm_wGG[iw]
                dfLFC_w[iw] = 1. / np.linalg.inv(tmp_GG)[0, 0]
                dfNLF_w[iw] = tmp_GG[0, 0]

            self.wcomm.all_gather(dfNLF_w, df1_w)
            self.wcomm.all_gather(dfLFC_w, df2_w)

            self.df1_w = df1_w
            self.df2_w = df2_w

        return self.df1_w, self.df2_w


    def get_surface_response_function(self, z0=0., filename='surf_EELS'):
        """Calculate surface response function."""

	if self.chi0_wGG is None:
            self.initialize()
            self.calculate()

        g_w2 = np.zeros((self.Nw,2), dtype=complex)
        assert self.acell_cv[0,2] == 0. and self.acell_cv[1,2] == 0.

        Nz = self.nG[2] # number of points in z direction
        tmp = np.zeros(Nz, dtype=int)
        nGz = 0         # number of G_z 
        for i in range(self.npw):
            if self.Gvec_Gc[i, 0] == 0 and self.Gvec_Gc[i, 1] == 0:
                tmp[nGz] = self.Gvec_Gc[i, 2]
                nGz += 1
        assert (np.abs(self.Gvec_Gc[:nGz, :2]) < 1e-10).all()

        for id, xc in enumerate(['RPA', 'ALDA']):
            chi_wGG = self.get_chi(xc=xc)
    
            # The first nGz are all Gx=0 and Gy=0 component
            chi_wgg_LFC = chi_wGG[:, :nGz, :nGz]
            del chi_wGG
            chi_wzz_LFC = np.zeros((self.Nw_local, Nz, Nz), dtype=complex)        
    
            # Fourier transform of chi_wgg to chi_wzz
            Gz_g = tmp[:nGz] * self.bcell_cv[2,2]
            z_z = np.linspace(0, self.acell_cv[2,2]-self.h_cv[2,2], Nz)
            phase1_zg = np.exp(1j  * np.outer(z_z, Gz_g))
            phase2_gz = np.exp(-1j * np.outer(Gz_g, z_z))
    
            for iw in range(self.Nw_local):
                chi_wzz_LFC[iw] = np.dot(np.dot(phase1_zg, chi_wgg_LFC[iw]), phase2_gz)
            chi_wzz_LFC /= self.acell_cv[2,2]        
    
            # Get surface response function
    
            z_z -= z0 / Bohr
            q_v = np.dot(self.q_c, self.bcell_cv)
            qq = sqrt(np.inner(q_v, q_v))
            phase1_1z = np.array([np.exp(qq*z_z)])
            phase2_z1 = np.exp(qq*z_z)
    
            tmp_w = np.zeros(self.Nw_local, dtype=complex)        
            for iw in range(self.Nw_local):
                tmp_w[iw] = np.dot(np.dot(phase1_1z, chi_wzz_LFC[iw]), phase2_z1)[0]            
    
            tmp_w *= -2 * pi / qq * self.h_cv[2,2]**2        
            g_w = np.zeros(self.Nw, dtype=complex)
            self.wcomm.all_gather(tmp_w, g_w)
            g_w2[:, id] = g_w
    
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.imag(g_w2[iw, 0]), np.imag(g_w2[iw, 1])
            f.close()

        # Wait for I/O to finish
        self.comm.barrier()


    def check_sum_rule(self, df1_w=None, df2_w=None):
        """Check f-sum rule."""

	if df1_w is None:
            df1_w = self.df1_w
            df2_w = self.df2_w

        N1 = N2 = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N1 += np.imag(df1_w[iw]) * w
            N2 += np.imag(df2_w[iw]) * w
        N1 *= self.dw * self.vol / (2 * pi**2)
        N2 *= self.dw * self.vol / (2 * pi**2)

        self.printtxt('')
        self.printtxt('Sum rule for ABS:')
        nv = self.nvalence
        self.printtxt('Without local field: N1 = %f, %f  %% error' %(N1, (N1 - nv) / nv * 100) )
        self.printtxt('Include local field: N2 = %f, %f  %% error' %(N2, (N2 - nv) / nv * 100) )

        N1 = N2 = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N1 -= np.imag(1/df1_w[iw]) * w
            N2 -= np.imag(1/df2_w[iw]) * w
        N1 *= self.dw * self.vol / (2 * pi**2)
        N2 *= self.dw * self.vol / (2 * pi**2)
                
        self.printtxt('')
        self.printtxt('Sum rule for EELS:')
        nv = self.nvalence
        self.printtxt('Without local field: N1 = %f, %f  %% error' %(N1, (N1 - nv) / nv * 100) )
        self.printtxt('Include local field: N2 = %f, %f  %% error' %(N2, (N2 - nv) / nv * 100) )


    def get_macroscopic_dielectric_constant(self, df1=None, df2=None):
        """Calculate macroscopic dielectric constant. Returns eM1 and eM2

        Macroscopic dielectric constant is defined as the real part of dielectric function at w=0.
        
        Parameters:

        eM1: float
            Dielectric constant without local field correction.
        eM2: float
            Dielectric constant with local field correction.

        """

        if df1 is None:
            df1, df2 = self.get_dielectric_function()
        eM1, eM2 = np.real(df1[0]), np.real(df2[0])

        self.printtxt('')
        self.printtxt('Macroscopic dielectric constant:')
        self.printtxt('    Without local field : %f' %(eM1) )
        self.printtxt('    Include local field : %f' %(eM2) )        
            
        return eM1, eM2


    def get_absorption_spectrum(self, df1=None, df2=None, filename='Absorption.dat'):
        """Calculate optical absorption spectrum. By default, generate a file 'Absorption.dat'.

        Optical absorption spectrum is obtained from the imaginary part of dielectric function.
        """

        if df1 is None:
            df1, df2 = self.get_dielectric_function()
        Nw = df1.shape[0]

        if rank == 0:
            f = open(filename,'w')
            for iw in range(Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.real(df1[iw]), np.imag(df1[iw]), \
                      np.real(df2[iw]), np.imag(df2[iw])
            f.close()

        # Wait for I/O to finish
        self.comm.barrier()


    def get_EELS_spectrum(self, df1=None, df2=None, filename='EELS.dat'):
        """Calculate EELS spectrum. By default, generate a file 'EELS.dat'.

        EELS spectrum is obtained from the imaginary part of the inverse of dielectric function.
        """

        # calculate RPA dielectric function
        if df1 is None:
            df1, df2 = self.get_dielectric_function()
        Nw = df1.shape[0]

        # calculate LDA chi
        q_v = np.dot(self.q_c, self.bcell_cv)
        coef = 4 * pi / np.inner(q_v, q_v)
        chi_wGG = self.get_chi(xc='ALDA')
        chi_w = np.zeros(self.Nw, dtype=complex)
        self.wcomm.all_gather(chi_wGG[:,0,0].copy(), chi_w)
        chi_w *= coef
        
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, -np.imag(1./df1[iw]), -np.imag(1./df2[iw]), -np.imag(chi_w[iw])
            f.close()

        # Wait for I/O to finish
        self.comm.barrier()


    def get_jdos(self, f_kn, e_kn, kq, dw, Nw, sigma):
        """Calculate Joint density of states"""

        JDOS_w = np.zeros(Nw)
        nkpt = f_kn.shape[0]
        nbands = f_kn.shape[1]

        for k in range(nkpt):
            for n in range(nbands):
                for m in range(nbands):
                    focc = f_kn[k, n] - f_kn[kq[k], m]
                    w0 = e_kn[kq[k], m] - e_kn[k, n]
                    if focc > 0 and w0 >= 0:
                        deltaw = delta_function(w0, dw, Nw, sigma)
                        for iw in range(Nw):
                            if deltaw[iw] > 1e-8:
                                JDOS_w[iw] += focc * deltaw[iw]

        w = np.arange(Nw) * dw * Hartree

        return w, JDOS_w


    def calculate_induced_density(self, q, w):
        """ Evaluate induced density for a certain q and w.

        Parameters:

        q: ndarray
            Momentum tranfer at reduced coordinate.
        w: scalar
            Energy (eV).
        """

        if type(w) is int:
            iw = w
            w = self.wlist[iw] / Hartree
        elif type(w) is float:
            w /= Hartree
            iw = int(np.round(w / self.dw))
        else:
            raise ValueError('Frequency not correct !')

        self.printtxt('Calculating Induced density at q, w (iw)')
        self.printtxt('(%f, %f, %f), %f(%d)' %(q[0], q[1], q[2], w*Hartree, iw))

        # delta_G0
        delta_G = np.zeros(self.npw)
        delta_G[0] = 1.

        # coef is (q+G)**2 / 4pi
        coef_G = np.zeros(self.npw)
        for iG in range(self.npw):
            qG = np.dot(q + self.Gvec_Gc[iG], self.bcell_cv)
            coef_G[iG] = np.dot(qG, qG)
        coef_G /= 4 * pi

        # obtain chi_G0(q,w)
        dm_wGG = self.get_RPA_dielectric_matrix()
        tmp_GG = dm_wGG[iw]
        del dm_wGG
        chi_G = (np.linalg.inv(tmp_GG)[:, 0] - delta_G) * coef_G

        gd = self.gd
        r = gd.get_grid_point_coordinates()

        # calculate dn(r,q,w)
        drho_R = gd.zeros(dtype=complex)
        for iG in range(self.npw):
            qG = np.dot(q + self.Gvec_Gc[iG], self.bcell_cv)
            qGr_R = np.inner(qG, r.T).T
            drho_R += chi_G[iG] * np.exp(1j * qGr_R)

        # phase = sum exp(iq.R_i)
        # drho_R /= self.vol * nkpt / phase
        return drho_R


    def get_induced_density_z(self, q, w):
        """Get induced density on z axis (summation over xy-plane). """

        drho_R = self.calculate_induced_density(q, w)

        drho_z = np.zeros(self.nG[2],dtype=complex)
#        dxdy = np.cross(self.h_c[0], self.h_c[1])

        for iz in range(self.nG[2]):
            drho_z[iz] = drho_R[:,:,iz].sum()

        return drho_z


    def project_chi_to_LCAO_pair_orbital(self, orb_MG):

        nLCAO = orb_MG.shape[0]
        N = np.zeros((self.Nw, nLCAO, nLCAO), dtype=complex)

        kcoulinv_GG = np.zeros((self.npw, self.npw))
        for iG in range(self.npw):
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            kcoulinv_GG[iG, iG] = np.dot(qG, qG)

        kcoulinv_GG /= 4.*pi

        dm_wGG = self.get_RPA_dielectric_matrix()

        for mu in range(nLCAO):
            for nu in range(nLCAO):
                pairorb_R = orb_MG[mu] * orb_MG[nu]
                if not (pairorb_R * pairorb_R.conj() < 1e-10).all():
                    tmp_G = np.fft.fftn(pairorb_R) * self.vol / self.nG0

                    pairorb_G = np.zeros(self.npw, dtype=complex)
                    for iG in range(self.npw):
                        index = self.Gindex[iG]
                        pairorb_G[iG] = tmp_G[index[0], index[1], index[2]]

                    for iw in range(self.Nw):
                        chi_GG = (dm_wGG[iw] - np.eye(self.npw)) * kcoulinv_GG
                        N[iw, mu, nu] = (np.outer(pairorb_G.conj(), pairorb_G) * chi_GG).sum()
#                        N[iw, mu, nu] = np.inner(pairorb_G.conj(),np.inner(pairorb_G, chi_GG))

        return N


    def write(self, filename, all=False):
        """Dump essential data"""

        data = {'nbands': self.nbands,
                'acell': self.acell_cv, #* Bohr,
                'bcell': self.bcell_cv, #/ Bohr,
                'h_cv' : self.h_cv,   #* Bohr,
                'nG'   : self.nG,
                'nG0'  : self.nG0,
                'vol'  : self.vol,   #* Bohr**3,
                'BZvol': self.BZvol, #/ Bohr**3,
                'nkpt' : self.nkpt,
                'ecut' : self.ecut,  #* Hartree,
                'npw'  : self.npw,
                'eta'  : self.eta,   #* Hartree,
                'ftol' : self.ftol,  #* self.nkpt,
                'Nw'   : self.Nw,
                'NwS'  : self.NwS,
                'dw'   : self.dw,    # * Hartree,
                'q_red': self.q_c,
                'q_car': self.qq_v,    # / Bohr,
                'qmod' : np.dot(self.qq_v, self.qq_v), # / Bohr
                'nvalence'     : self.nvalence,                
                'hilbert_trans' : self.hilbert_trans,
                'optical_limit' : self.optical_limit,
                'e_kn'         : self.e_kn,          # * Hartree,
                'f_kn'         : self.f_kn,          # * self.nkpt,
                'bzk_kc'       : self.bzk_kc,
                'ibzk_kc'      : self.ibzk_kc,
                'kq_k'         : self.kq_k,
                'op_scc'       : self.op_scc,
                'Gvec_Gc'       : self.Gvec_Gc,
                'dfNLF_w'      : self.df1_w,
                'dfLFC_w'      : self.df2_w}

        if all == True:
            from gpaw.response.parallel import par_write
            par_write('chi0','chi0_wGG',self.wcomm,self.chi0_wGG)
        
        if rank == 0:
            pickle.dump(data, open(filename, 'w'), -1)

        self.comm.barrier()


    def read(self, filename):
        """Read data from pickle file"""

        data = pickle.load(open(filename))
        
        self.nbands = data['nbands']
        self.acell_cv = data['acell']
        self.bcell_cv = data['bcell']
        self.h_cv   = data['h_cv']
        self.nG    = data['nG']
        self.nG0   = data['nG0']
        self.vol   = data['vol']
        self.BZvol = data['BZvol']
        self.nkpt  = data['nkpt']
        self.ecut  = data['ecut']
        self.npw   = data['npw']
        self.eta   = data['eta']
        self.ftol  = data['ftol']
        self.Nw    = data['Nw']
        self.NwS   = data['NwS']
        self.dw    = data['dw']
        self.q_c   = data['q_red']
        self.qq_v  = data['q_car']
        self.qmod  = data['qmod']
        
        self.hilbert_trans = data['hilbert_trans']
        self.optical_limit = data['optical_limit']
        self.e_kn  = data['e_kn']
        self.f_kn  = data['f_kn']
        self.nvalence= data['nvalence']
        self.bzk_kc  = data['bzk_kc']
        self.ibzk_kc = data['ibzk_kc']
        self.kq_k    = data['kq_k']
        self.op_scc  = data['op_scc']
        self.Gvec_Gc  = data['Gvec_Gc']
        self.df1_w   = data['dfNLF_w']
        self.df2_w   = data['dfLFC_w']

        self.printtxt('Read succesfully !')
