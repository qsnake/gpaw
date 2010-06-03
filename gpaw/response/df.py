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
                 nband=None,
                 wmax=None,
                 dw=None,
                 wlist=None,
                 q=None,
                 Ecut=100.,
                 eta=0.2,
                 sigma=1e-5,
                 ftol=1e-7,
                 txt=None,
                 HilbertTrans=True,
                 OpticalLimit=False,
                 kcommsize=None):

        CHI.__init__(self, calc, nband, wmax, dw, wlist, q, Ecut,
                     eta, sigma, ftol, txt, HilbertTrans, OpticalLimit, kcommsize)


    def get_RPA_dielectric_matrix(self):

            
        tmp = np.eye(self.npw, self.npw)
        dm_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype = complex)

        for iw in range(self.Nw_local):
            for iG in range(self.npw):
                qG = np.array([np.inner(self.q + self.Gvec[iG],
                                       self.bcell[:,i]) for i in range(3)])
                dm_wGG[iw,iG] =  tmp[iG] - 4 * pi / np.inner(qG, qG) * self.chi0_wGG[iw,iG]

        return dm_wGG


    def get_dielectric_function(self):

        dm_wGG = self.get_RPA_dielectric_matrix()

        Nw_local = dm_wGG.shape[0]
        dfNLF_w = np.zeros(Nw_local, dtype = complex)
        dfLFC_w = np.zeros(Nw_local, dtype = complex)
        df1_w = np.zeros(self.Nw, dtype = complex)
        df2_w = np.zeros(self.Nw, dtype = complex)

        for iw in range(Nw_local):
            tmp = dm_wGG[iw]
            dfLFC_w[iw] = 1. / np.linalg.inv(tmp)[0, 0]
            dfNLF_w[iw] = tmp[0, 0]

        self.wcomm.all_gather(dfNLF_w, df1_w)
        self.wcomm.all_gather(dfLFC_w, df2_w)

        self.df1_w = df1_w
        self.df2_w = df2_w

        return df1_w, df2_w


    def check_sum_rule(self, df1_w, df2_w):

        N1 = N2 = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N1 += np.imag(df1_w[iw]) * w
            N2 += np.imag(df2_w[iw]) * w
        N1 *= self.dw * self.vol / (2 * pi**2)
        N2 *= self.dw * self.vol / (2 * pi**2)

        self.printtxt('')
        self.printtxt('Sum rule:')
        nv = self.nvalence
        self.printtxt('Without local field: N1 = %f, %f  %% error' %(N1, (N1 - nv) / nv * 100) )
        self.printtxt('Include local field: N2 = %f, %f  %% error' %(N2, (N2 - nv) / nv * 100) )


    def get_macroscopic_dielectric_constant(self, df1=None, df2=None):

        if df1 is None:
            df1, df2 = self.get_dielectric_function()
        eM1, eM2 = np.real(df1[0]), np.real(df2[0])
        
        self.printtxt('Macroscopic dielectric constant:')
        self.printtxt('    Without local field : %f' %(eM1) )
        self.printtxt('    Include local field : %f' %(eM2) )        
            
        return eM1, eM2


    def get_absorption_spectrum(self, df1=None, df2=None, filename='Absorption'):

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


    def get_EELS_spectrum(self, df1=None, df2=None, filename='EELS'):

        if df1 is None:
            df1, df2 = self.get_dielectric_function()
        Nw = df1.shape[0]

        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, -np.imag(1./df1[iw]), -np.imag(1./df2[iw])
            f.close()

        # Wait for I/O to finish
        self.comm.barrier()


    def get_jdos(self, f_kn, e_kn, kq, dw, Nw, sigma):
        """Calculate Joint density of states"""

        JDOS_w = np.zeros(Nw)
        nkpt = f_kn.shape[0]
        nband = f_kn.shape[1]

        for k in range(nkpt):
            for n in range(nband):
                for m in range(nband):
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
            qG = np.array([np.inner(q + self.Gvec[iG],
                            self.bcell[:,i]) for i in range(3)])

            coef_G[iG] = np.inner(qG, qG)
        coef_G /= 4 * pi

        # obtain chi_G0(q,w)
        dm_wGG = self.get_RPA_dielectric_matrix()
        tmp = dm_wGG[iw]
        del dm_wGG
        chi_G = (np.linalg.inv(tmp)[:, 0] - delta_G) * coef_G

        gd = self.calc.wfs.gd
        r = gd.get_grid_point_coordinates()

        # calculate dn(r,q,w)
        drho_R = gd.zeros(dtype=complex)
        for iG in range(self.npw):
            qG = np.array([np.inner(self.Gvec[iG],
                            self.bcell[:,i]) for i in range(3)])
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
            qG = np.array([np.inner(self.q + self.Gvec[iG],
                            self.bcell[:,i]) for i in range(3)])
            kcoulinv_GG[iG, iG] = np.inner(qG, qG)

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

        data = {'nband': self.nband,
                'acell': self.acell * Bohr,
                'bcell': self.bcell / Bohr,
                'h_cv' : self.h_c * Bohr,
                'nG'   : self.nG,
                'nG0'  : self.nG0,
                'vol'  : self.vol * Bohr**3,
                'BZvol': self.BZvol / Bohr**3,
                'nkpt' : self.nkpt,
                'Ecut' : self.Ecut * Hartree,
                'npw'  : self.npw,
                'eta'  : self.eta * Hartree,
                'ftol' : self.ftol * self.nkpt,
                'Nw'   : self.Nw,
                'NwS'  : self.NwS,
                'q_red': self.q,
                'q_car': self.qq / Bohr,
                'qmod' : np.inner(self.qq / Bohr, self.qq / Bohr),
                'HilbertTrans' : self.HilbertTrans,
                'OpticalLimit' : self.OpticalLimit,
                'e_kn'         : self.e_kn * Hartree,
                'f_kn'         : self.f_kn * self.nkpt,
                'bzk_kv'       : self.bzk_kv,
                'ibzk_kv'      : self.ibzk_kv,
                'kq_k'         : self.kq,
                'op_scc'       : self.op,
                'Gvec'         : self.Gvec,
                'dfNLF_w'      : self.df1_w,
                'dfLFC_w'      : self.df2_w}

        if all == True:
            data += {'chi0_wGG' : self.chi0_wGG}
        
        if rank == 0:
            pickle.dump(data, open(filename, 'w'), -1)

        self.comm.barrier()


    def read(self, filename):
        """Read data from pickle file"""

        if rank == 0:
            data = pickle.load(open(filename))
        self.comm.barrier()
        
        self.nband = data['nband']
        self.acell = data['acell']
        self.bcell = data['bcell']
        self.h_c   = data['h_cv']
        self.nG    = data['nG']
        self.nG0   = data['nG0']
        self.vol   = data['vol']
        self.BZvol = data['BZvol']
        self.nkpt  = data['nkpt']
        self.Ecut  = data['Ecut']
        self.npw   = data['npw']
        self.eta   = data['eta']
        self.ftol  = data['ftol']
        self.Nw    = data['Nw']
        self.NwS   = data['NwS']
        self.q     = data['q_red']
        self.qq    = data['q_car']
        self.qmod  = data['qmod']
        self.HilbertTrans = data['HilbertTrans']
        self.OpticalLimit = data['OpticalLimit']
        self.e_kn  = data['e_kn']
        self.f_kn  = data['f_kn']
        self.bzk_kv  = data['bzk_kv']
        self.ibzk_kv = data['ibzk_kv']
        self.kq      = data['kq_k']
        self.op      = data['op_scc']
        self.Gvec    = data['Gvec']
        self.df1_w   = data['dfNLF_w']
        self.df2_w   = data['dfLFC_w']

        self.printtxt('Read succesfully !')
