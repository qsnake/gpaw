import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw.utilities.blas import gemmdot, gemv, scal, axpy
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.fd_operators import Gradient
from gpaw.coulomb import CoulombNEW
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.response.symmetrize import find_kq, find_ibzkpt, symmetrize_wavefunction
from gpaw.response.math_func import two_phi_planewave_integrals
from gpaw.response.chi import CHI
from gpaw.xc import XC

class BSE(CHI):
    """This class defines Belth-Selpether equations."""

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

    def initialize(self):

        self.ecut /= Hartree

        calc = self.calc
        
        # kpoint init
        self.bzk_kc = calc.get_bz_k_points()
        self.ibzk_kc = calc.get_ibz_k_points()
        self.nkpt = self.bzk_kc.shape[0]
        self.ftol /= self.nkpt

        # band init
        if self.nbands is None:
            self.nbands = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence
        self.nv = (self.nvalence + 1) // 2 # for semicondutors or molecules
        self.nc = self.nbands - self.nv
        self.nS = self.nc * self.nv * self.nkpt
        print 'bands:',  self.nv, self.nc

        assert calc.wfs.nspins == 1

        # cell init
        self.acell_cv = calc.atoms.cell / Bohr
        self.bcell_cv, self.vol, self.BZvol = get_primitive_cell(self.acell_cv)

        # grid init
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
#        gd = GridDescriptor(self.nG, calc.wfs.gd.cell_cv, pbc_c=True, comm=serial_comm)
        self.gd = gd = calc.wfs.gd      

        # obtain eigenvalues, occupations
        nibzkpt = self.ibzk_kc.shape[0]
        kweight_k = calc.get_k_point_weights()

        try:
            self.e_kn
        except:
            self.printtxt('Use eigenvalues from the calculator.')
            self.e_kn = np.array([calc.get_eigenvalues(kpt=k)
                    for k in range(nibzkpt)]) / Hartree
            self.printtxt('Eigenvalues(k=0) are:')
            print  >> self.txt, self.e_kn[0] * Hartree
        self.f_kn = np.array([calc.get_occupation_numbers(kpt=k) / kweight_k[k]
                    for k in range(nibzkpt)]) / self.nkpt
        print self.f_kn

        # k + q init
        assert self.q_c is not None
        self.qq_v = np.dot(self.q_c, self.bcell_cv) # summation over c

        if self.optical_limit:
            kq_k = np.arange(self.nkpt)
            self.expqr_g = 1.
        else:
            r_vg = gd.get_grid_point_coordinates() # (3, nG)
            qr_g = gemmdot(self.qq_v, r_vg, beta=0.0)
            self.expqr_g = np.exp(-1j * qr_g)
            del r_vg, qr_g
            kq_k = find_kq(self.bzk_kc, self.q_c)
        self.kq_k = kq_k

        # Plane wave init
        self.npw, self.Gvec_Gc, self.Gindex_G = set_Gvectors(self.acell_cv, self.bcell_cv, self.nG, self.ecut)

        # Projectors init
        setups = calc.wfs.setups
        pt = LFC(gd, [setup.pt_j for setup in setups],
                 calc.wfs.kpt_comm, dtype=calc.wfs.dtype, forces=True)
        spos_ac = calc.atoms.get_scaled_positions()
        pt.set_k_points(self.bzk_kc)
        pt.set_positions(spos_ac)
        self.pt = pt

        # Symmetry operations init
        usesymm = calc.input_parameters.get('usesymm')
        if usesymm == None or self.nkpt == 1:
            op_scc = (np.eye(3, dtype=int),)
        elif usesymm == False:
            op_scc = (np.eye(3, dtype=int), -np.eye(3, dtype=int))
        else:
            op_scc = calc.wfs.symmetry.op_scc
        self.op_scc = op_scc

        self.printtxt('Number of planewaves, %d' %(self.npw))

        # For LCAO wfs
        if calc.input_parameters['mode'] == 'lcao':
            calc.initialize_positions()        
        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        kk_Gv = gemmdot(self.q_c + self.Gvec_Gc, self.bcell_cv.copy(), beta=0.0)
        phi_aGp = {}
        for a, id in enumerate(setups.id_a):
            phi_aGp[a] = two_phi_planewave_integrals(kk_Gv, setups[a])
            for iG in range(self.npw):
                phi_aGp[a][iG] *= np.exp(-1j * 2. * pi *
                                         np.dot(self.q_c + self.Gvec_Gc[iG], spos_ac[a]) )

        # For optical limit, G == 0 part should change
        if self.optical_limit:
            for a, id in enumerate(setups.id_a):
                nabla_iiv = setups[a].nabla_iiv
                phi_aGp[a][0] = -1j * (np.dot(nabla_iiv, self.qq_v)).ravel()

        self.phi_aGp = phi_aGp
        self.printtxt('')
        self.printtxt('Finished phi_Gp !')

        self.kc_G = np.zeros(self.npw)
        for iG in range(1,self.npw):
            index = self.Gindex_G[iG]
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            self.kc_G[iG] = 1. / np.inner(qG, qG)

        self.rhoG0_s = np.zeros((self.nv, self.nc, self.nkpt), dtype=complex)
        self.flag = np.zeros((self.nv, self.nc, self.nkpt), dtype=bool)

        self.dw = self.w_w[1] - self.w_w[0]
        assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
        assert self.w_w.max() == self.w_w[-1]

        self.eta /= Hartree
        self.dw /= Hartree
        self.w_w  /= Hartree
        self.wmax = self.w_w[-1] 
        self.Nw  = int(self.wmax / self.dw) + 1

        return

    def calculate(self):

        calc = self.calc
        f_kn = self.f_kn
        e_kn = self.e_kn
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        
        # find the corresponding ibzkpt for k and k+q
        self.ibzk1_k = ibzk1_k = np.zeros(self.nkpt, dtype=int)
        iop1_k = np.zeros(self.nkpt, dtype=int)
        timerev1_k = np.zeros(self.nkpt, dtype=bool)
        
        self.ibzk2_k = ibzk2_k = np.zeros(self.nkpt, dtype=int)
        iop2_k = np.zeros(self.nkpt, dtype=int)
        timerev2_k = np.zeros(self.nkpt, dtype=bool)
        
        for ik in range(self.nkpt):
            ibzk1_k[ik], iop1_k[ik], timerev1_k[ik] = find_ibzkpt(self.op_scc,
                                                ibzk_kc, bzk_kc[ik])
            ibzk2_k[ik], iop2_k[ik], timerev2_k[ik] = find_ibzkpt(self.op_scc,
                                                ibzk_kc, bzk_kc[kq_k[ik]])

        focc_s = np.zeros((self.nv, self.nc, self.nkpt))
        e_s = np.zeros((self.nv, self.nc, self.nkpt))
        for k1 in range(self.nkpt):
            for n1 in range(self.nv):
                for m1 in range(self.nc):
                    e_s[n1, m1, k1] = e_kn[ibzk2_k[k1],m1+self.nv] - e_kn[ibzk1_k[k1],n1]
                    focc_s[n1, m1, k1] = f_kn[ibzk1_k[k1],n1] - f_kn[ibzk2_k[k1],m1+self.nv]

        K_ss = np.zeros((self.nv, self.nc, self.nkpt,
                         self.nv, self.nc, self.nkpt), dtype=complex)
        for k1 in range(self.nkpt):
            for n1 in range(self.nv):
                for m1 in range(self.nc):
                    focc = f_kn[ibzk1_k[k1],n1] - f_kn[ibzk2_k[k1],m1+self.nv] # m1 + self.nv
                    if focc > self.ftol:
                        for k2 in range(self.nkpt):
                            for n2 in range(self.nv):
                                for m2 in range(self.nc):
                                    if 1: #n2 <= n1 and m2 <= m1:
                                        K_ss[n1, m1, k1, n2, m2, k2] = \
                                                 self.calculate_kernel(n1,m1+self.nv,k1,n2,m2+self.nv,k2)
                                        print k1, n1, m1+self.nv, k2, n2, m2+self.nv

        e_S = e_s.ravel()
        focc_S = focc_s.ravel()
        K_SS = K_ss.reshape(self.nv*self.nc*self.nkpt, self.nv*self.nc*self.nkpt)
        H_SS = np.zeros_like(K_SS)
        for iS in range(self.nS):
            H_SS[iS,iS] = e_S[iS]
            for jS in range(self.nS):
                H_SS[iS,jS] += focc_S[iS] * K_SS[iS,jS]

        # Including the resonant and anti-resonant term
        H2_SS = np.zeros((self.nS*2, self.nS*2),dtype=complex)
        for iS in range(self.nS):
            for jS in range(self.nS):
                H2_SS[iS, jS] = H_SS[iS, jS]
                H2_SS[iS+self.nS, jS+self.nS] = -H_SS[iS, jS]
                H2_SS[iS, jS+self.nS] = focc_S[iS] * K_SS[iS, jS]
                H2_SS[iS+self.nS, jS] = -focc_S[iS] * K_SS[iS, jS]

        w, v = np.linalg.eig(H2_SS)
        print 'Solve BSE (without Tamm-Dancoff appx.):'
        print w * Hartree

        # Using Tamm-Dancoff approximation
        w_S = np.zeros(self.nS)
        diagonalize(H_SS, w_S) # H_SS is replaced by the eigenvectors after diagonalization
        print 'Solve BSE (with Tamm-Dancoff appx.):'        
        print w_S * Hartree

        # solve casida's equation
        Omega_SS = np.zeros_like(K_SS)
        for iS in range(self.nS):
            Omega_SS[iS,iS] = e_S[iS]**2
            for jS in range(self.nS):
                Omega_SS[iS,jS] += 2 * focc_S[iS] * e_S[iS] * K_SS[iS,jS]
        w2_S = np.zeros(self.nS)
        diagonalize(Omega_SS, w2_S)
        print 'Solve Casida equation:'
        print np.sqrt(w2_S) * Hartree           


        rhoG0_S = self.rhoG0_s.ravel()
        epsilon_w = np.zeros(self.Nw, dtype=complex)
        
        for iS in range(len(w_S)):
            tmp_S = H_SS[iS] * rhoG0_S
            tmp = np.outer(tmp_S*focc_S, tmp_S.conj()).sum()
            for iw in range(self.Nw):
                epsilon_w[iw] += tmp / (iw*self.dw - w_S[iS] + 1j * self.eta)
        epsilon_w /=  self.nkpt * self.vol
        epsilon_w *= 1-4 * pi / np.inner(self.qq_v, self.qq_v)

        return w_S, epsilon_w


    def calculate_kernel(self, n1,m1,k1,n2,m2,k2):

        gd = self.calc.wfs.gd
        
        psit1_g, P1_ap, rho1_G0 = self.density_matrix_Rspace(n1,m1,k1)
        psit2_g, P2_ap, rho2_G0 = self.density_matrix_Rspace(n2,m2,k2)

        if 0: #FFT:
            rho1_G = self.density_matrix_Gspace(psit1_g, P1_ap, rho1_G0, n1, m1, k1)
            rho2_G = self.density_matrix_Gspace(psit2_g, P2_ap, rho2_G0, n2, m2, k2)

            kernel= 4 * pi / self.vol * np.sum(rho1_G.conj() * rho2_G * self.kc_G)

        if 1: # Poisson:
            spos_ac = self.calc.atoms.get_scaled_positions()
            coulomb = CoulombNEW(gd, self.calc.wfs.setups, spos_ac, fft=False)
            P1_aP = {}
            P2_aP = {}
            for a, id in enumerate(self.calc.wfs.setups.id_a):
                ni = self.calc.wfs.setups[a].ni
                P1_aP[a] = pack(P1_ap[a].conj().reshape(ni,ni), tolerance=1e30)
                P2_aP[a] = pack(P2_ap[a].reshape(ni,ni), tolerance=1e30)
                            
            kernel = coulomb.calculate(psit1_g.conj(), psit2_g, P1_aP, P2_aP) / Hartree

        # xc kernel, PAW term not included yet.
        xc = XC('LDA')
        nt_sG = self.calc.density.nt_sG
        fxc_sg = np.zeros_like(nt_sG)
        xc.calculate_fxc(gd, nt_sG, fxc_sg)

#        kernel += gd.integrate(psit1_g.conj() * psit2_g * fxc_sg[0])

        return kernel

    def density_matrix_Gspace(self, psit_g, P_ap, rho_G0, n, m, k):


        rho_g = np.fft.fftn(psit_g*self.expqr_g) * self.vol / self.nG0

        # Here, planewave cutoff is applied
        rho_G = np.zeros(self.npw, dtype=complex)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            rho_G[iG] = rho_g[index[0], index[1], index[2]]

        if self.optical_limit:
            rho_G[0] = rho_G0

        # PAW correction
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            gemv(1.0, self.phi_aGp[a], P_ap[a], 1.0, rho_G)

        if self.optical_limit: 
            rho_G[0] /= self.e_kn[self.ibzk2_k[k], m] - self.e_kn[self.ibzk1_k[k], n]
        if self.flag[n, m-self.nv, k] == False:
            self.rhoG0_s[n, m-self.nv, k] = rho_G[0]
            self.flag[n, m-self.nv, k] = True

        return rho_G

    
    def density_matrix_Rspace(self,n,m,k):

        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k

        ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[k])
        ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[kq_k[k]])
        
        psitold_g = self.get_wavefunction(ibzkpt1, n, k, True)
        psit1_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop1], ibzk_kc[ibzkpt1],
                                                      bzk_kc[k], timerev1)
        
        psitold_g = self.get_wavefunction(ibzkpt2, m, kq_k[k], True)
        psit2_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop2], ibzk_kc[ibzkpt2],
                                          bzk_kc[kq_k[k]], timerev2)

        psit1_g = np.complex128(self.calc.wfs.kpt_u[0].psit_nG[n])
        psit2_g = np.complex128(self.calc.wfs.kpt_u[0].psit_nG[m])
        
        pt = self.pt
        P1_ai = pt.dict()
        pt.integrate(psit1_g, P1_ai, k)
        P2_ai = pt.dict()
        pt.integrate(psit2_g, P2_ai, kq_k[k])
        P_ap = {}
        for a, id in enumerate(self.calc.wfs.setups.id_a):        
            P_ap[a] = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
            P_ni = self.calc.wfs.kpt_u[0].P_ani[a]

        gd = self.gd
        rhoG0 = None
        if self.optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

            phase_cd = np.exp(2j * pi * gd.sdisp_cd * bzk_kc[kq_k[k], :, np.newaxis])
            for ix in range(3):
                d_c[ix](psit2_g, dpsit_g, phase_cd)
                tmp[ix] = gd.integrate(psit1_g.conj() * dpsit_g)
            rhoG0 = -1j * np.dot(self.qq_v, tmp)

        return psit1_g.conj()* psit2_g, P_ap, rhoG0
