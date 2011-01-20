import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw.utilities.blas import gemmdot, gemv, scal, axpy
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities import pack
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.fd_operators import Gradient
from gpaw.coulomb import CoulombNEW
from gpaw.xc import XC
from gpaw.mpi import world, size, rank
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.response.symmetrize import find_kq, find_ibzkpt, symmetrize_wavefunction
from gpaw.response.math_func import two_phi_planewave_integrals
from gpaw.response.chi import CHI
from gpaw.response.parallel import parallel_partition


class BSE(CHI):
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
                 hilbert_trans=True,
                 optical_limit=False,
                 kcommsize=None):

        CHI.__init__(self, calc, nbands, w, q, ecut,
                     eta, ftol, txt, hilbert_trans, optical_limit, kcommsize)

    def initialize(self):

        calc = self.calc

        # frequency points init
        self.dw = self.w_w[1] - self.w_w[0]
        assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
        assert self.w_w.max() == self.w_w[-1]

        self.eta /= Hartree
        self.ecut /= Hartree
        self.dw /= Hartree
        self.w_w  /= Hartree
        self.wmax = self.w_w[-1] 
        self.Nw  = int(self.wmax / self.dw) + 1
        
        # kpoint init
        self.bzk_kc = calc.get_bz_k_points()
        self.ibzk_kc = calc.get_ibz_k_points()
        self.nkpt = self.bzk_kc.shape[0]
        self.ftol /= self.nkpt

        assert calc.wfs.nspins == 1

        # cell init
        self.acell_cv = calc.atoms.cell / Bohr
        self.bcell_cv, self.vol, self.BZvol = get_primitive_cell(self.acell_cv)

        # grid init
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
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

        # band init
        if self.nbands is None:
            self.nbands = calc.wfs.nbands

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

        # find the pair index and initialized pair energy (e_i - e_j) and occupation(f_i-f_j)
        self.e_S = {}
        focc_s = {}
        self.Sindex_S3 = {}
        iS = 0
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
        self.print_stuff()


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

        # Coulomb kernel init
        self.kc_G = np.zeros(self.npw)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            qG = np.dot(self.q_c + self.Gvec_Gc[iG], self.bcell_cv)
            self.kc_G[iG] = 1. / np.inner(qG, qG)
        if self.optical_limit:
            self.kc_G[0] = 0.
        self.printtxt('Finished Coulomb kernel !')
        
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
            print iS
            for jS in range(self.nS):
                tmp_jS = v_SS[:,jS] * rhoG0_S * focc_S
                tmp = np.outer(tmp_iS, tmp_jS.conj()).sum() * overlap_SS[iS, jS]
                epsilon_w += tmp * tmp_w
        self.Scomm.sum(epsilon_w)

        epsilon_w *=  - 4 * pi / np.inner(self.qq_v, self.qq_v) / self.vol
        epsilon_w += 1        

        return w_S, epsilon_w

    
    def density_matrix_Gspace(self,n,m,k):

        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        gd = self.gd

        ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[k])
        ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[kq_k[k]])
        
        psitold_g = self.get_wavefunction(ibzkpt1, n, k, True)
        psit1_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop1], ibzk_kc[ibzkpt1],
                                                      bzk_kc[k], timerev1)
        
        psitold_g = self.get_wavefunction(ibzkpt2, m, kq_k[k], True)
        psit2_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop2], ibzk_kc[ibzkpt2],
                                          bzk_kc[kq_k[k]], timerev2)

        # FFT
        tmp_g = psit1_g.conj()* psit2_g * self.expqr_g
        rho_g = np.fft.fftn(tmp_g) * self.vol / self.nG0

        # Here, planewave cutoff is applied
        rho_G = np.zeros(self.npw, dtype=complex)
        for iG in range(self.npw):
            index = self.Gindex_G[iG]
            rho_G[iG] = rho_g[index[0], index[1], index[2]]

        if self.optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

            phase_cd = np.exp(2j * pi * gd.sdisp_cd * bzk_kc[kq_k[k], :, np.newaxis])
            for ix in range(3):
                d_c[ix](psit2_g, dpsit_g, phase_cd)
                tmp[ix] = gd.integrate(psit1_g.conj() * dpsit_g)
            rho_G[0] = -1j * np.dot(self.qq_v, tmp)

        # PAW correction
        pt = self.pt
        P1_ai = pt.dict()
        pt.integrate(psit1_g, P1_ai, k)
        P2_ai = pt.dict()
        pt.integrate(psit2_g, P2_ai, kq_k[k])
                        
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
            gemv(1.0, self.phi_aGp[a], P_p, 1.0, rho_G)

        if self.optical_limit:
            rho_G[0] /= self.e_kn[ibzkpt2, m] - self.e_kn[ibzkpt1, n]

        return rho_G


    def printtxt(self, text):
        print >> self.txt, text


    def print_stuff(self):

        printtxt = self.printtxt
        printtxt('')
        printtxt('Parameters used:')
        printtxt('')
        printtxt('Unit cell (a.u.):')
        printtxt(self.acell_cv)
        printtxt('Reciprocal cell (1/a.u.)')
        printtxt(self.bcell_cv)
        printtxt('Number of Grid points / G-vectors, and in total: (%d %d %d), %d'
                  %(self.nG[0], self.nG[1], self.nG[2], self.nG0))
        printtxt('Volome of cell (a.u.**3)     : %f' %(self.vol) )
        printtxt('BZ volume (1/a.u.**3)        : %f' %(self.BZvol) )
        printtxt('')                         
        printtxt('Number of bands              : %d' %(self.nbands) )
        printtxt('Number of kpoints            : %d' %(self.nkpt) )
        printtxt('Planewave ecut (eV)          : (%f, %f, %f)' %(self.ecut[0]*Hartree,self.ecut[1]*Hartree,self.ecut[2]*Hartree) )
        printtxt('Number of planewave used     : %d' %(self.npw) )
        printtxt('Broadening (eta)             : %f' %(self.eta * Hartree))
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        if self.optical_limit:
            printtxt('Optical limit calculation ! (q=0.00001)')
        else:
            printtxt('q in reduced coordinate        : (%f %f %f)' %(self.q_c[0], self.q_c[1], self.q_c[2]) )
            printtxt('q in cartesian coordinate (1/A): (%f %f %f) '
                  %(self.qq_v[0] / Bohr, self.qq_v[1] / Bohr, self.qq_v[2] / Bohr) )
            printtxt('|q| (1/A)                      : %f' %(sqrt(np.dot(self.qq_v / Bohr, self.qq_v / Bohr))) )
        printtxt('')
        printtxt('Number of pair orbitals      : %d' %(self.nS) )
        printtxt('Parallelization scheme:')
        printtxt('   Total cpus         : %d' %(world.size))
        printtxt('   pair orb parsize   : %d' %(self.Scomm.size))        
        
