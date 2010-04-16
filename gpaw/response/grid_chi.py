import sys
from time import time, ctime
from math import pi, sqrt
from scipy.special import sph_jn
import numpy as np
from ase.units import Hartree, Bohr
from ase.data import chemical_symbols
from ase.dft.kpoints import get_monkhorst_shape

from gpaw.xc_functional import XCFunctional
from gpaw.utilities.blas import gemmdot
from gpaw.utilities import unpack, devnull
from gpaw.utilities.memory import maxrss

from gpaw.gaunt import gaunt as G_LLL
from gpaw.spherical_harmonics import Y
from gpaw.setup_data import SetupData
from gpaw.setup import Setup
from gpaw.fd_operators import Gradient
from gpaw.mpi import _Communicator, world, rank, size

class CHI:
    def __init__(self):
        self.xc = 'LDA'
        self.nspin = 1
        
        self.comm = _Communicator(world)
        if rank == 0:
            self.txt = sys.stdout #open('out.txt','w')
        else:
            sys.stdout = devnull
            self.txt = devnull


    def initialize(self, c, q, wmax, dw, eta=0.2, Ecut=100.,
                   sigma=1e-5, HilbertTrans = True): # eV

        print  >> self.txt
        print  >> self.txt, 'Response function calculation started at:'
        self.starttime = time()
        print  >> self.txt, ctime()

        try:
            self.ncalc = len(c)
        except:
            self.ncalc = 1
            c = (c,)
            
        self.calc = calc = c[0]
        self.c = c

        bzkpt_kG = calc.get_bz_k_points()
        self.nkpt = bzkpt_kG.shape[0]
        self.nkptxyz = get_monkhorst_shape(bzkpt_kG)

        # parallize in kpoints
        self.nkpt_local = int(self.nkpt / size)

        self.kstart = rank * self.nkpt_local
        self.kend = (rank + 1) * self.nkpt_local
        if rank == size - 1:
            self.kend = self.nkpt

        try:
            self.nband
        except:
            self.nband = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence

        assert calc.wfs.nspins == 1
    
        self.acell = calc.atoms.cell / Bohr
        self.get_primitive_cell()

        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]

        self.h_c = calc.wfs.gd.h_cv

        if self.ncalc == 1:
            self.nibzkpt = calc.get_ibz_k_points().shape[0]
            kweight = calc.get_k_point_weights()

            # obtain eigenvalues, occupations
            self.e_kn = np.array([calc.get_eigenvalues(kpt=k)
                        for k in range(self.nibzkpt)]) / Hartree
            self.f_kn = np.array([calc.get_occupation_numbers(kpt=k) / kweight[k]
                        for k in range(self.nibzkpt)]) / self.nkpt

        else:
            
            assert self.ncalc == 2
            assert calc.get_bz_k_points().shape == calc.get_ibz_k_points().shape
            
            # obtain eigenvalues, occupations
            self.e1_kn = np.array([c[0].get_eigenvalues(kpt=k)
                         for k in range(self.nkpt)]) / Hartree
            self.f1_kn = np.array([c[0].get_occupation_numbers(kpt=k)
                         for k in range(self.nkpt)]) 
    
            self.e2_kn = np.array([c[1].get_eigenvalues(kpt=k)
                         for k in range(self.nkpt)]) / Hartree
            self.f2_kn = np.array([c[1].get_occupation_numbers(kpt=k)
                         for k in range(self.nkpt)])
    
        # construct q.r
        r = calc.wfs.gd.get_grid_point_coordinates() # (3, nG[0], nG[1], nG[2])
        h_c = self.h_c # 3 * 3 matrix
        self.q = q
        self.qq = qq = np.array([np.inner(self.q, self.bcell[:,i]) for i in range(3)])
        self.qr = np.inner(self.qq, r.T).T

        # unit conversion
        self.wmin = 0
        self.wmax  = wmax / Hartree
        self.wcut = (wmax + 5.) / Hartree
        self.dw = dw / Hartree
        self.Nw = int((self.wmax - self.wmin) / self.dw) + 1
        self.NwS = int((self.wcut - self.wmin) / self.dw) + 1
        self.eta = eta / Hartree
        self.Ecut = Ecut / Hartree
        self.sigma = sigma
        
        self.set_Gvectors()

#        nt_G = calc.density.nt_sG[0] # G is the number of grid points
#        self.Kxc_GG = self.calculate_Kxc(calc.wfs.gd, nt_G)          # G here is the number of plane waves

        # dielectric function and macroscopic dielectric function
        self.eRPA_wGG = np.zeros((self.Nw, self.npw, self.npw), dtype = complex)
        self.eMRPA_GG = np.zeros((self.npw, self.npw), dtype = complex)
        self.epsilonM = 0.

        self.HilbertTrans = HilbertTrans

        self.print_stuff()

        return


    def periodic(self):
        
        if self.ncalc == 1:
            # Disable optical limit calculation at the moment
            #print >> self.txt, 'Optical limit calculation !'            
            #self.OpticalLimit()
            print >> self.txt, 'EELS spectrum (finite q) calculation !'
            self.Finiteq()

        else:
            print >> self.txt, 'Numerically shift kpoint calculation !'
            self.ShiftKpoint()

        print  >> self.txt
        print  >> self.txt, 'Response function calculation ended at:'
        endtime = time()
        print  >> self.txt, ctime()
        dt = (endtime - self.starttime) / 60
        print  >> self.txt, 'For excited states calc, it  took',dt, 'minutes'
        print  >> self.txt, '    and use memory', maxrss() / 1024**2, 'M'


    def OpticalLimit(self):

        calc = self.calc

        setups = calc.wfs.setups
        gd = calc.wfs.gd
        
        f_kn = self.f_kn
        e_kn = self.e_kn
        qq = self.qq
        eta = self.eta

        chi0_w = np.zeros(self.Nw, dtype = complex)

        phi_ii = {}
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_ii.has_key(Z):
                phi_ii[Z] = self.two_phi_derivative(Z)

        d_c = [Gradient(gd, i, dtype=complex).apply for i in range(3)]
        dpsit_G = gd.empty(dtype=complex)
        tmp = np.zeros((3), dtype=complex)


        for k in range(self.nkpt):
            kpt = calc.wfs.kpt_u[k]
            P_ani = kpt.P_ani
            psit_nG = kpt.psit_nG    

            rho_nn = np.zeros((self.nband, self.nband), dtype=complex)            

            for n in range(self.nband):
                for m in range(self.nband):
                    # G = G' = 0 <psi_nk | e**(-iqr) | psi_n'k+q>
                    
                    if np.abs(e_kn[k, m] - e_kn[k, n]) > 1e-8:
                        for ix in range(3):
                            d_c[ix](psit_nG[m], dpsit_G, kpt.phase_cd)
                            tmp[ix] = gd.integrate( psit_nG[n].conj() * dpsit_G)
                        rho_nn[n, m] = -1j * np.inner(qq, tmp) 

                        # PAW correction
                        for a, id in enumerate(setups.id_a):
                            Z, type, basis = id
                            P_ii = np.outer(P_ani[a][n].conj(), P_ani[a][m])
                            rho_nn[n, m] += (P_ii * phi_ii[Z]).sum() 
                        rho_nn[n, m] /= e_kn[k, m] - e_kn[k, n]

            # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
            C_nn = np.zeros((self.nband, self.nband), dtype=complex)
            for iw in range(self.Nw):
                w = iw * self.dw
                for n in range(self.nband):
                    for m in range(self.nband):
                        if  np.abs(f_kn[k, n] - f_kn[k, m]) > 1e-8:
                            C_nn[n, m] = (f_kn[k, n] - f_kn[k, m]) / (
                             w + e_kn[k, n] - e_kn[k, m] + 1j * eta)

                # get chi0(G=0,G'=0,w)                
                chi0_w[iw] += (rho_nn * C_nn * rho_nn.conj()).sum()

#            chi0_w *= kweight[k] * calc.get_ibz_k_points().shape[0]
            # Obtain Macroscopic Dielectric Constant
            for n in range(self.nband):
                for m in range(self.nband):
                    C_nn[n, m] = 0.
                    if np.abs(f_kn[k, n] - f_kn[k, m]) > 1e-8:
                        C_nn[n, m] = (f_kn[k, n] - f_kn[k, m]) / (
                                  e_kn[k, n] - e_kn[k, m] )
            self.epsilonM += (rho_nn * C_nn * rho_nn.conj()).sum()

            print >> self.txt, 'finished kpoint', k
            
        for iw in range(self.Nw):
            self.epsilonRPA[iw] =  1 - 4 * pi / np.inner(qq, qq) * chi0_w[iw] / self.vol


    def ShiftKpoint(self):

        f1_kn = self.f1_kn
        f2_kn = self.f2_kn
        e1_kn = self.e1_kn
        e2_kn = self.e2_kn
        eta = self.eta
        qr = self.qr
        qq = self.qq
        c = self.c
        setups = c[0].wfs.setups
        gd = c[0].wfs.gd

        if not self.HilbertTrans:
            chi0_wGG = np.zeros((self.Nw, self.npw, self.npw), dtype = complex)
        else:
            specfunc_wGG = np.zeros((self.NwS, self.npw, self.npw), dtype = complex)
        chi0M_GG = np.zeros((self.npw, self.npw), dtype = complex)

        # calculate <phi_i | e**(-iq.r) | phi_j>
        phi_Gp = {}
        R_a = c[0].atoms.positions / Bohr
        
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_Gp.has_key(Z):
                phi_Gp[Z] = ( self.two_phi_planewave_integrals(Z)
                                  * np.exp(-1j * np.inner(qq, R_a[a])) )
        print >> self.txt, 'phi_Gii obtained!'

        expqr_G = np.exp(-1j * self.qr)
        
        # calculate chi0
        for k in range(self.kstart, self.kend):
            t1 = time()
            
            kpt0 = c[0].wfs.kpt_u[k]
            kpt1 = c[1].wfs.kpt_u[k]
            P1_ani = kpt0.P_ani
            P2_ani = kpt1.P_ani

            rho_Gnn = np.zeros((self.npw, self.nband, self.nband), dtype=complex)

            for n in range(self.nband):
                psit1_G = kpt0.psit_nG[n].conj() * expqr_G
                for m in range(self.nband):
                    if  np.abs(f1_kn[k, n] - f2_kn[k, m]) > 1e-8:
                        psit2_G = kpt1.psit_nG[m] * psit1_G
                        # fft
                        tmp_G = np.fft.fftn(psit2_G) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex[iG]
                            rho_Gnn[iG, n, m] = tmp_G[index[0], index[1], index[2]]

                            # PAW correction
                        for a, id in enumerate(setups.id_a):
                            Z, type, basis = id
                            P_p = np.outer(P1_ani[a][n].conj(), P2_ani[a][m]).ravel()
                            rho_Gnn[:, n, m] += np.dot(phi_Gp[Z], P_p)

            del psit1_G, psit2_G
            t2 = time()
            #print  >> self.txt,'Time for density matrix:', t2 - t1, 'seconds'

            if not self.HilbertTrans:
                # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
                C_nn = np.zeros((self.nband, self.nband), dtype=complex)
                for iw in range(self.Nw):
                    w = iw * self.dw
                    for n in range(self.nband):
                        for m in range(self.nband):
                            if  np.abs(f1_kn[k, n] - f2_kn[k, m]) > 1e-8:
                                C_nn[n, m] = (f1_kn[k, n] - f2_kn[k, m]) / (
                                 w + e1_kn[k, n] - e2_kn[k, m] + 1j * eta)
                
                    # get chi0(G=0,G'=0,w)
                    for iG in range(self.npw):
                        for jG in range(self.npw):
                            chi0_wGG[iw,iG,jG] += (rho_Gnn[iG] * C_nn * rho_Gnn[jG].conj()).sum()
            else:
            # calculate spectral function
                for n in range(self.nband):
                    for m in range(self.nband):
                        focc = f1_kn[k,n] - f2_kn[k,m]
                        if focc > 1e-8:
                            w0 = e2_kn[k,m] - e1_kn[k,n]
                            tmp_GG = focc * np.outer(rho_Gnn[:,n,m], rho_Gnn[:,n,m].conj() )
                
                            # calculate delta function
                            deltaw = self.delta_function(w0, self.dw, self.NwS, self.sigma)
                            for wi in range(self.NwS):
                                if deltaw[wi] > 1e-5:
                                    specfunc_wGG[wi] += tmp_GG * deltaw[wi]

            t4 = time()
            #print  >> self.txt,'Time for spectral function loop:', t4 - t2, 'seconds'
            
            # Obtain Macroscopic Dielectric Constant
            C_nn = np.zeros((self.nband, self.nband))
            for n in range(self.nband):
                for m in range(self.nband):
                    C_nn[n, m] = 0.
                    if np.abs(f1_kn[k, n] - f2_kn[k, m]) > 1e-8:
                        C_nn[n, m] = (f1_kn[k, n] - f2_kn[k, m]) / (
                                  e1_kn[k, n] - e2_kn[k, m] )
            for iG in range(self.npw):
                for jG in range(self.npw):
                    chi0M_GG[iG,jG] += (rho_Gnn[iG] * C_nn * rho_Gnn[jG].conj()).sum()
                    
            print >> self.txt, 'finished k', k

        comm = self.comm
        comm.sum(chi0M_GG)

        # Hilbert Transform
        if not self.HilbertTrans:
            comm.sum(chi0_wGG)
        else:
            comm.sum(specfunc_wGG)
            chi0_wGG = self.hilbert_transform(specfunc_wGG)
            del specfunc_wGG
            
        tmp = np.eye(self.npw, self.npw)        
        for iw in range(self.Nw):
            for iG in range(self.npw):
                qG = np.array([np.inner(self.q + self.Gvec[iG],
                                       self.bcell[:,i]) for i in range(3)])
                self.eRPA_wGG[iw,iG] =  tmp[iG] - 4 * pi / np.inner(qG, qG) * chi0_wGG[iw,iG] / self.vol
                if iw == 0:
                    self.eMRPA_GG[iG] = tmp[iG] - 4 * pi / np.inner(qG, qG) * chi0M_GG[iG] / self.vol


    def Finiteq(self):

        assert self.ncalc == 1
        calc = self.calc
        setups = calc.wfs.setups
        gd = calc.wfs.gd
        op = calc.wfs.symmetry.op_scc

        f_kn = self.f_kn
        e_kn = self.e_kn
        eta = self.eta
        qr = self.qr
        qq = self.qq
        q = self.q
        bzkpt_kG = calc.get_bz_k_points()
        IBZkpt_kG = calc.get_ibz_k_points()
        kq = self.find_kq(bzkpt_kG, q)

        chi0_wGG = np.zeros((self.Nw, self.npw, self.npw), dtype = complex)
        specfunc_wGG = np.zeros((self.NwS, self.npw, self.npw), dtype = complex)
        
        # calculate <phi_i | e**(-iq.r) | phi_j>
        phi_Gp = {}
        R_a = calc.atoms.positions / Bohr
        
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_Gp.has_key(Z):
                phi_Gp[Z] = ( self.two_phi_planewave_integrals(Z)
                                  * np.exp(-1j * np.inner(qq, R_a[a])) )
        print >> self.txt, 'phi_Gii obtained!'

        expqr_G = np.exp(-1j * self.qr)

        # defined Projectors 
        from gpaw.lfc import LocalizedFunctionsCollection as LFC
        pt = LFC(gd, [setup.pt_j for setup in setups],
                 calc.wfs.kpt_comm, dtype=calc.wfs.dtype, forces=True)
        spos_ac = calc.atoms.get_scaled_positions()
        pt.set_k_points(calc.get_bz_k_points())
        pt.set_positions(spos_ac)
        
        # calculate chi0
        for k in range(self.kstart, self.kend):
            t1 = time()

            if op is None:
                assert IBZkpt_kG.shape[0] == bzkpt_kG.shape[0]
                ibzkpt1, iop1 = k, 0
                ibzkpt2, iop2 = kq[k], 0
                op = np.zeros((1, 3, 3), dtype=int)
                op[0] = np.eye(3, dtype=int)
            else:
                ibzkpt1, iop1 = self.find_ibzkpt(op, IBZkpt_kG, bzkpt_kG[k])
                ibzkpt2, iop2 = self.find_ibzkpt(op, IBZkpt_kG, bzkpt_kG[kq[k]])
            
            rho_Gnn = np.zeros((self.npw, self.nband, self.nband), dtype=complex)
            for n in range(self.nband):

                psit1old_G = calc.wfs.kpt_u[ibzkpt1].psit_nG[n]
                psit1new_G = self.symmetrize_wavefunction(psit1old_G, op[iop1], IBZkpt_kG[ibzkpt1])

                P1_ai = pt.dict()
                pt.integrate(psit1new_G, P1_ai, k)
                
                psit1_G = psit1new_G.conj() * expqr_G
                
                for m in range(self.nband):
                    if  np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > 1e-8:

                        psit2old_G = calc.wfs.kpt_u[ibzkpt2].psit_nG[m]
                        psit2_G = self.symmetrize_wavefunction(psit2old_G, op[iop2], IBZkpt_kG[ibzkpt2])

                        P2_ai = pt.dict()
                        pt.integrate(psit2_G, P2_ai, kq[k])
                        
                        psit2_G *= psit1_G

                        # fft
                        tmp_G = np.fft.fftn(psit2_G) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex[iG]
                            rho_Gnn[iG, n, m] = tmp_G[index[0], index[1], index[2]]

                            # PAW correction
                        for a, id in enumerate(setups.id_a):
                            Z, type, basis = id
                            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                            rho_Gnn[:, n, m] += np.dot(phi_Gp[Z], P_p)

            t2 = time()
            #print  >> self.txt,'Time for density matrix:', t2 - t1, 'seconds'

            if not self.HilbertTrans:
                # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
                C_nn = np.zeros((self.nband, self.nband), dtype=complex)
                for iw in range(self.Nw):
                    w = iw * self.dw
                    for n in range(self.nband):
                        for m in range(self.nband):
                            if  np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > 1e-8:
                                C_nn[n, m] = (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) / (
                                 w + e_kn[ibzkpt1, n] - e_kn[ibzkpt2, m] + 1j * eta)
                
                    # get chi0(G=0,G'=0,w)
                    for iG in range(self.npw):
                        for jG in range(self.npw):
                            chi0_wGG[iw,iG,jG] += (rho_Gnn[iG] * C_nn * rho_Gnn[jG].conj()).sum()
            else:
            # calculate spectral function
                for n in range(self.nband):
                    for m in range(self.nband):
                        focc = f_kn[ibzkpt1,n] - f_kn[ibzkpt2,m]

                        if focc > 1e-8:
                            w0 = e_kn[ibzkpt2,m] - e_kn[ibzkpt1,n]
 
                            tmp_GG = focc * np.outer(rho_Gnn[:,n,m], rho_Gnn[:,n,m].conj())
                
                            # calculate delta function
                            deltaw = self.delta_function(w0, self.dw, self.NwS, self.sigma)
                            for wi in range(self.NwS):
                                if deltaw[wi] > 1e-8:
                                    specfunc_wGG[wi] += tmp_GG * deltaw[wi]

            t4 = time()
            #print  >> self.txt,'Time for spectral function loop:', t4 - t2, 'seconds'
            
            print >> self.txt, 'finished k', k

        comm = self.comm
 
        # Hilbert Transform
        if not self.HilbertTrans:
            comm.sum(chi0_wGG)
        else:
            comm.sum(specfunc_wGG)
            chi0_wGG = self.hilbert_transform(specfunc_wGG)

        tmp = np.eye(self.npw, self.npw)        
        for iw in range(self.Nw):
            for iG in range(self.npw):
                qG = np.array([np.inner(self.q + self.Gvec[iG],
                                       self.bcell[:,i]) for i in range(3)])
                self.eRPA_wGG[iw,iG] =  tmp[iG] - 4 * pi / np.inner(qG, qG) * chi0_wGG[iw,iG] / self.vol


    def find_ibzkpt(self, symrel, kpt_IBZkG, kptBZ):

        find = False
        
        for ioptmp in range(symrel.shape[0]):
            for i in range(kpt_IBZkG.shape[0]):
                tmp = np.inner(symrel[ioptmp], kpt_IBZkG[i])
                if (np.abs(tmp - kptBZ) < 1e-8).all():
                    ibzkpt = i
                    iop = ioptmp
                    find = True
                    break
            if find == True:
                break
    
        return ibzkpt, iop


    def symmetrize_wavefunction(self, a_g, op_cc, kpt):

        if (np.abs(op_cc - np.eye(3,dtype=int)) < 1e-10).all():
            return a_g
        else:
            import _gpaw
            b_g = np.zeros_like(a_g)
            _gpaw.symmetrize_wavefunction(a_g, b_g, op_cc.T.copy(), kpt)
    
        return b_g

        
    def rotate_wfs(self, op_cc, psi_old, kpt):
    
        nG = psi_old.shape
        psi_new = np.zeros_like(psi_old)
        
        for i in range(nG[0]):
            for j in range(nG[1]):
                for k in range(nG[2]):
                    rold = np.array([i,j,k])
        
                    rnew = np.inner(op_cc, rold)
                    assert rnew.dtype == int
        
                    R = np.zeros(3)
                    for id, RR in enumerate(rnew):
        
                        R[id] = RR / nG[id]
                        rnew[id] -= R[id] * nG[id] 
    
                    ii,jj,kk = rnew
        
                    phase = np.exp(1j*2.*pi*np.inner(kpt,R))
    
                    psi_new[ii,jj,kk] = psi_old[i,j,k] * phase
        return psi_new
     

    def find_kq(self, bzkpt_kG, q):
        """Find the index of k+q for all kpoints in BZ."""

        kq = np.zeros(self.nkpt, dtype=int)
        assert self.nkptxyz is not None
        nkptxyz = self.nkptxyz
        dk = 1. / nkptxyz 
        kmax = (nkptxyz - 1) * dk / 2.
        N = np.zeros(3, dtype=int)

        for k in range(self.nkpt):
            kplusq = bzkpt_kG[k] + q
            for dim in range(3):
                if kplusq[dim] > 0.5: # 
                    kplusq[dim] -= 1.
                elif kplusq[dim] < -0.5:
                    kplusq[dim] += 1.

                N[dim] = int(np.round((kplusq[dim] + kmax[dim])/dk[dim]))

            kq[k] = N[2] + N[1] * nkptxyz[2] + N[0] * nkptxyz[2]* nkptxyz[1]

            tmp = bzkpt_kG[kq[k]]
            if (abs(kplusq - tmp)).sum() > 1e-8:
                print k, kplusq, tmp
                raise ValueError('k+q index not correct!')

        return kq


    def delta_function(self, x0, dx, Nx, sigma):

        deltax = np.zeros(Nx)
        for i in range(Nx):
            deltax[i] = np.exp(-(i * dx - x0)**2/(4. * sigma))
        return deltax / (2. * sqrt(pi * sigma))



    def hilbert_transform(self, specfunc_wGG):

        tmp_ww = np.zeros((self.Nw, self.NwS), dtype=complex)

        eta = self.eta
        for iw in range(self.Nw):
            w = iw * self.dw
            for jw in range(self.NwS):
                ww = jw * self.dw 
                tmp_ww[iw, jw] = 1. / (w - ww + 1j*eta) - 1. / (w + ww + 1j*eta)

        chi0_wGG = gemmdot(tmp_ww, specfunc_wGG, beta = 0.)

        return chi0_wGG * self.dw



    def check_ortho(self, calc, psit_knG):
        # Check the orthonormalization of wfs
        gd = calc.wfs.gd
        setups = calc.wfs.setups
        rho_nn = np.zeros((self.nband, self.nband))

        phi_ii = {}
        
        for a in range(len(setups)):
            phi_p = setups[a].Delta_pL[:,0].copy()
            phi_ii[a] = unpack(phi_p) * sqrt(4*pi)

        for k in range(self.nkpt):
            P_ani = calc.wfs.kpt_u[k].P_ani
            for n in range(self.nband):
                for m in range(self.nband):
                    rho_nn[n, m] = gd.integrate(psit_knG[k,n].conj() * psit_knG[k,m])

                    for a in range(len(setups)):
                        P_ii = np.outer(P_ani[a][n].conj(), P_ani[a][m])
                        rho_nn[n, m] += (P_ii * phi_ii[a]).sum()
                    #print 'after PAW', (n, m), rho_nn[n, m]
                    if n == m and np.abs(rho_nn[n, m] -1) > 1e-10:
                        print 'after PAW', (n, m), rho_nn[n, m]
                    if n != m and np.abs(rho_nn[n, m]) > 1e-10:
                        print 'after PAW', (n, m), rho_nn[n, m]

        return


    def get_primitive_cell(self):

        a = self.acell

        self.vol = np.abs(np.dot(a[0],np.cross(a[1],a[2])))
        self.BZvol = (2. * pi)**3 / self.vol

        b = np.linalg.inv(a.T)
    
        self.bcell = 2 * pi * b

        assert np.abs((np.dot(self.bcell.T, a) - 2.*pi*np.eye(3)).sum()) < 1e-10

        return


    def two_phi_planewave_integrals(self, Z):

        xcfunc = XCFunctional('LDA',nspins=1)
        symbol = chemical_symbols[Z]
        data = SetupData(symbol,'LDA')
        s = Setup(data,xcfunc,lmax=2)

        # radial grid stuff
        ng = s.ng
        g = np.arange(ng, dtype=float)
        r_g = s.beta * g / (ng - g) 
        dr_g = s.beta * ng / (ng - g)**2
        r2dr_g = r_g **2 * dr_g
        gcut2 = s.gcut2
            
        # Obtain the phi_j and phit_j
        phi_jg = []
        phit_jg = []
        
        for (phi_g, phit_g) in zip(s.data.phi_jg, s.data.phit_jg):
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.
            phi_jg.append(phi_g)
            phit_jg.append(phit_g)

        # Construct L (l**2 + m) and j (nl) index
        L_i = []
        j_i = []
        lmax = 0 
        for j, l in enumerate(s.l_j):
            for m in range(2 * l + 1):
                L_i.append(l**2 + m)
                j_i.append(j)
                if l > lmax:
                    lmax = l
        ni = len(L_i)
        lmax = 2 * lmax + 1

        # Initialize        
        R_jj = np.zeros((s.nj, s.nj))
        R_ii = np.zeros((ni, ni))
        phi_Gii = np.zeros((self.npw, ni, ni), dtype=complex)
        j_lg = np.zeros((lmax, ng))
   
        # Store (phi_j1 * phi_j2 - phit_j1 * phit_j2 ) for further use
        tmp_jjg = np.zeros((s.nj, s.nj, ng))
        for j1 in range(s.nj):
            for j2 in range(s.nj): 
                tmp_jjg[j1, j2] = phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2]

        # Loop over G vectors
        Gvec = self.Gvec
        for iG in range(self.npw):
            kk = np.array([np.inner(self.q + Gvec[iG], self.bcell[:,i]) for i in range(3)])
            k = np.sqrt(np.inner(kk, kk)) # calculate length of q+G
            
            # Calculating spherical bessel function
            for ri in range(ng):
                j_lg[:,ri] = sph_jn(lmax - 1,  k*r_g[ri])[0]

            for li in range(lmax):
                # Radial part 
                for j1 in range(s.nj):
                    for j2 in range(s.nj): 
                        R_jj[j1, j2] = np.dot(r2dr_g, tmp_jjg[j1, j2] * j_lg[li])

                for mi in range(2 * li + 1):
                    # Angular part
                    for i1 in range(ni):
                        L1 = L_i[i1]
                        j1 = j_i[i1]
                        for i2 in range(ni):
                            L2 = L_i[i2]
                            j2 = j_i[i2]
                            R_ii[i1, i2] =  G_LLL[L1, L2, li**2+mi]  * R_jj[j1, j2]

                    phi_Gii[iG] += R_ii * Y(li**2 + mi, kk[0], kk[1], kk[2]) * (-1j)**li
        
        phi_Gii *= 4 * pi

        return phi_Gii.reshape(self.npw, ni*ni)


    def two_phi_derivative(self, Z):

        # Create setup for a certain specie
        xcfunc = XCFunctional('LDA',nspins=1)
        symbol = chemical_symbols[Z]
        data = SetupData(symbol,'LDA')
        s = Setup(data,xcfunc,lmax=2)

        # radial grid stuff
        ng = s.ng
        g = np.arange(ng, dtype=float)
        r_g = s.beta * g / (ng - g) 
        dr_g = s.beta * ng / (ng - g)**2
        r2dr_g = r_g **2 * dr_g
        gcut2 = s.gcut2

        # Obtain the phi_j and phit_j
        phi_jg = []
        phit_jg = []
        
        for (phi_g, phit_g) in zip(s.data.phi_jg, s.data.phit_jg):
            phi_g = phi_g.copy()
            phit_g = phit_g.copy()
            phi_g[gcut2:] = phit_g[gcut2:] = 0.
            phi_jg.append(phi_g)
            phit_jg.append(phit_g)

        # Construct L (l**2 + m) and j (nl) index
        L_i = []
        j_i = []
        for j, l in enumerate(s.l_j):
            for m in range(2 * l + 1):
                L_i.append(l**2 + m)
                j_i.append(j)

        Lmax = s.Lmax
        nj = s.nj
        ni = len(L_i)
        nii = ni * (ni + 1) // 2
        dphidr_jg = np.zeros(np.shape(phi_jg))
        dphitdr_jg = np.zeros(np.shape(phit_jg))
        phi_ii = np.zeros((ni, ni))

        from gpaw.xc_correction import A_Liy
        from gpaw.sphere import Y_nL, points, weights
        from gpaw.grid_descriptor import RadialGridDescriptor

        rgd = RadialGridDescriptor(r_g, dr_g)
        ny = len(points)
        
        for j in range(nj):
            rgd.derivative(phi_jg[j], dphidr_jg[j])
            rgd.derivative(phit_jg[j], dphitdr_jg[j])
        ##second term
        for y in range(ny):
            Y_L = Y_nL[y]
            weight = weights[y]
            for i1 in range(ni):
                L1 = L_i[i1]
                j1 = j_i[i1]
                for i2 in range(ni):
                    L2 = L_i[i2]
                    j2 = j_i[i2]

                    c = Y_L[L1]*Y_L[L2] # c is a number
                    temp  = c * ( phi_jg[j1] *  dphidr_jg[j2]
                                  - phit_jg[j1] *  dphitdr_jg[j2] )
                    phi_g = ( temp * self.qq[0] + temp * self.qq[1]
                                                + temp * self.qq[2] )
                    
                    A_Li = A_Liy[:Lmax, :, y]
                    temp = ( A_Li[L2, 0] * self.qq[0] + A_Li[L2, 1] * self.qq[1]
                             + A_Li[L2, 2] * self.qq[2] ) * Y_L[L1]
                    temp *= phi_jg[j1] * phi_jg[j2] - phit_jg[j1] * phit_jg[j2] 
                    temp[1:] /= r_g[1:]
                    temp[0] = temp[1]
                    phi_g += temp

                    phi_ii[i1, i2] += rgd.integrate(phi_g) * weight
        
        return phi_ii * (-1j)


    def print_stuff(self):

        txt = self.txt
        print >> txt
        print >> txt, 'Parameters used:'

        print >> txt 
        print >> txt, 'Number of bands:', self.nband
        print >> txt, 'Number of kpoints:', self.nkpt
        print >> txt, 'Unit cell (a.u.):'
        print >> txt, self.acell
        print >> txt, 'Reciprocal cell (1/a.u.)'
        print >> txt, self.bcell
        print >> txt, 'Volome of cell (a.u.**3):', self.vol
        print >> txt, 'BZ volume (1/a.u.**3):', self.BZvol

        print >> txt 
        print >> txt, 'Number of frequency points:', self.Nw
        print >> txt, 'Number of Grid points / G-vectors, and in total:', self.nG, self.nG0
        print >> txt, 'Grid spacing (a.u.):', self.h_c

        print >> txt 
        print >> txt, 'q in reduced coordinate:', self.q
        print >> txt, 'q in cartesian coordinate (1/A):', self.qq / Bohr
        print >> txt, '|q| (1/A):', sqrt(np.inner(self.qq / Bohr, self.qq / Bohr))

        print >> txt, 'Planewave cutoff energy (eV):', self.Ecut * Hartree
        print >> txt, 'Number of planewave used:', self.npw

        print >> txt, 'Use Hilbert Transform:', self.HilbertTrans

        print >> txt
        print >> txt, 'Memory usage estimation:'
        print >> txt, '    eRPA_wGG    :', self.Nw * self.npw**2 * 8. / 1024**2, 'M'
        print >> txt, '    chi0_wGG    :', self.Nw * self.npw**2 * 8. / 1024**2, 'M'
        print >> txt, '    specfunc_wGG:', self.NwS *self.npw**2 * 8. / 1024**2, 'M'
        print >> txt


    def memory_usage(self, a):
        assert type(a) == np.ndarray
        
        return a.itemsize * a.size /1024**2 # 'Megabyte'


    def check_sum_rule(self):

        N1 = N2 = 0
        for iw in range(self.Nw):
            w = iw * self.dw
            N1 += np.imag(self.eRPANLF_w[iw]) * w
            N2 += np.imag(self.eRPALFC_w[iw]) * w
        N1 *= self.dw * self.vol / (2 * pi**2)
        N2 *= self.dw * self.vol / (2 * pi**2)
        
        print >> self.txt, 'sum rule:'
        nv = self.nvalence
        print >> self.txt, 'Without local field correction, N1 = ', N1, (N1 - nv) / nv * 100, '% error'
        print >> self.txt, 'Include local field correction, N2 = ', N2, (N2 - nv) / nv * 100, '% error'


    def get_macroscopic_dielectric_constant(self):
        eMicro = self.eMRPA_GG[0, 0]
        eMacro = 1. / np.linalg.inv(self.eMRPA_GG)[0, 0]

        return np.real(eMicro), np.real(eMacro)
        

    def get_dielectric_function(self):

        self.eRPALFC_w = np.zeros(self.Nw, dtype = complex)
        self.eRPANLF_w = np.zeros(self.Nw, dtype = complex)
        
        for iw in range(self.Nw):
            tmp = self.eRPA_wGG[iw]
            self.eRPALFC_w[iw] = 1. / np.linalg.inv(tmp)[0, 0]
            self.eRPANLF_w[iw] = tmp[0, 0]    
        return 


    def get_absorption_spectrum(self, filename='Absorption'):
        self.get_dielectric_function()

        e1 = self.eRPANLF_w
        e2 = self.eRPALFC_w
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, np.real(e1[iw]), np.imag(e1[iw]),np.real(e2[iw]), np.imag(e2[iw])
            f.close()

        # Wait for I/O to finish
        self.comm.barrier()    


    def get_EELS_spectrum(self, filename='EELS'):
        
        self.get_dielectric_function()
        
        e1 = self.eRPANLF_w
        e2 = self.eRPALFC_w
        if rank == 0:
            f = open(filename,'w')
            for iw in range(self.Nw):
                energy = iw * self.dw * Hartree
                print >> f, energy, -np.imag(1./e1[iw]), -np.imag(1./e2[iw])
            f.close()

        # Wait for I/O to finish
        self.comm.barrier()


    def get_jdos(self, f_kn, e_kn, kq, dw, Nw, sigma):
        """Calculate Joint density of states"""

        # self.f_kn, self.e_kn, self.dw, self.Nw, self.sigma, self.kstart, self.kend

        JDOS_w = np.zeros(Nw)
        nkpt = f_kn.shape[0]
        nband = f_kn.shape[1]

        for k in range(nkpt):
            for n in range(nband):
                for m in range(nband):
                    focc = f_kn[k, n] - f_kn[kq[k], m]
                    w0 = e_kn[kq[k], m] - e_kn[k, n]
                    if focc > 0 and w0 >= 0:
                        deltaw = self.delta_function(w0, dw, Nw, sigma)
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

        w /= Hartree 
        iw = int(np.round(w / self.dw))
        print >> self.txt, 'Calculating Induced density at q, w (iw)'
        print >> self.txt, q, w*Hartree, iw

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
        tmp = self.eRPA_wGG[iw]
        chi_G = (np.linalg.inv(tmp)[:, 0] - delta_G) * coef_G

        from ase.parallel import paropen
        f = paropen('chi_G'+str(iw), 'w')
        for iG in range(self.npw):
            print >> f, np.real(chi_G[iG]), np.imag(chi_G[iG])
        f.close()

        # Wait for I/O to finish
        self.comm.barrier()  

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


    def set_Gvectors(self):

        # Refer to R.Martin P85
        Gcut = sqrt(2*self.Ecut)
        Gmax = np.zeros(3, dtype=int)
        for i in range(3):
            a = self.acell[i]
            Gmax[i] = sqrt(a[0]**2 + a[1]**2 + a[2]**2) * Gcut/ (2*pi)
         
        Nmax = 2 * Gmax + 1
        
        m = {}
        for dim in range(3):
            m[dim] = np.zeros(Nmax[dim],dtype=int)
            for i in range(Nmax[dim]):
                m[dim][i] = i
                if m[dim][i] > np.int(Gmax[dim]):
                    m[dim][i] = i- Nmax[dim]       

        G = np.zeros((Nmax[0]*Nmax[1]*Nmax[2],3),dtype=int)
        n = 0
        for i in range(Nmax[0]):
            for j in range(Nmax[1]):
                for k in range(Nmax[2]):
                    tmp = np.array([m[0][i], m[1][j], m[2][k]])
                    tmpG = np.array([np.inner(tmp, self.bcell[:,ii]) for ii in range(3)])
                    Gmod = sqrt(tmpG[0]**2 + tmpG[1]**2 + tmpG[2]**2)
                    if Gmod < Gcut:
                        G[n] = tmp
                        n += 1
        self.npw = n
        self.Gvec = G[:n]

        Gindex = {}
        id = np.zeros(3, dtype=int)

        for iG in range(self.npw):
            Gvec = self.Gvec[iG]
            for dim in range(3):
                if Gvec[dim] >= 0:
                    id[dim] = Gvec[dim]
                else:
                    id[dim] = self.nG[dim] - np.abs(Gvec[dim])
            Gindex[iG] = np.array(id)

        self.Gindex = Gindex        
        
        return


    def fxc(self, n):
        
        name = self.xc
        nspins = self.nspin

        libxc = XCFunctional(name, nspins)
       
        N = n.shape
        n = np.ravel(n)
        fxc = np.zeros_like(n)

        libxc.calculate_fxc_spinpaired(n, fxc)
        return np.reshape(fxc, N)


    def calculate_Kxc(self, gd, nt_G):
        # Currently without PAW correction
        
        Kxc_GG = np.zeros((self.npw, self.npw), dtype = complex)
        Gvec = self.Gvec

        fxc_G = self.fxc(nt_G)

        for iG in range(self.npw):
            for jG in range(self.npw):
                dG = np.array([np.inner(Gvec[iG] - Gvec[jG],
                              self.bcell[:,i]) for i in range(3)])
                dGr = np.inner(dG, self.r)
                Kxc_GG[iG, jG] = gd.integrate(np.exp(-1j * dGr) * fxc_G)
                
        return Kxc_GG / self.vol
                
