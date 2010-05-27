import sys
from time import time, ctime
import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw.utilities import unpack, devnull
from gpaw.mpi import world, rank, size
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.fd_operators import Gradient
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.response.symmetrize import find_kq, find_ibzkpt, symmetrize_wavefunction
from gpaw.response.math_func import delta_function, hilbert_transform, \
     two_phi_planewave_integrals

class CHI:
    """This class is a calculator for the linear density response function.

    Parameters:

        nband: int
            Number of bands.
        wmax: float
            Maximum energy for spectrum.
        dw: float
            Frequency interval.
        wlist: tuple
            Frequency points. 
        q: ndarray
            Momentum transfer in reduced coordinate.
        Ecut: ndarray
            Planewave cutoff energy.
        eta: float
            Spectrum broadening factor.
        sigma: float
            Width for delta function. 
    """    
        
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
                 HilbertTrans=True,
                 OpticalLimit=False):
        
        self.xc = 'LDA'
        self.nspin = 1
        
        self.comm = world
        self.output_init()

        self.calc = calc
        self.nband = nband
        self.q = q
        
        self.wmin = 0.
        self.wmax = wmax
        self.dw = dw
        self.wlist = wlist
        self.eta = eta 
        self.sigma = sigma
        self.Ecut = Ecut
        self.HilbertTrans = HilbertTrans
        self.OpticalLimit = OpticalLimit
        if wlist is not None:
            self.HilbertTrans = False

        
    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------')
        self.printtxt('Response function calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        # Frequency init
        self.wmin = 0
        self.wmax  /= Hartree
        self.wcut = self.wmax + 5. / Hartree
        self.dw /= Hartree
        self.Nw = int((self.wmax - self.wmin) / self.dw) + 1
        self.NwS = int((self.wcut - self.wmin) / self.dw) + 1
        self.eta /= Hartree
        self.Ecut /= Hartree

        if not self.HilbertTrans:
            self.Nw = len(self.wlist)
            assert self.wlist is not None

        calc = self.calc
        gd = calc.wfs.gd

        # kpoint init
        self.bzkpt_kG = calc.get_bz_k_points()
        self.Ibzkpt_kG = calc.get_ibz_k_points()
        self.nkpt = self.bzkpt_kG.shape[0]

        # parallize in kpoints
        self.nkpt_local = int(self.nkpt / size)

        self.kstart = rank * self.nkpt_local
        self.kend = (rank + 1) * self.nkpt_local
        if rank == size - 1:
            self.kend = self.nkpt                

        # band init
        if self.nband is None:    
            self.nband = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence

        assert calc.wfs.nspins == 1

        # cell init
        self.acell = calc.atoms.cell / Bohr
        self.bcell, self.vol, self.BZvol = get_primitive_cell(self.acell)

        # grid init
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        self.h_c = gd.h_cv

        # obtain eigenvalues, occupations
        nibzkpt = self.Ibzkpt_kG.shape[0]
        kweight = calc.get_k_point_weights()

        self.e_kn = np.array([calc.get_eigenvalues(kpt=k)
                    for k in range(nibzkpt)]) / Hartree
        self.f_kn = np.array([calc.get_occupation_numbers(kpt=k) / kweight[k]
                    for k in range(nibzkpt)]) / self.nkpt

        # k + q init
        assert self.q is not None
        self.qq = np.inner(self.bcell.T, self.q)
        
        if self.OpticalLimit:
            kq = np.arange(self.nkpt)
            self.expqr_G = 1. 
        else:
            r = gd.get_grid_point_coordinates() # (3, nG)
            qr = np.inner(self.qq, r.T).T
            self.expqr_G = np.exp(-1j * qr)
            del r, qr
            kq = find_kq(self.bzkpt_kG, self.q)
        self.kq = kq

        # Plane wave init
        self.npw, self.Gvec, self.Gindex = set_Gvectors(self.acell, self.bcell, self.nG, self.Ecut)


        # Projectors init
        setups = calc.wfs.setups
        pt = LFC(gd, [setup.pt_j for setup in setups],
                 calc.wfs.kpt_comm, dtype=calc.wfs.dtype, forces=True)
        spos_ac = calc.atoms.get_scaled_positions()
        for ia in range(spos_ac.shape[0]):
            for idim in range(3):
                if spos_ac[ia,idim] == 1.:
                    spos_ac[ia,idim] -= 1.
        pt.set_k_points(self.bzkpt_kG)
        pt.set_positions(spos_ac)
        self.pt = pt

        # Symmetry operations init
        usesymm = calc.input_parameters.get('usesymm')
        if usesymm == None:
            op = (np.eye(3, dtype=int),)
        elif usesymm == False:
            op = (np.eye(3, dtype=int), -np.eye(3, dtype=int))
        else:
            op = calc.wfs.symmetry.op_scc
        self.op = op
        

#        nt_G = calc.density.nt_sG[0] # G is the number of grid points
#        self.Kxc_GG = self.calculate_Kxc(calc.wfs.gd, nt_G)          # G here is the number of plane waves

        # Printing calculation information
        self.print_stuff()

        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        phi_Gp = {}
        phi_aGp = []
        R_a = calc.atoms.positions / Bohr

        kk_Gv = np.zeros((self.npw, 3))
        for iG in range(self.npw):
            kk_Gv[iG] = np.inner(self.bcell.T, self.q + self.Gvec[iG])

        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_Gp.has_key(Z):
                phi_Gp[Z] = two_phi_planewave_integrals(kk_Gv, setups[a])
            phi_aGp.append(phi_Gp[Z])

            for iG in range(self.npw):
                phi_aGp[a][iG] *= np.exp(-1j * np.inner(kk_Gv[iG], R_a[a]))

        if self.OpticalLimit:
            for a, id in enumerate(setups.id_a):
                # G == 0 part
                nabla_iiv = setups[a].nabla_iiv
                phi_aGp[a][0] = -1j * (np.dot(nabla_iiv, self.qq)).ravel()
            
        self.phi_aGp = phi_aGp
        self.printtxt('')
        self.printtxt('Finished phi_Gp !')
            
        
        return


    def calculate(self):

        calc = self.calc
        gd = calc.wfs.gd
        sdisp_cd = gd.sdisp_cd
        IBZkpt_kG = self.Ibzkpt_kG
        bzkpt_kG = self.bzkpt_kG
        kq = self.kq
        pt = self.pt
        f_kn = self.f_kn
        e_kn = self.e_kn

        # Matrix init
        chi0_wGG = np.zeros((self.Nw, self.npw, self.npw), dtype = complex)
        if self.HilbertTrans:
            specfunc_wGG = np.zeros((self.NwS, self.npw, self.npw), dtype = complex)

        if self.OpticalLimit:
            d_c = [Gradient(gd, i, dtype=complex).apply for i in range(3)]
            dpsit_G = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

        for k in range(self.kstart, self.kend):
            ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op, IBZkpt_kG, bzkpt_kG[k])
            if self.OpticalLimit:
                ibzkpt2, iop2, timerev2 = ibzkpt1, iop1, timerev1
            else:
                ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op, IBZkpt_kG, bzkpt_kG[kq[k]])
            
            rho_Gnn = np.zeros((self.npw, self.nband, self.nband), dtype=complex)
            for n in range(self.nband):

                psitold_G =  calc.wfs.kpt_u[ibzkpt1].psit_nG[n]
                psit1new_G = symmetrize_wavefunction(psitold_G, self.op[iop1], IBZkpt_kG[ibzkpt1],
                                                          bzkpt_kG[k], timerev1)        
                     
                P1_ai = pt.dict()
                pt.integrate(psit1new_G, P1_ai, k)
                
                psit1_G = psit1new_G.conj() * self.expqr_G
                
                for m in range(self.nband):
                    if  np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > 1e-8:

                        psitold_G =  calc.wfs.kpt_u[ibzkpt2].psit_nG[m]
                        psit2_G = symmetrize_wavefunction(psitold_G, self.op[iop2], IBZkpt_kG[ibzkpt2],
                                                               bzkpt_kG[kq[k]], timerev2)
                        
                        P2_ai = pt.dict()
                        pt.integrate(psit2_G, P2_ai, kq[k])
                        
                        # fft
                        tmp_G = np.fft.fftn(psit2_G*psit1_G) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex[iG]
                            rho_Gnn[iG, n, m] = tmp_G[index[0], index[1], index[2]]

                        if self.OpticalLimit:
                            phase_cd = np.exp(2j * pi * sdisp_cd * bzkpt_kG[kq[k], :, np.newaxis])
                            for ix in range(3):
                                d_c[ix](psit2_G, dpsit_G, phase_cd)
                                tmp[ix] = gd.integrate(psit1_G * dpsit_G)
                            rho_Gnn[0, n, m] = -1j * np.inner(self.qq, tmp) 

                        # PAW correction
                        for a, id in enumerate(calc.wfs.setups.id_a):                            
                            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                            rho_Gnn[:, n, m] += np.dot(self.phi_aGp[a], P_p)

                        if self.OpticalLimit:
                            rho_Gnn[0, n, m] /= e_kn[ibzkpt2, m] - e_kn[ibzkpt1, n]
                            
            t2 = time()
            #print  >> self.txt,'Time for density matrix:', t2 - t1, 'seconds'


            if not self.HilbertTrans:
                # construct (f_nk - f_n'k+q) / (w + e_nk - e_n'k+q + ieta )
                C_nn = np.zeros((self.nband, self.nband), dtype=complex)
                for iw in range(self.Nw):
                    w = self.wlist[iw] / Hartree 
                    for n in range(self.nband):
                        for m in range(self.nband):
                            if  np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > 1e-8:
                                C_nn[n, m] = (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) / (
                                 w + e_kn[ibzkpt1, n] - e_kn[ibzkpt2, m] + 1j * self.eta)
                
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
                            deltaw = delta_function(w0, self.dw, self.NwS, self.sigma)
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
            chi0_wGG = hilbert_transform(specfunc_wGG, self.Nw, self.dw, self.eta)
            del specfunc_wGG

        self.chi0_wGG = chi0_wGG / self.vol
        
        return 


    def output_init(self):
        if rank == 0:
            self.txt = sys.stdout #open('out.txt','w')
        else:
            sys.stdout = devnull
            self.txt = devnull    

    def parallel_init(self):
        
        pass
        
    
    def printtxt(self, text):
        print >> self.txt, text


    def print_stuff(self):

        printtxt = self.printtxt
        printtxt('')
        printtxt('Parameters used:')
        printtxt('')
        printtxt('Number of bands: %d' %(self.nband) )
        printtxt('Number of kpoints: %d' %(self.nkpt) )
        printtxt('Unit cell (a.u.):')
        printtxt(self.acell)
        printtxt('Reciprocal cell (1/a.u.)')
        printtxt(self.bcell)
        printtxt('Volome of cell (a.u.**3): %f' %(self.vol) )
        printtxt('BZ volume (1/a.u.**3): %f' %(self.BZvol) )
        printtxt('')
        printtxt('Number of frequency points: %d' %(self.Nw) )
        printtxt('Number of Grid points / G-vectors, and in total: (%d %d %d), %d'
                  %(self.nG[0], self.nG[1], self.nG[2], self.nG0))
        printtxt('Grid spacing (a.u.):')
        printtxt(self.h_c)
        printtxt('')
        if self.OpticalLimit:
            printtxt('Optical limit calculation ! (q=0.00001)')
        else:
            printtxt('q in reduced coordinate: (%f %f %f)' %(self.q[0], self.q[1], self.q[2]) )
            printtxt('q in cartesian coordinate (1/A): (%f %f %f) '
                  %(self.qq[0] / Bohr, self.qq[1] / Bohr, self.qq[2] / Bohr) )
            printtxt('|q| (1/A): %f' %(sqrt(np.inner(self.qq / Bohr, self.qq / Bohr))) )
        printtxt('')
        printtxt('Planewave cutoff energy (eV): %f' %(self.Ecut * Hartree) )
        printtxt('Number of planewave used: %d' %(self.npw) )
        printtxt('')
        printtxt('Use Hilbert Transform: %s' %(self.HilbertTrans) )
        printtxt('')
        printtxt('Memory usage estimation:')
        printtxt('     eRPA_wGG    : %f M' %(self.Nw * self.npw**2 * 8. / 1024**2) )
        printtxt('     chi0_wGG    : %f M' %(self.Nw * self.npw**2 * 8. / 1024**2) )
        printtxt('     specfunc_wGG: %f M' %(self.NwS *self.npw**2 * 8. / 1024**2) )
        
