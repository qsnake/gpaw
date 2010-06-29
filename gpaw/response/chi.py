import sys
from time import time, ctime
import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw import GPAW, extra_parameters
from gpaw.utilities import unpack, devnull
from gpaw.utilities.blas import gemmdot, gemv, scal, axpy
from gpaw.mpi import world, rank, size, serial_comm
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.fd_operators import Gradient
from gpaw.response.cell import get_primitive_cell, set_Gvectors
from gpaw.response.symmetrize import find_kq, find_ibzkpt, symmetrize_wavefunction
from gpaw.response.math_func import delta_function, hilbert_transform, \
     two_phi_planewave_integrals
from gpaw.response.parallel import set_communicator, \
     parallel_partition, SliceAlongFrequency, SliceAlongOrbitals
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.memory import maxrss

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

        self.xc = 'LDA'
        self.nspin = 1

        self.txtname = txt
        self.output_init()

        if isinstance(calc, str):
            self.calc = GPAW(calc, txt=None)
        else:
            self.calc = calc

        self.nbands = nbands
        self.q_c = q

        self.w_w = w
        self.eta = eta
        self.ftol = ftol
        self.ecut = ecut
        self.hilbert_trans = hilbert_trans
        self.optical_limit = optical_limit
        self.kcommsize = kcommsize
        self.comm = world
        self.chi0_wGG = None


    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------')
        self.printtxt('Response function calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        # Frequency init
        if len(self.w_w) == 1:
            self.HilberTrans = False
            
        if self.hilbert_trans:
            self.dw = self.w_w[1] - self.w_w[0]
            assert ((self.w_w[1:] - self.w_w[:-1] - self.dw) < 1e-10).all() # make sure its linear w grid
            assert self.w_w.max() == self.w_w[-1]
            
            self.dw /= Hartree
            self.w_w  /= Hartree
            self.wmax = self.w_w[-1] 
            self.wcut = self.wmax + 5. / Hartree
            self.Nw  = int(self.wmax / self.dw) + 1
            self.NwS = int(self.wcut / self.dw) + 1
        else:
            self.Nw = len(self.w_w)
            self.NwS = 0
            self.dw = None
            
        self.eta /= Hartree
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

        assert calc.wfs.nspins == 1

        # cell init
        self.acell_cv = calc.atoms.cell / Bohr
        self.bcell_cv, self.vol, self.BZvol = get_primitive_cell(self.acell_cv)

        # grid init
        self.nG = calc.get_number_of_grid_points()
        self.nG0 = self.nG[0] * self.nG[1] * self.nG[2]
        gd = GridDescriptor(self.nG, calc.wfs.gd.cell_cv, pbc_c=True, comm=serial_comm)
        self.gd = gd        
        self.h_cv = gd.h_cv

        # obtain eigenvalues, occupations
        nibzkpt = self.ibzk_kc.shape[0]
        kweight_k = calc.get_k_point_weights()

        self.e_kn = np.array([calc.get_eigenvalues(kpt=k)
                    for k in range(nibzkpt)]) / Hartree
        self.f_kn = np.array([calc.get_occupation_numbers(kpt=k) / kweight_k[k]
                    for k in range(nibzkpt)]) / self.nkpt

        # k + q init
        assert self.q_c is not None
        self.qq_v = np.dot(self.q_c, self.bcell_cv)

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
        for ia in range(spos_ac.shape[0]):
            for idim in range(3):
                if spos_ac[ia,idim] == 1.:
                    spos_ac[ia,idim] -= 1.
        pt.set_k_points(self.bzk_kc)
        pt.set_positions(spos_ac)
        self.pt = pt

        # Symmetry operations init
        usesymm = calc.input_parameters.get('usesymm')
        if usesymm == None:
            op_scc = (np.eye(3, dtype=int),)
        elif usesymm == False:
            op_scc = (np.eye(3, dtype=int), -np.eye(3, dtype=int))
        else:
            op_scc = calc.wfs.symmetry.op_scc
        self.op_scc = op_scc


#        nt_G = calc.density.nt_sG[0] # G is the number of grid points
#        self.Kxc_GG = self.calculate_Kxc(calc.wfs.gd, nt_G)

        # Parallelization initialize
        self.parallel_init()

        # Printing calculation information
        self.print_stuff()

        if extra_parameters.get('df_dry_run'):
            raise SystemExit

        # For LCAO wfs
        calc.initialize_positions()
        self.printtxt('     GS calculator   : %f M / cpu' %(maxrss() / 1024**2))
        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        phi_Gp = {}
        phi_aGp = []
        R_a = calc.atoms.positions / Bohr

        kk_Gv = gemmdot(self.q_c + self.Gvec_Gc, self.bcell_cv.copy(), beta=0.0)
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not Z in phi_Gp:
                phi_Gp[Z] = two_phi_planewave_integrals(kk_Gv, setups[a])
            phi_aGp.append(phi_Gp[Z])

            for iG in range(self.npw):
                phi_aGp[a][iG] *= np.exp(-1j * np.dot(kk_Gv[iG], R_a[a]))

        # For optical limit, G == 0 part should change
        if self.optical_limit:
            for a, id in enumerate(setups.id_a):
                nabla_iiv = setups[a].nabla_iiv
                phi_aGp[a][0] = -1j * (np.dot(nabla_iiv, self.qq_v)).ravel()

        self.phi_aGp = phi_aGp
        self.printtxt('')
        self.printtxt('Finished phi_Gp !')
        self.printtxt('')

        return


    def calculate(self):
        """Calculate the non-interacting density response function. """

        calc = self.calc
        gd = self.gd
        sdisp_cd = gd.sdisp_cd
        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        pt = self.pt
        f_kn = self.f_kn
        e_kn = self.e_kn

        # Matrix init
        chi0_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype=complex)
        if self.hilbert_trans:
            specfunc_wGG = np.zeros((self.NwS_local, self.npw, self.npw), dtype = complex)

        # Prepare for the derivative of pseudo-wavefunction
        if self.optical_limit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_g = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

        rho_G = np.zeros(self.npw, dtype=complex)
        ibzkpt_kcomm = np.zeros(self.kcomm.size, dtype=int)
        t0 = time()

        for k in range(self.kstart, self.kend):

            # Find corresponding kpoint in IBZ
            ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[k])
            if self.optical_limit:
                ibzkpt2, iop2, timerev2 = ibzkpt1, iop1, timerev1
            else:
                ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op_scc, ibzk_kc, bzk_kc[kq_k[k]])

            for n in range(self.nbands):

                self.kcomm.all_gather(np.array([ibzkpt1]), ibzkpt_kcomm)

                if self.hilbert_trans:
                    if (f_kn[ibzkpt_kcomm, n] < self.ftol).all():
                        break

                if calc.wfs.world.size != 1:
                    for ikcomm in range(self.kcomm.size):
                        psit_g = self.get_wavefunction(ibzkpt_kcomm[ikcomm], n)
                        if self.kcomm.rank == ikcomm:
                            psitold_g = psit_g
                else:
                    psitold_g = calc.wfs._get_wave_function_array(ibzkpt1, n)

                psit1new_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop1], ibzk_kc[ibzkpt1],
                                                      bzk_kc[k], timerev1)

                P1_ai = pt.dict()
                pt.integrate(psit1new_g, P1_ai, k)

                psit1_g = psit1new_g.conj() * self.expqr_g

                for m in range(self.nbands):
                    
                    self.kcomm.all_gather(np.array([ibzkpt2]), ibzkpt_kcomm)
                    
		    if self.hilbert_trans:
			check_focc = (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol
                    else:
                        check_focc = np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol 
                    check_focc_all = np.zeros(self.kcomm.size, dtype=bool)
                    self.kcomm.all_gather(np.array([check_focc]), check_focc_all) 

                    if calc.wfs.world.size != 1:
                        for ikcomm in range(self.kcomm.size):
                            if check_focc_all[ikcomm]:
                                psit_g = self.get_wavefunction(ibzkpt_kcomm[ikcomm], m)
                                if self.kcomm.rank == ikcomm:
                                    psitold_g = psit_g
                    else:
                        if check_focc:
                            psitold_g = calc.wfs._get_wave_function_array(ibzkpt2, m)

                    if check_focc:
        
                        psit2_g = symmetrize_wavefunction(psitold_g, self.op_scc[iop2], ibzk_kc[ibzkpt2],
                                                           bzk_kc[kq_k[k]], timerev2)

                        P2_ai = pt.dict()
                        pt.integrate(psit2_g, P2_ai, kq_k[k])

                        # fft
                        tmp_g = np.fft.fftn(psit2_g*psit1_g) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex_G[iG]
                            rho_G[iG] = tmp_g[index[0], index[1], index[2]]

                        if self.optical_limit:
                            phase_cd = np.exp(2j * pi * sdisp_cd * bzk_kc[kq_k[k], :, np.newaxis])
                            for ix in range(3):
                                d_c[ix](psit2_g, dpsit_g, phase_cd)
                                tmp[ix] = gd.integrate(psit1_g * dpsit_g)
                            rho_G[0] = -1j * np.dot(self.qq_v, tmp)

                        # PAW correction
                        for a, id in enumerate(calc.wfs.setups.id_a):
                            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                            gemv(1.0, self.phi_aGp[a], P_p, 1.0, rho_G)

                        if self.optical_limit:
                            rho_G[0] /= e_kn[ibzkpt2, m] - e_kn[ibzkpt1, n]

                        rho_GG = np.outer(rho_G, rho_G.conj())
                        
                        if not self.hilbert_trans:
                            for iw in range(self.Nw_local):
                                w = self.w_w[iw + self.wstart] / Hartree
                                C =  (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) / (
                                     w + e_kn[ibzkpt1, n] - e_kn[ibzkpt2, m] + 1j * self.eta)
                                axpy(C, rho_GG, chi0_wGG[iw])
                        else:
                            focc = f_kn[ibzkpt1,n] - f_kn[ibzkpt2,m]
                            w0 = e_kn[ibzkpt2,m] - e_kn[ibzkpt1,n]
                            scal(focc, rho_GG)

                            # calculate delta function
                            w0_id = int(w0 / self.dw)
                            if w0_id + 1 < self.NwS:
                                # rely on the self.NwS_local is equal in each node!
                                if self.wScomm.rank == w0_id // self.NwS_local:
                                    alpha = (w0_id + 1 - w0/self.dw) / self.dw
                                    axpy(alpha, rho_GG, specfunc_wGG[w0_id % self.NwS_local] )

                                if self.wScomm.rank == (w0_id+1) // self.NwS_local:
                                    alpha =  (w0 / self.dw - w0_id) / self.dw
                                    axpy(alpha, rho_GG, specfunc_wGG[(w0_id+1) % self.NwS_local] )

#                            deltaw = delta_function(w0, self.dw, self.NwS, self.sigma)
#                            for wi in range(self.NwS_local):
#                                if deltaw[wi + self.wS1] > 1e-8:
#                                    specfunc_wGG[wi] += tmp_GG * deltaw[wi + self.wS1]

            self.kcomm.barrier()            
            if k == 0:
                dt = time() - t0
                totaltime = dt * self.nkpt_local
                self.printtxt('Finished k 0 in %f seconds, estimatied %f seconds left.' %(dt, totaltime))
                
            if rank == 0 and self.nkpt_local // 5 > 0:            
                if k > 0 and k % (self.nkpt_local // 5) == 0:
                    dt =  time() - t0
                    self.printtxt('Finished k %d in %f seconds, estimated %f seconds left.  '%(k, dt, totaltime - dt) )

        del rho_GG, rho_G
        # Hilbert Transform
        if not self.hilbert_trans:
            self.kcomm.sum(chi0_wGG)
        else:
            self.kcomm.sum(specfunc_wGG)

            coords = np.zeros(self.wScomm.size, dtype=int)
            nG_local = self.npw**2 // self.wScomm.size
            if self.wScomm.rank == self.wScomm.size - 1:
                nG_local = self.npw**2 - (self.wScomm.size - 1) * nG_local
            self.wScomm.all_gather(np.array([nG_local]), coords)

            specfunc_Wg = SliceAlongFrequency(specfunc_wGG, coords, self.wScomm)

            chi0_Wg = hilbert_transform(specfunc_Wg, self.Nw, self.dw, self.eta)[:self.Nw]

            self.comm.barrier()
            del specfunc_wGG, specfunc_Wg

            # redistribute chi0_wGG
            assert self.kcomm.size == size // self.wScomm.size
            Nwtmp1 = (rank % self.kcomm.size ) * self.Nw_local
            Nwtmp2 = (rank % self.kcomm.size + 1) * self.Nw_local

            chi0_wGG = SliceAlongOrbitals(chi0_Wg, coords, self.wScomm)[Nwtmp1:Nwtmp2]

            self.comm.barrier()
            del chi0_Wg

        self.chi0_wGG = chi0_wGG / self.vol

        return

    def get_wavefunction(self, k, n):

        mynu = len(self.calc.wfs.kpt_u)
        kpt_rank, u = divmod(k, mynu)
        band_rank, myn = self.calc.wfs.bd.who_has(n)

        psit_g = self.calc.wfs._get_wave_function_array(u, myn)
        psit_G = self.calc.wfs.gd.collect(psit_g, broadcast=True)

        return psit_G


    def output_init(self):


        if self.txtname is None:
            if rank == 0:
                self.txt = sys.stdout
            else:
                sys.stdout = devnull
                self.txt = devnull

        else:
            assert type(self.txtname) is str
            from ase.parallel import paropen
            self.txt = paropen(self.txtname,'w')


    def parallel_init(self):
        """Parallel initialization. By default, only use kcomm and wcomm.

        Parameters:

            kcomm:
                 kpoint communicator
            wScomm:
                 spectral function communicator
            wcomm:
                 frequency communicator
        """

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
            rank = world.rank
            self.comm = world
        else:
            from gpaw.mpi import world, rank, size

        wcommsize = int(self.NwS * self.npw**2 * 16. / 1024**2) // 1500 # megabyte
        wcommsize += 1
        if size < wcommsize:
            raise ValueError('Number of cpus are not enough ! ')
        if wcommsize > 1: # if matrix too large, overwrite kcommsize and distribute matrix
            while size % wcommsize != 0:
                wcommsize += 1
            self.kcommsize = size // wcommsize
            assert self.kcommsize * wcommsize == size
            if self.kcommsize < 1:
                raise ValueError('Number of cpus are not enough ! ')

        self.kcomm, self.wScomm, self.wcomm = set_communicator(world, rank, size, self.kcommsize)

        self.nkpt, self.nkpt_local, self.kstart, self.kend = parallel_partition(
                               self.nkpt, self.kcomm.rank, self.kcomm.size, reshape=False)

        self.NwS, self.NwS_local, self.wS1, self.wS2 = parallel_partition(
                               self.NwS, self.wScomm.rank, self.wScomm.size, reshape=True)

        if self.hilbert_trans:
            self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=True)
        else:
            if self.Nw // size > 1:
                self.Nw, self.Nw_local, self.wstart, self.wend =  parallel_partition(
                               self.Nw, self.wcomm.rank, self.wcomm.size, reshape=False)
            else:
                # if frequency point is too few, then dont parallelize
                self.wcomm = serial_comm
                self.wstart = 0
                self.wend = self.Nw
                self.Nw_local = self.Nw

        return


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
        printtxt('Grid spacing (a.u.):')
        printtxt(self.h_cv)
        printtxt('Number of Grid points / G-vectors, and in total: (%d %d %d), %d'
                  %(self.nG[0], self.nG[1], self.nG[2], self.nG0))
        printtxt('Volome of cell (a.u.**3)     : %f' %(self.vol) )
        printtxt('BZ volume (1/a.u.**3)        : %f' %(self.BZvol) )
        printtxt('')                         
        printtxt('Number of bands              : %d' %(self.nbands) )
        printtxt('Number of kpoints            : %d' %(self.nkpt) )
        printtxt('Planewave ecut (eV)          : %f' %(self.ecut * Hartree) )
        printtxt('Number of planewave used     : %d' %(self.npw) )
        printtxt('Broadening (eta)             : %f' %(self.eta * Hartree))
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        if self.hilbert_trans:
            printtxt('Number of specfunc points    : %d' % (self.NwS))
        printtxt('')
        if self.optical_limit:
            printtxt('Optical limit calculation ! (q=0.00001)')
        else:
            printtxt('q in reduced coordinate        : (%f %f %f)' %(self.q_c[0], self.q_c[1], self.q_c[2]) )
            printtxt('q in cartesian coordinate (1/A): (%f %f %f) '
                  %(self.qq_v[0] / Bohr, self.qq_v[1] / Bohr, self.qq_v[2] / Bohr) )
            printtxt('|q| (1/A)                      : %f' %(sqrt(np.dot(self.qq_v / Bohr, self.qq_v / Bohr))) )
        printtxt('')
        printtxt('Use Hilbert Transform: %s' %(self.hilbert_trans) )
        printtxt('')
        printtxt('Parallelization scheme:')
        printtxt('     Total cpus      : %d' %(self.comm.size))
        printtxt('     kpoint parsize  : %d' %(self.kcomm.size))
        printtxt('     specfunc parsize: %d' %(self.wScomm.size))
        printtxt('     w parsize       : %d' %(self.wcomm.size))
        printtxt('')
        printtxt('Memory usage estimation:')
        printtxt('     chi0_wGG        : %f M / cpu' %(self.Nw_local * self.npw**2 * 16. / 1024**2) )
        if self.hilbert_trans:
            printtxt('     specfunc_wGG    : %f M / cpu' %(self.NwS_local *self.npw**2 * 16. / 1024**2) )





