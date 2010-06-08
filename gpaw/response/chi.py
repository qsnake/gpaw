import sys
from time import time, ctime
import numpy as np
from math import sqrt, pi
from ase.units import Hartree, Bohr
from gpaw import extra_parameters
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
                 Ecut=10.,
                 eta=0.2,
                 sigma=1e-5,
                 ftol=1e-7,
                 txt=None,
                 HilbertTrans=True,
                 OpticalLimit=False,
                 kcommsize=None):

        self.xc = 'LDA'
        self.nspin = 1

        self.txtname = txt
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
        self.ftol = ftol
        self.Ecut = Ecut
        self.HilbertTrans = HilbertTrans
        self.OpticalLimit = OpticalLimit
        self.kcommsize = kcommsize
        self.comm = world
        if wlist is not None:
            self.HilbertTrans = False
        self.chi0_wGG = None


    def initialize(self):

        self.printtxt('')
        self.printtxt('-----------------------------------------')
        self.printtxt('Response function calculation started at:')
        self.starttime = time()
        self.printtxt(ctime())

        # Frequency init
        if self.HilbertTrans:
            self.wmin = 0
            self.wmax  /= Hartree
            self.wcut = self.wmax + 5. / Hartree
            self.dw /= Hartree
            self.Nw = int((self.wmax - self.wmin) / self.dw) + 1
            self.NwS = int((self.wcut - self.wmin) / self.dw) + 1
        else:
            self.Nw = len(self.wlist)
            self.NwS = 0
            assert self.wlist is not None
            
        self.eta /= Hartree
        self.Ecut /= Hartree

        calc = self.calc
        gd = calc.wfs.gd

        # kpoint init
        self.bzk_kv = calc.get_bz_k_points()
        self.ibzk_kv = calc.get_ibz_k_points()
        self.nkpt = self.bzk_kv.shape[0]
        self.ftol /= self.nkpt

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
        nibzkpt = self.ibzk_kv.shape[0]
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
            qr = gemmdot(self.qq, r, beta=0.0)
            self.expqr_G = np.exp(-1j * qr)
            del r, qr
            kq = find_kq(self.bzk_kv, self.q)
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
        pt.set_k_points(self.bzk_kv)
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
#        self.Kxc_GG = self.calculate_Kxc(calc.wfs.gd, nt_G)

        # Parallelization initialize
        self.parallel_init()

        # Printing calculation information
        self.print_stuff()

        if extra_parameters.get('df_dry_run'):
            raise SystemExit

        # PAW part init
        # calculate <phi_i | e**(-i(q+G).r) | phi_j>
        # G != 0 part
        phi_Gp = {}
        phi_aGp = []
        R_a = calc.atoms.positions / Bohr

        kk_Gv = gemmdot(self.q + self.Gvec, self.bcell.copy(), beta=0.0)
        for a, id in enumerate(setups.id_a):
            Z, type, basis = id
            if not phi_Gp.has_key(Z):
                phi_Gp[Z] = two_phi_planewave_integrals(kk_Gv, setups[a])
            phi_aGp.append(phi_Gp[Z])

            for iG in range(self.npw):
                phi_aGp[a][iG] *= np.exp(-1j * np.inner(kk_Gv[iG], R_a[a]))

        # For optical limit, G == 0 part should change
        if self.OpticalLimit:
            for a, id in enumerate(setups.id_a):
                nabla_iiv = setups[a].nabla_iiv
                phi_aGp[a][0] = -1j * (np.dot(nabla_iiv, self.qq)).ravel()

        self.phi_aGp = phi_aGp
        self.printtxt('')
        self.printtxt('Finished phi_Gp !')
        self.printtxt('')

        return


    def calculate(self):
        """Calculate the non-interacting density response function. """

        calc = self.calc
        gd = calc.wfs.gd
        sdisp_cd = gd.sdisp_cd
        ibzk_kv = self.ibzk_kv
        bzk_kv = self.bzk_kv
        kq = self.kq
        pt = self.pt
        f_kn = self.f_kn
        e_kn = self.e_kn

        # Matrix init
        chi0_wGG = np.zeros((self.Nw_local, self.npw, self.npw), dtype=complex)
        if self.HilbertTrans:
            specfunc_wGG = np.zeros((self.NwS_local, self.npw, self.npw), dtype = complex)

        # Prepare for the derivative of pseudo-wavefunction
        if self.OpticalLimit:
            d_c = [Gradient(gd, i, n=4, dtype=complex).apply for i in range(3)]
            dpsit_G = gd.empty(dtype=complex)
            tmp = np.zeros((3), dtype=complex)

        rho_G = np.zeros(self.npw, dtype=complex)
        t0 = time()

        for k in range(self.kstart, self.kend):

            # Find corresponding kpoint in IBZ
            ibzkpt1, iop1, timerev1 = find_ibzkpt(self.op, ibzk_kv, bzk_kv[k])
            if self.OpticalLimit:
                ibzkpt2, iop2, timerev2 = ibzkpt1, iop1, timerev1
            else:
                ibzkpt2, iop2, timerev2 = find_ibzkpt(self.op, ibzk_kv, bzk_kv[kq[k]])

            for n in range(self.nband):
                psitold_G =  calc.wfs.kpt_u[ibzkpt1].psit_nG[n]
                psit1new_G = symmetrize_wavefunction(psitold_G, self.op[iop1], ibzk_kv[ibzkpt1],
                                                      bzk_kv[k], timerev1)

                P1_ai = pt.dict()
                pt.integrate(psit1new_G, P1_ai, k)

                psit1_G = psit1new_G.conj() * self.expqr_G

                for m in range(self.nband):
		    if self.HilbertTrans:
			check_focc = (f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol
                    else:
                        check_focc = np.abs(f_kn[ibzkpt1, n] - f_kn[ibzkpt2, m]) > self.ftol 

                    if check_focc:
                        psitold_G =  calc.wfs.kpt_u[ibzkpt2].psit_nG[m]
                        psit2_G = symmetrize_wavefunction(psitold_G, self.op[iop2], ibzk_kv[ibzkpt2],
                                                           bzk_kv[kq[k]], timerev2)

                        P2_ai = pt.dict()
                        pt.integrate(psit2_G, P2_ai, kq[k])

                        # fft
                        tmp_G = np.fft.fftn(psit2_G*psit1_G) * self.vol / self.nG0

                        for iG in range(self.npw):
                            index = self.Gindex[iG]
                            rho_G[iG] = tmp_G[index[0], index[1], index[2]]

                        if self.OpticalLimit:
                            phase_cd = np.exp(2j * pi * sdisp_cd * bzk_kv[kq[k], :, np.newaxis])
                            for ix in range(3):
                                d_c[ix](psit2_G, dpsit_G, phase_cd)
                                tmp[ix] = gd.integrate(psit1_G * dpsit_G)
                            rho_G[0] = -1j * np.inner(self.qq, tmp)

                        # PAW correction
                        for a, id in enumerate(calc.wfs.setups.id_a):
                            P_p = np.outer(P1_ai[a].conj(), P2_ai[a]).ravel()
                            gemv(1.0, self.phi_aGp[a], P_p, 1.0, rho_G)

                        if self.OpticalLimit:
                            rho_G[0] /= e_kn[ibzkpt2, m] - e_kn[ibzkpt1, n]

                        rho_GG = np.outer(rho_G, rho_G.conj())
                        
                        if not self.HilbertTrans:
                            for iw in range(self.Nw_local):
                                w = self.wlist[iw + self.wstart] / Hartree
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
        if not self.HilbertTrans:
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
            size = extra_parameters['df_dry_run']
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

        if self.HilbertTrans:
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
        printtxt(self.acell)
        printtxt('Reciprocal cell (1/a.u.)')
        printtxt(self.bcell)
        printtxt('Grid spacing (a.u.):')
        printtxt(self.h_c)
        printtxt('Number of Grid points / G-vectors, and in total: (%d %d %d), %d'
                  %(self.nG[0], self.nG[1], self.nG[2], self.nG0))
        printtxt('Volome of cell (a.u.**3)     : %f' %(self.vol) )
        printtxt('BZ volume (1/a.u.**3)        : %f' %(self.BZvol) )
        printtxt('')                         
        printtxt('Number of bands              : %d' %(self.nband) )
        printtxt('Number of kpoints            : %d' %(self.nkpt) )
        printtxt('Planewave Ecut (eV)          : %f' %(self.Ecut * Hartree) )
        printtxt('Number of planewave used     : %d' %(self.npw) )
        printtxt('Broadening (eta)             : %f' %(self.eta * Hartree))
        printtxt('Number of frequency points   : %d' %(self.Nw) )
        if self.HilbertTrans:
            printtxt('Number of specfunc points    : %d' % (self.NwS))
        printtxt('')
        if self.OpticalLimit:
            printtxt('Optical limit calculation ! (q=0.00001)')
        else:
            printtxt('q in reduced coordinate        : (%f %f %f)' %(self.q[0], self.q[1], self.q[2]) )
            printtxt('q in cartesian coordinate (1/A): (%f %f %f) '
                  %(self.qq[0] / Bohr, self.qq[1] / Bohr, self.qq[2] / Bohr) )
            printtxt('|q| (1/A)                      : %f' %(sqrt(np.inner(self.qq / Bohr, self.qq / Bohr))) )
        printtxt('')
        printtxt('Use Hilbert Transform: %s' %(self.HilbertTrans) )
        printtxt('')
        printtxt('Parallelization scheme:')
        printtxt('     Total cpus      : %d' %(self.comm.size))
        printtxt('     kpoint parsize  : %d' %(self.kcomm.size))
        printtxt('     specfunc parsize: %d' %(self.wScomm.size))
        printtxt('     w parsize       : %d' %(self.wcomm.size))
        printtxt('')
        printtxt('Memory usage estimation:')
        printtxt('     chi0_wGG        : %f M / cpu' %(self.Nw_local * self.npw**2 * 16. / 1024**2) )
        if self.HilbertTrans:
            printtxt('     specfunc_wGG    : %f M / cpu' %(self.NwS_local *self.npw**2 * 16. / 1024**2) )





