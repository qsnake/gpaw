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
from gpaw.response.math_func import delta_function, hilbert_transform, \
     two_phi_planewave_integrals
from gpaw.response.parallel import set_communicator, \
     parallel_partition, SliceAlongFrequency, SliceAlongOrbitals
from gpaw.response.kernel import calculate_Kxc
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.memory import maxrss

class BASECHI:
    """This class is to store the basic common stuff for chi and bse."""

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

        self.xc = 'LDA'

        self.txtname = txt
        self.output_init()

        if isinstance(calc, str):
            # Always use serial_communicator when a filename is given.
            self.calc = GPAW(calc, communicator=serial_comm, txt=None)
        else:
            # To be optimized so that the communicator is loaded automatically 
            # according to kcommsize.
            # 
            # so temporarily it is used like this :
            # kcommsize = int (should <= world.size)
            # r0 = rank % kcommsize
            # ranks = np.arange(r0, r0+size, kcommsize)
            # calc = GPAW(filename.gpw, communicator=ranks, txt=None)
            self.calc = calc

        self.nbands = nbands
        self.nspins = self.calc.wfs.nspins
        self.q_c = q

        self.w_w = w
        self.eta = eta
        self.ftol = ftol
        if type(ecut) is int or type(ecut) is float:
            self.ecut = np.ones(3) * ecut
        else:
            assert len(ecut) == 3
            self.ecut = np.array(ecut, dtype=float)
        self.optical_limit = optical_limit


    def initialize(self):
                        
        self.eta /= Hartree
        self.ecut /= Hartree

        calc = self.calc
        
        # kpoint init
        self.kd = kd = calc.wfs.kd
        self.bzk_kc = kd.bzk_kc
        self.ibzk_kc = kd.ibzk_kc
        self.nkpt = kd.nbzkpts
        self.ftol /= self.nkpt

        # band init
        if self.nbands is None:
            self.nbands = calc.wfs.nbands
        self.nvalence = calc.wfs.nvalence

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
        nibzkpt = kd.nibzkpts
        kweight_k = kd.weight_k

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
            kq_k = kd.find_k_plus_q(self.q_c)
        self.kq_k = kq_k

        # Plane wave init
        self.npw, self.Gvec_Gc, self.Gindex_G = set_Gvectors(self.acell_cv, self.bcell_cv, self.nG, self.ecut)

        # Projectors init
        setups = calc.wfs.setups
        pt = LFC(gd, [setup.pt_j for setup in setups],
                 dtype=calc.wfs.dtype, forces=True)
        spos_ac = calc.atoms.get_scaled_positions()
        pt.set_k_points(self.bzk_kc)
        pt.set_positions(spos_ac)
        self.pt = pt

        # Printing calculation information
        self.print_stuff()

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
        if self.optical_limit:
            printtxt('Optical limit calculation ! (q=0.00001)')
        else:
            printtxt('q in reduced coordinate        : (%f %f %f)' %(self.q_c[0], self.q_c[1], self.q_c[2]) )
            printtxt('q in cartesian coordinate (1/A): (%f %f %f) '
                  %(self.qq_v[0] / Bohr, self.qq_v[1] / Bohr, self.qq_v[2] / Bohr) )
            printtxt('|q| (1/A)                      : %f' %(sqrt(np.dot(self.qq_v / Bohr, self.qq_v / Bohr))) )

        return


    def get_phi_aGp(self):
        setups = self.calc.wfs.setups
        spos_ac = self.calc.atoms.get_scaled_positions()
        
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

        return


    def get_wavefunction(self, ibzk, n, check_focc=True, spin=0):

        if self.calc.wfs.kpt_comm.size != world.size or world.size == 1:

            if check_focc == False:
                return
            else:
                psit_G = self.calc.wfs.get_wave_function_array(n, ibzk, spin)
        
                if self.calc.wfs.world.size == 1:
                    return np.complex128(psit_G)
                
                if not self.calc.wfs.world.rank == 0:
                    psit_G = self.calc.wfs.gd.empty(dtype=self.calc.wfs.dtype,
                                                    global_array=True)
                self.calc.wfs.world.broadcast(psit_G, 0)
        
                return np.complex128(psit_G)
        else:
            if self.nkpt % size != 0:
                raise ValueError('The number of kpoints should be divided by the number of cpus for no wfs dumping mode ! ')

            # support ground state calculation with only kpoint parallelization
            kpt_rank, u = self.calc.wfs.kd.get_rank_and_index(0, ibzk)
            bzkpt_rank = rank
            
            klist = np.array([kpt_rank, u, bzkpt_rank, n])
            klist_kcomm = np.zeros((self.kcomm.size, 4), dtype=int)            
            self.kcomm.all_gather(klist, klist_kcomm)

            check_focc_global = np.zeros(self.kcomm.size, dtype=bool)
            self.kcomm.all_gather(np.array([check_focc]), check_focc_global)

            psit_G = self.calc.wfs.gd.empty(dtype=self.calc.wfs.dtype)

	    for i in range(self.kcomm.size):
                if check_focc_global[i] == True:
                    kpt_rank, u, bzkpt_rank, nlocal = klist_kcomm[i]
                    if kpt_rank == bzkpt_rank:
                        if rank == kpt_rank:
                            psit_G = self.calc.wfs.kpt_u[u].psit_nG[nlocal]
                    else:
                        if rank == kpt_rank:
                            world.send(self.calc.wfs.kpt_u[u].psit_nG[nlocal],
                                       bzkpt_rank, 1300+bzkpt_rank)
                        if rank == bzkpt_rank:
                            psit_G = self.calc.wfs.gd.empty(dtype=self.calc.wfs.dtype)
                            world.receive(psit_G, kpt_rank, 1300+bzkpt_rank)
                    
            self.wScomm.broadcast(psit_G, 0)

            return psit_G


    def density_matrix(self,n,m,k,Gspace=True):

        ibzk_kc = self.ibzk_kc
        bzk_kc = self.bzk_kc
        kq_k = self.kq_k
        gd = self.gd
        kd = self.kd

        ibzkpt1 = kd.kibz_k[k]
        ibzkpt2 = kd.kibz_k[kq_k[k]]
        
        psitold_g = self.get_wavefunction(ibzkpt1, n, True)
        psit1_g = kd.transform_wave_function(psitold_g, k)
        
        psitold_g = self.get_wavefunction(ibzkpt2, m, True)
        psit2_g = kd.transform_wave_function(psitold_g, kq_k[k])

        if Gspace is False:
            return psit1_g, psit2_g
        else:
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
