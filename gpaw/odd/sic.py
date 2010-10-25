"""Perdew-Zunger SIC to DFT functionals (currently only LDA)

   Self-consistent minimization of self-interaction corrected
   LDA functionals (Perdew-Zunger)
"""

from math import pi,cos,sin,log10,exp,atan2
#
import sys
import numpy as np

import gpaw.mpi as mpi
#from ase import *
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import axpy, gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XCFunctional
from gpaw.utilities.timing import Timer
from gpaw.poisson import PoissonSolver
from gpaw.odd.sic_gauss import Gaussian
from gpaw.atom.generator import Generator
from gpaw.atom.configurations import parameters
from ase.units import Bohr, Hartree
from gpaw.lfc import LFC
from gpaw.odd import SICXCCorrection
from gpaw.hgh import NullXCCorrection
from gpaw import extra_parameters

from _gpaw import localize

class SIC:
    def __init__(self, xcname='LDA', finegrid=False, coufac=1.0, excfac=1.0,
                 uominres=1E-1, uomaxres=1E-12, uorelres=1.0E-2,
                 nspins=1, test=0, txt=None, rattle=-0.1, maxuoiter=30):
        """Self-Interaction Corrected (SIC) Functionals.

        xcname: string
            Name of LDA functional which acts as
            a starting point for the construction of
            the self-interaction corrected functional

        nspins: int
            Number of spins.

        finegrid: boolean
            Use fine grid for energy functional evaluations?

        coufac:
            Scaling factor for Hartree-functional

        excfac:
            Scaling factor for xc-functional

        uominres:
            Minimum residual before unitary optimization starts

        uomaxres:
            Target accuracy for unitary optimization
            (absolute variance)

        uorelres:
            Target accuracy for unitary optimization
            (rel. to basis residual)

        maxuoiter:
            Maximum number of unitary optimization steps

        test:
            debug level

        txt:
            log file for unitary optimization

        rattle:
            perturbation to the initial states

        """
        #
        # parameters
        self.nspins = nspins
        self.coufac = coufac    
        self.excfac = excfac    
        self.finegrid = finegrid
        self.rattle = rattle    
        self.uominres = uominres
        self.uomaxres = uomaxres
        self.uorelres = uorelres
        self.maxuoiter = maxuoiter 
        #
        self.optcmplx = False      # complex optimization
        self.adderror = False      # add unit-opt. residual to basis-residual
        self.virt_SIC = False      # evaluate SIC for virtual orbitals
        self.new_coul = True       # use the ODD-coulomb solver 
        self.maxlsiter = 1         # maximum number of line-search steps
        self.maxcgiter = 2         # maximum number of CG-iterations
        self.lsinterp = True       # interpolate for minimum during line search
        self.act_SIC = True        # self-consistent SIC
        self.opt_SIC = True        # unitary optimization
        #
        # debugging parameters
        self.units = Hartree       # output units (set to 1 for Hartree)
        self.debug = test          # debug level
        #
        # initialization         
        self.dtype = None     # complex/float
        self.nbands = None     # total number of bands/orbitals
        self.nblocks = None     # total number of blocks
        self.mybands = None     # list of bands of node's responsibility
        self.myblocks = None     # list of blocks of node's responsibility
        self.init_SIC = True     # SIC functional has to be initialized?
        self.init_cou = True     # coulomb solver has to be initialized?
        self.active_SIC = False    # SIC is activated
        self.ESI = 0.0      # orbital dependent energy (SIC)
        self.RSI = 0.0      # residual of unitary optimization
        self.Sha = 0.0      
        self.Sxc = 0.0
        self.Stot = 0.0
        self.npoisson = 0
        self.basiserror = 1E+20
        self.subiter = 0
        self.nsubiter = 0
        #
        # default variables of xc-functionals
        self.xcbasisname = xcname
        self.xcname = xcname + '-SIC'
        self.xcbasis = XCFunctional(self.xcbasisname, nspins)
        self.gga = self.xcbasis.gga
        self.mgga = False
        self.orbital_dependent = True
        self.hybrid = 0.0
        self.uses_libxc = self.xcbasis.uses_libxc
        self.gllb = False
        #
        # set extra parameter 'sic'
        extra_parameters['sic'] = True
        #
        # turn of unitary optimization of ODD functional is zero
        if (self.coufac == 0.0 and self.excfac == 0.0):
            self.opt_SIC = False
        #
        # open log files for unitary optimization
        if (txt == None):
            self.logging = False
        else:
            self.logging = True
            self.logfile = {}
            
            if (nspins==1):
                self.logfile[0] = open(txt+'.log', 'w')
            else:
                self.logfile[0] = open(txt+'_0.log', 'w')
                self.logfile[1] = open(txt+'_1.log', 'w')

    def set_non_local_things(self, density, hamiltonian, wfs, atoms,
                             energy_only=False):

        self.wfs = wfs
        self.density = density
        self.hamiltonian = hamiltonian
        self.atoms = atoms
        self.nbands = wfs.nbands
        self.nspins = hamiltonian.nspins
        self.nblocks = hamiltonian.nspins
        self.timer = hamiltonian.timer
        #
        # list of blocks stored on this node
        self.mybands = range(self.nbands)
        self.myblocks = []
        for kpt in wfs.kpt_u:
            if not kpt.s in self.myblocks:
                self.myblocks.append(kpt.s)
        #
        # make sure that we are doing a gamma-point or finite system
        # calculation
        assert wfs.nibzkpts == 1, ( 
            'ODD functionals requires finite or Gamma-Point calculation')
        #
        # check for periodicity in any dimension
        # and switch to custom coulomb solver if necessary
        pbc = self.atoms.get_pbc()
        if pbc[0] or pbc[1] or pbc[2]:
            self.pbc = pbc
            self.periodic = True
            self.new_coul = True
        else:
            self.pbc = pbc
            self.periodic = False
            self.new_coul = False
        #
        # check if complex optimization is toggled on
        if (self.optcmplx):
            self.dtype = complex
        else:
            self.dtype = float
        #
        # setup grid-descriptors
        self.gd = density.gd
        if self.finegrid:
            self.finegd = density.finegd
        else:
            self.finegd = density.gd
        #
        # ODD requires a spin-polarized variant of the
        # xc-functional, create it if necessary
        if self.nspins == 1:
            self.xcsic = XCFunctional(self.xcbasisname, 2)
        else:
            self.xcsic = self.xcbasis

        for setup in wfs.setups.setups.values():
            if isinstance(setup.xc_correction, NullXCCorrection):
                setup.xc_correction_sic = setup.xc_correction
            else:
                setup.xc_correction_sic = SICXCCorrection(setup.xc_correction,
                                                          self.xcsic)
        #
        # setup localized functions on coarse grid (if necessary)
        if not self.finegrid:
            self.ghat = LFC(density.gd,
                           [setup.ghat_l
                             for setup in density.setups],
                            integral=np.sqrt(4 * np.pi))
            self.ghat.set_positions(atoms.get_scaled_positions() % 1.0)
        #
        # initialize poisson solver for self-coulomb
        self.psolver = PoissonSolver(nn=3, relax='J')
        self.psolver.set_grid_descriptor(self.finegd)
        self.psolver.initialize()
        #
        # summarize
        if mpi.rank==0 and self.logging:
            print '# complex optimization   :',self.optcmplx
            print '# fine-grid integrations :',self.finegrid
            print '# periodic boundaries    :',self.periodic
            print '# custom coulomb solver  :',self.new_coul
            print '# CPU  blocks     bands'
            mpi.world.barrier()
            for node in range(mpi.size):
                mpi.world.barrier()
                if (node == mpi.rank):
                    print ('# %4d %s %s' % (mpi.rank, str(self.myblocks).ljust(10),
                                               str(self.mybands).ljust(40)))
        #
        # list of nodes containing the real-space partitioning for the
        # current k-point (for now we do not mix k-points).
        nodes = self.gd.comm.get_members()
        self.blk_comm = mpi.world.new_communicator(nodes)
        #
        nblocks = self.nblocks
        nbands = self.nbands
        mynbands = len(self.mybands)
        mynblocks = len(self.myblocks)
        #
        # real-space representations of densities/WF and fields
        # on the coarse grid
        self.nt_G        = self.gd.empty(dtype=float)
        self.rnt_G       = self.gd.zeros(dtype=float)
        self.v_unG       = self.gd.empty((mynblocks,mynbands),dtype=float)
        self.v_cou_unG   = self.gd.empty((mynblocks,mynbands),dtype=float)
        self.phit_unG    = self.gd.empty((mynblocks,mynbands),dtype=float)
        self.rphit_unG   = self.gd.empty((mynblocks,mynbands),dtype=float)
        self.phit_G      = self.gd.empty(dtype=float)
        self.Htphit_unG  = self.gd.empty((mynblocks,mynbands),dtype=float)
        if (self.optcmplx):
            self.phit2_unG   = self.gd.empty((mynblocks,mynbands),dtype=float)
            self.Htphit2_unG = self.gd.empty((mynblocks,mynbands),dtype=float)
        #
        # projections
        self.P_uani      = [None] * mynblocks
        self.dH_unap     = []
        for u in self.myblocks:
            dH_nap = []
            self.dH_unap.append(dH_nap)
            for n in self.mybands:
                dH_ap = {}
                dH_nap.append(dH_ap)
        #
        # utility fields on the fine grid
        self.nt_g        = self.finegd.zeros(dtype=float)
        self.rnt_g       = self.finegd.zeros(dtype=float)
        self.nt_g0       = self.finegd.zeros(dtype=float)
        self.v_g0        = self.finegd.zeros(dtype=float)
        self.e_g         = self.finegd.zeros(dtype=float)
        self.v_g         = self.finegd.zeros(dtype=float)
        self.v_cou_g     = self.finegd.zeros(dtype=float)
        #
        # occupation numbers and single-particle energies
        # for all states
        self.f_un        = np.zeros((nblocks,nbands),dtype=float)
        self.m_un        = np.zeros((nblocks,nbands),dtype=float)
        self.eps_un      = np.zeros((nblocks,nbands),dtype=float)
        self.Sha_un      = np.zeros((nblocks,nbands),dtype=float)
        self.Sxc_un      = np.zeros((nblocks,nbands),dtype=float)
        self.Stot_un     = np.zeros((nblocks,nbands),dtype=float)
        #self.nocc_u      = np.zeros((nblocks)       ,dtype=float)
        #
        # transformation from canonic to energy optimal states
        self.W_unn         = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.H0_unn        = np.zeros((mynblocks,nbands,nbands),dtype=float)
        self.init_states   = True
        self.setup_unified = [True]*mynblocks
        #
        # constraint matrix
        self.K_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.V_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.O_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.Tmp_nn      = np.zeros((nbands,nbands),dtype=float)
        self.Tmp2_nn     = np.zeros((nbands,nbands),dtype=float)
        #
        self.normK_q     = np.zeros((mynblocks),dtype=float)
        self.RSI_q       = np.zeros((mynblocks),dtype=float)
        self.Ecorr_q     = np.zeros((mynblocks),dtype=float)
        self.npart_q     = np.zeros((mynblocks),dtype=int)
        #
        # TEST stuff
        self.Rgrid       = None
        self.Rphase      = None
        self.mask        = None
        self.cell        = atoms.cell/Bohr
        self.gauss       = Gaussian(self.finegd,self.cell,self.pbc,
                                    self.density)
        self.rho_gauss   = self.finegd.zeros(dtype=float)
        self.phi_gauss   = self.finegd.zeros(dtype=float)
        self.mask        = self.finegd.zeros(dtype=float)
        self.rcell       = 2.0*np.pi*np.linalg.solve(self.cell.T, np.eye(3))
        self.pos_un      = np.zeros((3,nblocks,nbands),dtype=np.float64)
        self.loc_un      = np.zeros((nblocks,nbands),dtype=np.float64)
        self.rho_n       = None
        #
        if (self.periodic):
            self.masked_nt_g = self.finegd.empty(dtype=float)
            self.rho_n       = self.finegd.empty(dtype=float)
        else:
            self.masked_nt_g = self.nt_g

    def is_gllb(self):
        return False

    def get_name(self):
        return self.xcname

    def get_setup_name(self):
        return self.xcbasisname

    def apply_non_local(self, kpt):
        pass

    def get_non_local_kinetic_corrections(self):
        if (self.act_SIC):
            Ecorr = 0.0
            for u in self.myblocks:
                q = self.myblocks.index(u)
                Ecorr = Ecorr + self.Ecorr_q[q]
            return self.wfs.kpt_comm.sum(Ecorr)
        else:
            return 0.0

    def adjust_non_local_residual(self, pR_G, dR_G, kpt, n):
        pass

    def get_non_local_force(self, kpt):
        return 0.0

    def get_non_local_energy(self, n_g=None, a2_g=None, e_LDAc_g=None,
                             v_LDAc_g=None, v_g=None, deda2_g=None):
        if (self.act_SIC):
            return self.Stot
        else:
            return 0.0

    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g=None, deda2_g=None):
        """ calculate the spin paired LDA/GGA part of the energy functional
            and optimize the energy optimal states by unitary optimization
        """
        #
        # the LDA/GGA part of the functional
        self.xcbasis.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
        #
        # optimization of energy optimal states (unitary optimization)
        if n_g.ndim == 3:
            self.unitary_optimization()


    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                                a2_g=None,  aa2_g=None, ab2_g=None,
                                deda2_g=None, dedaa2_g=None, dedab2_g=None):
        """ calculate the spin polarized LDA/GGA part of the energy
            functional and optimize the energy optimal states by
            unitary optimization
        """       
        #
        # the LDA/GGA part of the functional 
        self.xcbasis.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                        a2_g, aa2_g, ab2_g,
                                        deda2_g, dedaa2_g, dedab2_g)
        #
        # optimization of energy optimal states (unitary optimization)
        if na_g.ndim == 3:
            self.unitary_optimization()

    def calculate_sic_matrixelements(self,blocks=[]):
        #
        # check if wavefunctions and ODD potentials
        # have already been initialized
        if self.wfs.kpt_u[0].psit_nG is None or self.init_SIC:
            return 
        #
        # start the timer
        self.timer.start('ODD - matrixelements')
        #
        # select blocks which need to be evaluated
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        mybands     = self.mybands
        #
        # action of the SIC potentials
        self.Htphit_unG   = self.v_unG*self.phit_unG
        #
        # evaluate symmetrized V_ij and Kappa_ij
        for u in myblocks:
            #
            # get the local index of the block u
            q=self.myblocks.index(u)
            f=self.f_un[u]/(3-self.nspins)
            #
            # calculate SIC matrix <Phi_i|V_j|Phi_j>
            if (self.optcmplx):
                gemm(self.gd.dv,self.phit_unG[q],self.Htphit_unG[q],
                     0.0,self.Tmp_nn,'t')
                gemm(self.gd.dv,self.phit2_unG[q],self.Htphit2_unG[q],
                     1.0,self.Tmp_nn,'t')
                self.V_unn[q] = self.Tmp_nn
                gemm(self.gd.dv,self.phit_unG[q],self.Htphit2_unG[q],
                     0.0,self.Tmp_nn,'t')
                gemm(self.gd.dv,self.phit2_unG[q],self.Htphit_unG[q],
                    -1.0,self.Tmp_nn,'t')
                self.V_unn[q] += 1j*self.Tmp_nn
            else:
                gemm(self.gd.dv,self.phit_unG[q],self.Htphit_unG[q],
                     0.0,self.V_unn[q],'t')
                
            #
            # add PAW corrections
#PAW            if (1==0):
#PAW                for n in range(self.nbands):
#PAW                    for a, dH_p in self.dH_unap[q][n].items():
#PAW                        dH_ii = unpack(dH_p)
#PAW                        P_ni = self.P_uani[q][a]
#PAW                        P_i = P_ni[n]
#PAW                    self.V_unn[q, :, n] += np.dot(P_ni, np.dot(dH_ii, P_i))
            #
            # collect contributions from different nodes
            self.gd.comm.sum(self.V_unn[q])
            #
            # apply subspace mask
            self.V_unn[q] *= np.outer(f,f)
            #
            # symmetrization of V and kappa-Matrix
            self.K_unn[q] = 0.5*(self.V_unn[q] - self.V_unn[q].T.conj())
            self.V_unn[q] = 0.5*(self.V_unn[q] + self.V_unn[q].T.conj())
            #
            #print 'SIC-potential matrix <Phi_i|V_j|Phi_j> of block ',q,' on CPU',mpi.rank
            #print self.V_unn[q]+self.K_unn[q]
            #
            # total norm of kappa-matrix
            self.normK_q[q] = np.sum(self.K_unn[q]*self.K_unn[q].conj())
        #
        # stop the timer
        self.timer.stop('ODD - matrixelements')

    def unitary_optimization(self,forced=False):
        #
        test       = self.debug
        myblocks   = self.myblocks
        #
        #
        # skip unitary optimization if initialization is not finished
        if self.wfs.kpt_u[0].psit_nG is None:
            return
        #
        # check if ODD-functional is active
        basiserror = self.wfs.eigensolver.error
        if basiserror > self.uominres:
            if (not self.active_SIC and not forced):
                return
            self.basiserror = 1E-4
        else:
            self.active_SIC = True
        #
        # get the basis error from the eigensolver
        self.basiserror = min(basiserror,self.basiserror)
        basiserror = self.basiserror
        #
        localerror=0.0
        maxiter   =self.maxuoiter
        if forced:
            maxiter = maxiter*20
        #
        ESI_init = 0.0
        ESI      = 0.0
        dE       = 1e-16  
        #
        #
        # set variables to zero
        self.loc_un[:]  = 0.0
        self.pos_un[:]  = 0.0
        self.f_un[:]    = 0.0
        self.eps_un[:]  = 0.0
        self.Sha_un[:]  = 0.0
        self.Sxc_un[:]  = 0.0
        self.Stot_un[:] = 0.0      
        #
        #
        # compensate the change in the basis functions during subspace
        # diagonalization and update the energy optimal states
        if self.init_states:
            self.init_states=False
            self.init_unitary_transformation(blocks=self.myblocks,
                                             rattle=self.rattle)
        else:
            self.update_unitary_transformation(myblocks)
            #
        #
        # allocate temporary arrays
        U_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        O_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        #
        # loop all blocks
        for u in myblocks:
            #
            if self.logging:
                log=self.logfile[u]
            #
            # get the local index of the block u
            q=self.myblocks.index(u)
            #
            # logging
            #if test>2:
            #    print 'CPU ',mpi.rank,' optimizing block ',u,\
            #    ' (local index ',q,')'
            #
            optstep  = 0.0
            Gold     = 0.0
            cgiter   = 0
            #
            epsstep  = 0.001  # 0.005
            dltstep  = 0.1    # 0.1
            prec     = 1E-6
            #
            #
            # get the initial ODD potentials/energy/matrixelements
            self.update_optimal_states([u])
            ESI=self.update_potentials([u])
            self.calculate_sic_matrixelements([u])
            ESI_init = ESI
            #
            # decide if unitary optimization is necessary
            # ------------------------------------------------------------------
            if self.normK_q[q] < 1E-10 or self.subiter != 0:
                #
                # no unitary optimization necessary
                # reason: optimal states already sufficiently converged
                # --------------------------------------------------------------
                dE2      = 1E-16
                dE       = 1E-16
                K        = 1E-16
                self.RSI_q[q] = 1E-16
                optstep  = 0.0
                lsiter   = 0
                failed   = False
                lsmethod = 'skipped'
                #
            elif self.npart_q[q] == 0 or self.npart_q[q] == 1:
                #
                # no unitary optimization necessary
                # reason: no or only one particle in current (spin) block
                # --------------------------------------------------------------
                dE2      = 1E-16
                dE       = 1E-16
                K        = 1E-16
                self.RSI_q[q] = 1E-16
                optstep  = 0.0
                lsiter   = 0
                failed   = False
                lsmethod = 'skipped'
                #
            elif not self.opt_SIC:
                #
                # no unitary optimization necessary
                # reason: deactivated unitary optimization
                # --------------------------------------------------------------
                dE2      = 1E-16
                dE       = 1E-16                
                K        = 1E-16
                self.RSI_q[q] = 1E-16
                optstep  = 0.0
                lsiter   = 0
                failed   = False
                lsmethod = 'skipped'
            else:
                #
                # optimize the unitary transformation
                # --------------------------------------------------------------
                #
                # allocate arrays for the search direction,
                # i.e., the (conjugate) gradient
                D_nn  = np.zeros_like(self.W_unn[q])
                D_old = np.zeros_like(self.W_unn[q])
                W_old = np.zeros_like(self.W_unn[q])
                K_nn  = np.zeros_like(self.W_unn[q])
                #
                for iter in range(maxiter):
                    #
                    # copy the initial unitary transformation and orbital
                    # dependent energies
                    W_old    = self.W_unn[q]
                    K_nn     = self.K_unn[q]
                    ESI_old  = ESI
                    #
                    # setup the steepest-descent/conjugate gradient
                    # D_nn:  search direction
                    # K_nn:  inverse gradient
                    # G0  :  <K,D> (projected length of D along K)
                    if (Gold!=0.0):
                        #
                        # conjugate gradient
                        G0        = np.sum(K_nn*K_nn.conj()).real
                        beta      = G0/Gold
                        Gold      = G0
                        D_nn      = K_nn + beta*D_old
                        G0        = np.sum(K_nn*D_nn.conj()).real
                    else:
                        #
                        # steepest-descent
                        G0        = np.sum(K_nn*K_nn.conj()).real
                        Gold      = G0
                        D_nn      = K_nn
                    #
                    updated  = False
                    minimum  = False
                    failed   = True
                    E0       = ESI
                    #
                    # try to estimate optimal step-length from change in length
                    # of the gradient (force-only)
                    # ----------------------------------------------------------
                    if (epsstep!=0.0):
                        #
                        # infinitesimal steepest descent
                        step = max(min(epsstep/np.sqrt(abs(G0)),1.0),1E-3)
                        while (True):
                            U_nn = matrix_exponential(D_nn, step)
                            self.W_unn[q] = np.dot(U_nn, W_old)
                            self.update_optimal_states([u],rotate_only=True)
                            E1 = self.update_potentials([u])
                            self.calculate_sic_matrixelements([u])
                            #
                            # projected length of the gradient at the new position
                            G1 = np.sum(self.K_unn[q]*D_nn.conj()).real
                            #
                            if (abs(E1-E0)<prec):
                                #
                                eps_works = True
                                Eeps      = E1
                            elif (E1<E0):
                                #
                                # trial step reduced energy
                                eps_works = True
                                Eeps      = E1
                            else:
                                #
                                # scale down trial step
                                eps_works = False
                                optstep   = 0.0
                                break
                                #step = 0.5*step
                                #if step<1.0:
                                #    eps_works=False
                                #    break
                                #print 'scaling down steplength', step
                                #continue
                            #
                            # compute the optimal step size
                            optstep = step/(1.0-G1/G0)
                            #
                            if (eps_works):
                                break
                            #
                        #print 'trial step: ',step,optstep,E1-E0,G0,G1
                        #
                        # decide on the method for stepping
                        if (optstep > 0.0):
                            #
                            # convex region -> force only estimate for minimum
                            U_nn = matrix_exponential(D_nn,optstep)
                            self.W_unn[q] = np.dot(U_nn,W_old)
                            self.update_optimal_states([u],rotate_only=True)
                            E1=self.update_potentials([u])
                            if (abs(E1-E0)<prec):
                                self.calculate_sic_matrixelements([u])
                                ESI       = E1
                                optstep   = optstep
                                lsiter    = 0
                                maxlsiter = -1
                                updated   = True
                                minimum   = True
                                failed    = False
                                lsmethod  = 'CV-N'
                            if (E1<E0):
                                self.calculate_sic_matrixelements([u])
                                ESI       = E1
                                optstep   = optstep
                                lsiter    = 0
                                maxlsiter = -1
                                updated   = True
                                minimum   = True
                                failed    = False
                                lsmethod  = 'CV-S'
                            else:
                                self.K_unn[q] = K_nn
                                ESI       = E0
                                step      = optstep
                                optstep   = 0.0
                                lsiter    = 0
                                maxlsiter = self.maxlsiter
                                updated   = False
                                minimum   = False
                                failed    = True
                                lsmethod  = 'CV-F-CC'
                        else:
                            self.K_unn[q] = K_nn
                            ESI       = E0
                            step      = optstep
                            optstep   = 0.0
                            lsiter    = 0
                            maxlsiter = self.maxlsiter
                            updated   = False
                            minimum   = False
                            failed    = True
                            lsmethod  = 'CC'
                        #
                    if (optstep==0.0):
                        #
                        # we are in the concave region or force-only estimate failed,
                        # just follow the (conjugate) gradient
                        step = dltstep * abs(step)
                        #print step
                        U_nn = matrix_exponential(D_nn,step)
                        self.W_unn[q] = np.dot(U_nn,W_old)
                        self.update_optimal_states([u],rotate_only=True)
                        E1 = self.update_potentials([u])
                        #
                        #
                        if (abs(E1-E0)<prec):
                            ESI       = E1
                            optstep   = 0.0
                            updated   = False
                            minimum   = True
                            failed    = True
                            lsmethod  = lsmethod+'-DLT-N'
                            maxlsiter = -1
                        elif (E1<E0):
                            ESI       = E1
                            optstep   = step
                            updated   = True
                            minimum   = False
                            failed    = False
                            lsmethod  = lsmethod+'-DLT'
                            maxlsiter = self.maxlsiter
                        elif (eps_works):
                            ESI       = Eeps
                            E1        = Eeps
                            step      = epsstep
                            updated   = False
                            minimum   = False
                            failed    = False
                            lsmethod  = lsmethod+'-EPS'
                            maxlsiter = self.maxlsiter
                        else:
                            optstep   = 0.0
                            updated   = False
                            minimum   = False
                            failed    = True
                            lsmethod  = lsmethod+'-EPS-failed'
                            maxlsiter = -1
                        #
                        G       = (E1-E0)/step
                        step0   = 0.0
                        step1   = step
                        step2   = 2*step
                        #
                        for lsiter in range(maxlsiter):
                            #
                            # energy at the new position
                            U_nn = matrix_exponential(D_nn,step2)
                            self.W_unn[q] = np.dot(U_nn,W_old)
                            self.update_optimal_states([u],rotate_only=True)
                            E2=self.update_potentials([u])
                            G  = (E2-E1)/(step2-step1)
                            #
                            #print lsiter,E2,G,step2,step
                            #
                            if (G>0.0):
                                if self.lsinterp:
                                    a= E0/((step2-step0)*(step1-step0)) \
                                     + E2/((step2-step1)*(step2-step0)) \
                                     - E1/((step2-step1)*(step1-step0))
                                    b=(E2-E0)/(step2-step0)-a*(step2+step0)
                                    if (a<step**2):
                                        optstep = 0.5*(step0+step2)
                                    else:
                                        optstep =-0.5*b/a
                                    updated  = False
                                    minimum  = True
                                    break
                                else:
                                    optstep  = step1
                                    updated  = False
                                    minimum  = True
                                    break
                            #
                            elif (G<0.0):
                                optstep = step2
                                step0   = step1
                                step1   = step2
                                step2   = step2 + step
                                E0      = E1
                                E1      = E2
                                ESI     = E2
                                updated = True
                                minimum = False
                    #
                    if (cgiter!=0):
                        lsmethod = lsmethod + ' CG'
                    #
                    if (cgiter>=self.maxcgiter or not minimum):
                        Gold        = 0.0
                        cgiter      = 0
                    else:
                        cgiter      = cgiter + 1
                        D_old[:,:]  = D_nn[:,:]
                    #
                    # update the energy and matrixelements of V and Kappa
                    # and accumulate total residual of unitary optimization
                    if (not updated):
                        if (optstep==0.0):
                            self.W_unn[q,:] = W_old[:]
                            self.update_optimal_states([u],rotate_only=True)
                            ESI=self.update_potentials([u])
                            self.calculate_sic_matrixelements([u])
                        else:
                            U_nn = matrix_exponential(D_nn,optstep)
                            self.W_unn[q] = np.dot(U_nn,W_old)
                            self.update_optimal_states([u],rotate_only=True)
                            ESI=self.update_potentials([u])
                            self.calculate_sic_matrixelements([u])

                    if (lsiter==maxlsiter-1):
                        optstep = step1
                        self.calculate_sic_matrixelements([u])
                    #
                    E0=ESI
                    #
                    # orthonormalize the energy optimal orbitals
                    self.W_unn[q] = ortho(self.W_unn[q])
                    self.RSI_q[q] = self.normK_q[q]
                    #
                    # logging
                    dE2=max(abs(ESI-ESI_old),1.0E-16)
                    dE =max(abs(ESI-ESI_init),1.0E-16)
                    K  =max(self.RSI_q[q],1.0E-16)
                    #
                    # logging
                    if self.finegd.comm.rank == 0 and self.logging:
                        log.write((" %3i  %10.5f %10.5f %5.1f %5.1f %5.1f %10.3f %3i %s\n" %
                                   (iter+1,ESI*self.units,
                                    (ESI-ESI_init)*self.units,
                                    log10(dE),log10(dE2),log10(K),
                                    optstep,lsiter+1,lsmethod)))
                        log.flush()
                    #
                    # iteration converged
                    if K<basiserror*self.uorelres or K<self.uomaxres:
                        localerror = localerror + K
                        break
                #
            if self.finegd.comm.rank == 0 and self.logging:
                log.write("\n")
                log.flush()
        #
        # add residual of unitary optimization to basis error to
        # avoid premature exit of basis optimization
        if self.adderror:
            self.wfs.eigensolver.error = basiserror + localerror
        #
        # update the single-particle ODD-energies
        self.update_energies(self.myblocks,self.mybands)
        #
        for u in self.myblocks:
            q=self.myblocks.index(u)
            self.setup_unified[q] = True
        #
        #print self.subiter
        if self.subiter==self.nsubiter:
            self.subiter=0
        else:
            self.subiter+=1
        
        

    def add_non_local_terms(self, psit_nG, Htpsit_nG, u):
        #
        # skip if SIC is not initialized or if feedback is
        # temporary or permanently disabled
        if self.init_SIC or not self.act_SIC or not self.active_SIC:
            return
        #
        q      = self.myblocks.index(u)
        f      = self.f_un[u]/(3-self.nspins)
        eps_n  = self.eps_un[u]
        nbands = psit_nG.shape[0]
        #
        #if (not self.unified_type==0):
            
        #
        # start the timer
        self.timer.start('ODD - basis action')
        #
        if (nbands==self.nbands and self.setup_unified[q]):
            #
            #q_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            #
            # get the unitary transformation from
            # energy optimal states |phi_k> to canonic states |k>
            W_nn  =  self.W_unn[q].T.conj().copy()
            #
            # action of the unitary invariant hamiltonian on the canonic
            # states (stored on psit_nG) is stored on Htpsit_nG
            #
            # compute matrixelements H^0_{ij} in the basis of the canonic
            # states psit_nG
            gemm(self.gd.dv,psit_nG,Htpsit_nG,0.0,self.H0_unn[q],'t')
            self.gd.comm.sum(self.H0_unn[q])
            #
            # add matrix-elements of the ODD-potential
            #
            # transform V_ij from energy optimal basis to the canonic
            # basis
            V_nn = np.dot(np.dot(W_nn,self.V_unn[q]),W_nn.T.conj())
            V_nn += self.H0_unn[q]
            #
            # separate occupied subspace from unoccupied subspace
            V_nn *= np.outer(f,f) + np.outer(1-f,1-f)
            #
            # diagonalize hamiltonian matrix 
            diagonalize(V_nn,eps_n)
            W_nn = np.dot(V_nn,W_nn.copy())
            #
            # store V_ij (basis of new canonic states)
            self.H0_unn[q]  =  np.dot(np.dot(W_nn,self.V_unn[q]),W_nn.T.conj())
            self.H0_unn[q] *=  np.outer(f,f)
            self.Ecorr_q[q] = -np.sum(np.diag(self.H0_unn[q]))*(3-self.nspins)
            #
            # new canonic states are now defined by |k> \mapsto V|k>
            #
            # action of ODD potential V_i|phi_i>
            self.v_unG[q,:] *= self.phit_unG[q,:]
            #
            # action of the canonic ODD potential
            gemm(1.0,self.v_unG[q],W_nn,0.0,self.Htphit_unG[q])
            #
            # setup new canonic states |k>
            gemm(1.0,  psit_nG,V_nn,0.0,self.phit_unG[q])
            #
            for i in range(nbands):
                self.phit_unG[q,i,:]   *= f[i]
                self.Htphit_unG[q,i,:] *= f[i]
            
            #print self.H0_unn[q]
            #
            q_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            H_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            #
            gemm(self.gd.dv,self.phit_unG[q]  ,psit_nG,0.0,q_nn,'t')
            gemm(self.gd.dv,self.Htphit_unG[q],psit_nG,0.0,H_nn,'t')
            self.gd.comm.sum(q_nn)
            self.gd.comm.sum(H_nn)
            #
            V_nn  = H_nn - np.dot(q_nn,self.H0_unn[q])
            #
            gemm(+1.0,self.phit_unG[q]  ,V_nn, 1.0,Htpsit_nG)
            gemm(+1.0,self.Htphit_unG[q],q_nn, 1.0,Htpsit_nG)
            #
            self.setup_unified[q]=False
            #
        else:
            #
            q_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            H_nn   = np.zeros((nbands,self.nbands),dtype=self.dtype)
            #
            gemm(self.gd.dv,self.phit_unG[q]  ,psit_nG,0.0,q_nn,'t')
            gemm(self.gd.dv,self.Htphit_unG[q],psit_nG,0.0,H_nn,'t')
            self.gd.comm.sum(q_nn)
            self.gd.comm.sum(H_nn)
            #
            V_nn  = H_nn - np.dot(q_nn,self.H0_unn[q])            
            #
            gemm(+1.0,self.phit_unG[q]  ,V_nn, 1.0,Htpsit_nG)
            gemm(+1.0,self.Htphit_unG[q],q_nn, 1.0,Htpsit_nG)
            #
        self.timer.stop('ODD - basis action')


    def write_states(self, filename='orbitals',
                     canonic=True, optimal=True):
        #
        nbands =self.nbands
        nblocks=self.nblocks
        nspins =self.hamiltonian.nspins
        atoms  =self.atoms
        #
        myblocks    = self.myblocks
        mybands     = self.mybands
        #
        # the spin polarized case
        if nspins==2:
            for u in range(nblocks):
                for n in range(nbands):
                    #
                    if u in myblocks and n in mybands:
                        #
                        # get the local index of the block u
                        q=self.myblocks.index(u)
                        #
                        if canonic:
                            phi_G=self.wfs.kpt_u[q].psit_nG[n]
                            #
                            write(filename+'-can-%d-%d.cube' % (u,n),
                                      atoms, data=np.abs(phi_G)**2)
                            #
                        if optimal:
                            phi_G=self.hamiltonian.xcfunc.phit_unG[q,n]
                            #
                            write(filename+'-opt-%d-%d.cube' % (u,n),
                                      atoms, data=np.abs(phi_G)**2)
        #
        # no spin polarization
        else:
            for u in range(nblocks):
                for n in range(nbands):
                    #
                    if u in myblocks and n in mybands:
                        #
                        #
                        # get the local index of the block u
                        q=self.myblocks.index(u)
                        #
                        if canonic:
                            phi_G=self.wfs.kpt_u[q].psit_nG[n]
                            #
                            write(filename+'-can-sat-%d.cube' % (n),
                                      atoms, data=np.abs(phi_G)**2)
                            #
                        if optimal:
                            phi_G=self.hamiltonian.xcfunc.phit_unG[q,n]
                            #
                            write(filename+'-opt-sat-%d.cube' % (n),
                                      atoms, data=np.abs(phi_G)**2)

    def get_sic_energy(self):
        #
        return self.Stot*Hartree
        #

    def write_energies(self,t):
        #
        units   = self.units
        nblocks = self.nblocks
        nbands  = self.nbands
        eps_un  = self.eps_un
        f_un    = self.f_un
        Sha_un  = self.Sha_un
        Sxc_un  = self.Sxc_un
        Stot_un = self.Stot_un
        pos_cun = self.pos_un
        #
        if (mpi.rank==0):
            t()
            t("Self Interaction Corrections:")
            t(" blk orb     eps     occ        S_cou       S_xc      S_tot ")
            t("============================================================")
            #
            for u in range(nblocks):
                for n in range(nbands):
                    #
                    if (self.periodic):
                        pos = pos_cun[:,u,n]
                        t("%3d %3d %10.5f %5.2f : %10.5f %10.5f %10.5f"+
                               "%7.2f %7.2f %7.2f  %7.2f" %
                                   (u,n,eps_un[u,n]*units,f_un[u,n],
                                    Sha_un[u,n]*units,Sxc_un[u,n]*units,
                                    Stot_un[u,n]*units,
                                    pos[0],pos[1],pos[2],loc_un[u,n]))
                    else:
                        t("%3d %3d %10.5f %5.2f : %10.5f %10.5f %10.5f" %
                               (u,n,eps_un[u,n]*units,f_un[u,n],
                                Sha_un[u,n]*units,Sxc_un[u,n]*units,
                                Stot_un[u,n]*units))
                t("---------------------------------------------" +
                  "---------------")
            t("  total                  : %10.5f %10.5f %10.5f" %
                   (self.Sha*units, self.Sxc*units, self.Stot*units))
            t() 

    def update_energies(self,blocks=[],bands=[],refresh=False):
        #
        f_un    = self.f_un
        eps_un  = self.eps_un
        Sha_un  = self.Sha_un
        Sxc_un  = self.Sxc_un
        Stot_un = self.Stot_un
        pos_un  = self.pos_un
        loc_un  = self.loc_un
        Sha     = self.Sha
        Sxc     = self.Sxc
        Stot    = self.Stot
        #
        if (refresh):
            # 
            self.update_optimal_states(self.myblocks)
            self.update_potentials(self.myblocks,self.mybands)     
        #
        # accumulate orbital dependent energies
        self.wfs.kpt_comm.sum(f_un)
        self.wfs.kpt_comm.sum(loc_un)
        self.wfs.kpt_comm.sum(pos_un)
        self.wfs.kpt_comm.sum(eps_un)
        self.wfs.kpt_comm.sum(Sha_un)
        self.wfs.kpt_comm.sum(Sxc_un)
        self.wfs.kpt_comm.sum(Stot_un)
        #
        # accumulate total SIC energies
        self.Sha  = np.sum(Sha_un*f_un)
        self.Sxc  = np.sum(Sxc_un*f_un)
        self.Stot = np.sum(Stot_un*f_un)
        #

    def update_potentials(self,blocks=[],bands=[]):
        #
        #
        # check if wavefunctions have already been initialized
        # -> else exit with E_SI=0
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0

        self.timer.start('ODD - potentials')
        #
        # some references to lengthy variable names
        wfs         = self.wfs
        setups      = self.wfs.setups
        density     = self.density
        hamiltonian = self.hamiltonian
        psolver     = self.psolver
        nt_g        = self.nt_g
        nt_g0       = self.nt_g0
        v_g0        = self.v_g0
        nt_G        = self.nt_G
        v_cou_unG   = self.v_cou_unG
        v_unG       = self.v_unG
        e_g         = self.e_g
        v_g         = self.v_g
        v_cou_g     = self.v_cou_g
        nbands      = self.nbands
        nblocks     = self.nblocks
        test        = self.debug
        unit        = self.units
        Sha_un      = self.Sha_un
        Sxc_un      = self.Sxc_un
        Stot_un     = self.Stot_un
        #nocc_u      = self.nocc_u
        m_nt_g      = self.masked_nt_g
        #
        # select blocks which need to be evaluated
        mybands         = self.mybands
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        # loop specified blocks and bands
        npoisson=0
        for u in myblocks:
            #
            # get the local index of the block u (which is
            q = self.myblocks.index(u)
            #
            Sha_un[u,:] = 0.0
            Sxc_un[u,:] = 0.0
            Stot_un[u,:] = 0.0
            self.npart_q[q] = 0.0
            #
            for n in mybands:
                #
                # initialize temporary variables
                # ------------------------------------------------------------
                f        = self.f_un[u,n]
                occ      = self.f_un[u,n]/(3-self.nspins)
                m        = self.m_un[u,n]
                eps      = self.eps_un[u,n]
                v_g[:]   = 0.0
                e_g[:]   = 0.0
                v_g0[:]  = 0.0
                nt_g0[:] = 0.0
                #
                # decide if SIC of unoccupied states should be evaluated
                # ------------------------------------------------------------
                if f!=0.0 or self.virt_SIC:
                    noSIC=False
                else:
                    noSIC=True
                #
                # compose orbital density
                # ------------------------------------------------------------
                self.timer.start('rho')
                if (self.optcmplx):
                    nt_G[:] = (self.phit_unG[q,n,:] **2 +
                               self.phit2_unG[q,n,:]**2)
                else:
                    nt_G[:] = self.phit_unG[q,n,:]**2
                #
                # compose atomic density matrix
#PAW                Q_aL = {}
#PAW                D_ap = {}
#PAW                dH_ap = self.dH_unap[q][n]
#PAW                for a, P_ni in self.P_uani[q].items():
#PAW                    P_i = P_ni[n]
#PAW                    D_ii = np.outer(P_i, P_i)
#PAW                    D_p = pack(D_ii)
#PAW                    D_ap[a]  = D_p
#PAW                    dH_ap[a] = np.zeros_like(D_p)
#PAW                    Q_aL[a] = np.dot(D_p, setups[a].Delta_pL)   
                #
                # interpolate density on the fine grid
                # ------------------------------------------------------------
                if self.finegrid:
                    Nt = self.gd.integrate(nt_G)
                    density.interpolator.apply(nt_G, nt_g)
                    Ntfine = self.finegd.integrate(nt_g)
                    nt_g  *= Nt / Ntfine
#PAW                    density.ghat.add(nt_g, Q_aL)
                else:
                    nt_g = nt_G
#PAW                    self.ghat.add(nt_g, Q_aL)

                self.timer.stop('rho')
                #
                # compute the masked density and its effect on the
                # norm of density
                # ------------------------------------------------------------
                if (self.periodic or self.new_coul):
                    #
                    self.timer.start('localization masks')
                    #
                    if (self.gauss==None):
                        self.gauss     = Gaussian(self.finegd,self.cell,
                                                  self.pbc,self.density)
                        self.rho_gauss = np.ndarray(nt_g.shape,dtype=np.float)
                        self.phi_gauss = np.ndarray(nt_g.shape,dtype=np.float)
                        self.mask      = np.ndarray(nt_g.shape,dtype=np.float)
                    #
                    # calculate positions of the orbitals
                    # --------------------------------------------------------
                    self.pos_un[:,u,n]=self.gauss.get_positions(nt_g)
                    self.gauss.get_fields(self.pos_un[:,u,n],self.rho_gauss,
                                          self.phi_gauss,self.mask)
                    if (self.periodic):
                        #
                        # apply the localization mask to the density
                        m_nt_g = nt_g*self.mask
                        #
                        # ... and compute effect of mask on total norm
                        self.loc_un[u,n] = self.finegd.integrate(m_nt_g)
                    else:
                        self.loc_un[u,n] = 1.0
                    self.timer.stop('localization masks')
                else:
                    #
                    # no mask needed for finite systems
                    self.loc_un[u,n] = 1.0
                #
                # self-interaction correction for E_xc
                # ------------------------------------------------------------
                self.timer.start('Exc')
                if self.excfac==0.0 or noSIC:
                    #
                    Sxc_un[u,n] = 0.0
                    #
                else:
                    #
                    self.xcsic.calculate_spinpolarized(e_g, nt_g, v_g, nt_g0, v_g0)
                    Sxc_un[u,n] = -self.excfac*self.finegd.integrate(e_g)
                    v_g[:]     *= -self.excfac*occ
                    #Sxc_un[u,n] = -self.excfac*np.sum(e_g.ravel())*self.finegd.dv

#PAW                    dSxc = 0.0
#PAW                    for a, D_p in D_ap.items():
#PAW                        D_sp = np.array([D_p, np.zeros_like(D_p)])
#PAW                        dH_sp = np.zeros_like(D_sp)
#PAW                        xccorr = setups[a].xc_correction_sic
#PAW                        dSxc += xccorr.calculate_energy_and_derivatives(
#PAW                                D_sp, dH_sp, a)
#PAW                        dH_ap[a] = -self.excfac * dH_sp[0]

                    #Sxc_un[u, n] -= self.excfac * self.gd.comm.sum(dSxc)
                    #Sxc_un[u, n] -= self.excfac * dSxc
                self.timer.stop('Exc')
                #
                # self-interaction correction for U_Hartree
                # ------------------------------------------------------------
                self.timer.start('Hartree')
                if self.coufac==0.0 or noSIC:
                    #
                    Sha_un[u,n] = 0.0
                    #
                else:
                    #
                    # use the last coulomb potential as initial guess
                    # for the coulomb solver and transform to the fine grid
                    if self.finegrid:
                        density.interpolator.apply(v_cou_unG[q,n], v_cou_g)
                    else:
                        v_cou_g[:]=v_cou_unG[q,n,:]
                    #
                    # initialize the coulomb solver (if necessary) and
                    # solve the poisson equation
                    if (not self.new_coul):
                        #
                        npoisson+=psolver.solve(v_cou_g, nt_g, charge=1, eps=1E-14,
                                      zero_initial_phi=self.init_cou)
                        #
                    else:
                        #
                        self.solve_poisson_charged(v_cou_g,nt_g,self.pos_un[:,u,n],
                                                   self.phi_gauss, self.rho_gauss,
                                                   zero_initial_phi=self.init_cou)
                        #
                    #
                    #
                    # compose the energy density and add potential
                    # contributions to orbital dependent potential.
                    e_g         = nt_g * v_cou_g
                    Sha_un[u,n] = -0.5*self.coufac*self.finegd.integrate(e_g)
                    v_g[:]     -= self.coufac * occ * v_cou_g[:]
                    #Sha_un[u,n] = -0.5 * self.coufac * \
                    #               np.sum(e_g.ravel()) * self.finegd.dv
                    #
                    # add PAW corrections to the self-Hartree-Energy
                    ##Sha_un[u, n] += self.gd.comm.sum(self.paw_sic_hartree_energy(D_ap, dH_ap))
                    #Sha_un[u, n] += self.paw_sic_hartree_energy(D_ap, dH_ap)
                #
                self.timer.stop('Hartree')
                #
                # apply the localization mask to the potentials
                # and set potential to zero for metallic and unoccupied states
                # -------------------------------------------------------------
                if (self.periodic):
                    v_g[:] = v_g[:]*self.mask[:]
                #
                # restrict to the coarse-grid
                # -------------------------------------------------------------
                if self.finegrid:
                    hamiltonian.restrictor.apply(v_g    , v_unG[q,n])
                    hamiltonian.restrictor.apply(v_cou_g, v_cou_unG[q,n])
                else:
                    v_unG[q,n,:]     = v_g[:]
                    v_cou_unG[q,n,:] = v_cou_g[:]
                #
                #
                # accumulate total SIC-energies and number of occupied states
                # in block
                # -------------------------------------------------------------
                #nocc_u[q]    = nocc_u[q]   + occ
                self.npart_q[q] = self.npart_q[q] + occ
                #
            #self.npart_q[q] = int(nocc_u[q]+0.5)
            self.npart_q[q] = int(self.npart_q[q] + 0.5)
        #
        Stot = 0.0
        for u in myblocks:
            Stot_un[u,:] = Sxc_un[u,:] + Sha_un[u,:]
            Stot = Stot + np.sum(Stot_un[u,:]*self.f_un[u,:])
        #
        # SIC potentials are now initialized
        self.init_SIC=False
        self.init_cou=False
        self.npoisson=npoisson
        #
        # decide which bands are metallic
        #if (self.periodic):
        #    self.m_un=np.where(Stot_un>0.0, 0.0, 1.0)
        #    Stot_un  =np.where(Stot_un>0.0, 0.0, Stot_un)
        #    #
        #    for u in myblocks:
        #        q=self.myblocks.index(u)
        #        for n in mybands:
        #            v_unG[q,n][:]     = self.m_un[u,n]*v_unG[q,n,:]
        #
        self.timer.stop('ODD - potentials')
        #
        # return the total orbital dependent energy for the
        # updated states
        #    self.finegd.comm.sum(Sxc_un[u,:])
        #    self.finegd.comm.sum(Sha_un[u,:])
        return Stot

    def paw_sic_hartree_energy(self, D_ap, dH_ap):
        """Calculates the PAW corrections for the SIC Hartree energy.

        returns the PAW correction to the SIC energy and adds corrections
        to the derivatives dH_ap."""

        setups      = self.wfs.setups
        dE = 0.0
        for a, D_p in D_ap.items():
            M_pp = setups[a].M_pp
            dE += np.dot(D_p, np.dot(M_pp, D_p))
            dH_ap[a] -= 2 * self.coufac * np.dot(M_pp, D_p)

        return -self.coufac * dE


    def update_optimal_states(self,blocks=[],rotate_only=False):
        #
        test    = self.debug
        nbands  = self.nbands
        nblocks = self.nblocks
        #
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        self.timer.start('SIC - state update')
        #
        # update the grid representation (and projectors)
        # of the energy optimal orbitals
        for u in myblocks:
            #
            # get the local index of the block u
            q=self.myblocks.index(u)
            #
            # write transformation matrix W
            if test>6:
                #print 'Transformation matrix W for block ',u
                #print self.W_unn[q]
                print 'RMS error in Transformation matrix W_ij for block ',u,'(block ',q,' on node',mpi.rank,')'
                Q=np.sum((np.dot(self.W_unn[q],self.W_unn[q].T.conj())-np.eye(nbands))**2)
                print np.sqrt(abs(Q.real))
            #
            # calculate the energy optimal orbitals |Phi> = W|Psi>
            if self.optcmplx:
                self.phit_unG[q]  = 0.0
                self.phit2_unG[q] = 0.0
                self.Tmp_nn = self.W_unn[q].real.copy()
                gemm(1.0,self.wfs.kpt_u[q].psit_nG,
                     self.Tmp_nn,0.0,self.phit_unG[q])
                self.Tmp_nn = self.W_unn[q].imag.copy()
                gemm(1.0,self.wfs.kpt_u[q].psit_nG,
                     self.Tmp_nn,0.0,self.phit2_unG[q])               
            else:
                self.phit_unG[q] = 0.0
                gemm(1.0,self.wfs.kpt_u[q].psit_nG,self.W_unn[q],
                     0.0,self.phit_unG[q])
            #
            # apply W to projectors
#PAW            P_ani = self.wfs.kpt_u[q].P_ani
#PAW            self.P_uani[q] = self.wfs.pt.dict(self.nbands, zero=True)
#PAW            for a, P_ni in P_ani.items():
#PAW                gemm(1.0, P_ni, self.W_unn[q], 0.0,
#PAW                     self.P_uani[q][a])

            # check overlap matrix of orbitals |Phi>
            if test>6:
                #
                if self.optcmplx:
                    gemm(self.gd.dv,self.phit_unG[q],self.phit_unG[q],
                         0.0,self.Tmp_nn,'t')
                    gemm(self.gd.dv,self.phit2_unG[q],self.phit2_unG[q],
                         1.0,self.Tmp_nn,'t')
                    self.O_unn[q] = self.Tmp_nn
                    gemm(self.gd.dv,self.phit_unG[q],self.phit2_unG[q],
                         0.0,self.Tmp_nn,'t')
                    gemm(self.gd.dv,self.phit2_unG[q],self.phit_unG[q],
                        -1.0,self.Tmp_nn,'t')
                    self.O_unn[q] += 1j*self.Tmp_nn
                else:
                    gemm(self.gd.dv,self.phit_unG[q],self.phit_unG[q],
                         0.0,self.O_unn[q],'c')
                #
#PAW                for a, P_ni in self.P_uani[q].items():
#PAW                    dS_ii = self.wfs.setups[a].dO_ii
#PAW                    self.O_unn[q] += np.dot(P_ni, np.dot(dS_ii, P_ni.T))
                #
                self.gd.comm.sum(self.O_unn[q])
                #
                #print 'Overlap matrix <Phi_i|Phi_j> for block ',u,'(block ',q,' on node',mpi.rank,')'
                #print self.O_unn[q]
                #
                self.O_unn[q] -= np.eye(nbands)
                print 'RMS error in Overlap matrix <Phi_i|Phi_j> for block ',u,'(block ',q,' on node',mpi.rank,')'
                Q = np.sum(self.O_unn[q]*self.O_unn[q].conj())
                print np.sqrt(abs(Q.real))
            #
            # check transformation matrix of orbitals <Psi|Phi> = W
            if test>6:
                #
                psit_nG = self.wfs.kpt_u[q].psit_nG
                if self.optcmplx:
                    gemm(self.gd.dv,psit_nG,self.phit_unG[q],
                         0.0,self.Tmp_nn,'t')
                    self.O_unn[q] = self.Tmp_nn
                    gemm(self.gd.dv,psit_nG,self.phit2_unG[q],
                         0.0,self.Tmp_nn,'t')
                    self.O_unn[q] += 1j*self.Tmp_nn
                else:
                    gemm(self.gd.dv,psit_nG,self.phit_unG[q],
                         0.0,self.O_unn[q],'t')
                #
                self.gd.comm.sum(self.O_unn[q])
                #print 'Transf matrix <Psi_i|Phi_j> for block ',u,'(block ',q,' on node',mpi.rank,')'
                #print self.O_unn[q]
                #
                self.O_unn[q] -= self.W_unn[q]
                print 'RMS error in transf. matrix <Psi_i|Phi_j> for block ',u,'(block ',q,' on node',mpi.rank,')'
                Q = np.sum(self.O_unn[q]*self.O_unn[q].conj())
                print np.sqrt(abs(Q.real))

        if rotate_only:
            self.timer.stop('SIC - state update')
            return
        #
        # single particle energies and occupation factors mapped from the
        # canonic orbitals
        if type(self.wfs.kpt_u[0].eps_n)==None:
            self.eps_un = np.zeros((nblocks,nbands),dtype=np.float)
        else:
            #self.eps_un = np.zeros((nblocks,nbands),dtype=np.float)
            for u in myblocks:
                q=self.myblocks.index(u)
                self.eps_un[u]=self.wfs.kpt_u[q].eps_n
        #
        if type(self.wfs.kpt_u[0].f_n)==None:
            self.f_un = np.ones((nblocks,nbands),dtype=np.float)
        else:
            #self.f_un = np.zeros((nblocks,nbands),dtype=np.float)
            for u in myblocks:
                q=self.myblocks.index(u)
                self.f_un[u] = self.wfs.kpt_u[q].f_n

        #print self.eps_un
        #print self.f_un
        self.timer.stop('SIC - state update')


    def linear_response(self,K_qnn,blocks=[]):
        #
        debug   = self.debug
        nbands  = self.nbands
        #
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        nt_G      = self.nt_G
        rnt_G     = self.rnt_G
        nt_g      = self.nt_g
        rnt_g     = self.rnt_g
        e_g       = self.e_g
        v_cou_g   = self.v_cou_g
        density   = self.density
        psolver   = self.psolver
        dv        = self.gd.dv
        phit_G    = self.phit_G
        #
        #
        # compute linear response to constraint matrix
        #     <phi_i|V_j|phi_j>
        for u in myblocks:
            #
            q         = self.myblocks.index(u)
            phit_mG   = self.phit_unG[q]
            rphit_mG  = self.rphit_unG[q]
            Htphit_mG = self.Htphit_unG[q]
            v_mG      = self.v_unG[q]
            f_m       = self.f_un[u]
            #
            # infinitesimal change to states
            K_norm = np.sqrt(np.sum(K_qnn[q]*K_qnn[q].conj()))
            K_mm   = K_qnn[q].copy() / K_norm
            rK_mm  = K_mm.copy()
            #
            # compute the response-wavefunctions
            gemm(1.0, phit_mG, K_mm, 0.0, rphit_mG)
            #
            # action of ODD potential on response wavefunctions
            Htphit_mG = v_mG * rphit_mG
            gemm(dv, phit_mG, Htphit_mG, 0.0, rK_mm, 'c')
            #
            # action of ODD potential on wavefunctions
            Htphit_mG = v_mG*phit_mG
            gemm(dv, rphit_mG, Htphit_mG, 1.0, rK_mm, 'c')
            #
            # compute response potentials
            self.timer.start('ODD - response potentials')
            for n in range(nbands):
                #
                npoisson = 0
                f        = f_m[n]
                #
                if f!=0.0 or self.virt_SIC:
                    noSIC=False
                else:
                    noSIC=True
                #
                # response density / density
                # --------------------------------------------------------------
                rnt_G[:] = 2.0*rphit_mG[n,:]*phit_mG[n,:]
                nt_G[:]  = phit_mG[n,:]*phit_mG[n,:]
                #
                # interpolate density on the fine grid
                if self.finegrid:
                    rNt = self.gd.integrate(rnt_G)
                    Nt  = self.gd.integrate(nt_G)
                    density.interpolator.apply(rnt_G, rnt_g)
                    density.interpolator.apply(nt_G, nt_g)
                    rNtfine = self.finegd.integrate(rnt_g)
                    Ntfine  = self.finegd.integrate(nt_g)
                    rnt_g   *= rNt / rNtfine
                    nt_g    *=  Nt /  Ntfine
                else:
                    rnt_g[:] = rnt_G[:]
                    nt_g[:]  = nt_G[:]
                #
                # response coulomb potential
                # --------------------------------------------------------------
                if self.coufac!=0.0 and not noSIC:
                    #
                    # use zero response potential as initial guess
                    v_cou_g[:] = 0.0
                    #
                    # initialize the coulomb solver (if necessary) and
                    # solve the poisson equation
                    npoisson += psolver.solve(v_cou_g, rnt_g, charge=0,
                                              zero_initial_phi=False)
                    #
                    # scale coulomb potential
                    v_cou_g[:] *= self.coufac
                #
                # response of the xc-functional
                # --------------------------------------------------------------
                #if self.excfac!=0.0 and not noSIC:
                    #
                    # 
                #
                # restrict to the coarse-grid
                # ----------------------------------------------------------
                if self.finegrid:
                    Htphit_mG[n,:] = 0.0
                    hamiltonian.restrictor.apply(v_cou_g, Htphit_mG[n])
                else:
                    Htphit_mG[n,:] = v_cou_g[:]
            #
            # response from changes in ODD potentials v_i
            # ----------------------------------------------------------
            Htphit_mG *= phit_mG
            gemm(dv, phit_mG, Htphit_mG, 1.0, rK_mm, 'c')
            #print (rK_mm - rK_mm.T.conj())
            self.timer.stop('ODD - response potentials')
            return 1.0/np.trace(np.dot(rK_mm.T.conj(),K_mm))
            #

    def update_unitary_transformation(self,blocks=[]):
        #
        test    = self.debug
        nbands  = self.nbands
        #
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        # compensate for the changes to the orbitals due to
        # last subspace diagonalization
        for u in myblocks:
            #
            # get the local index of the block u (which is
            q    = self.myblocks.index(u)
            f    = self.wfs.kpt_u[q].f_n
            mask = np.outer(f,f) 
            #
            # account for changes to the canonic states
            # during diagonalization of the unitary invariant hamiltonian
            W_nn = self.wfs.kpt_u[q].W_nn
            #W_nn *= mask
            #for n in range(nbands):
            #    if (mask[n,n]==0.0):
            #        W_nn[n,n]=1.0
            #W_nn=ortho(W_nn)
            #
            if (1==0):
                if self.optcmplx:
                    self.Tmp2_nn = self.W_unn[q].real
                    gemm(1.0,W_nn,self.Tmp2_nn,0.0,self.Tmp_nn)
                    self.W_unn[q]=self.Tmp_nn
                    self.Tmp2_nn = self.W_unn[q].imag
                    print self.Tmp2_nn
                    gemm(1.0,W_nn,self.Tmp2_nn,0.0,self.Tmp_nn)
                    self.W_unn[q] += 1j*self.Tmp_nn
                else:
                    gemm(1.0,W_nn,self.W_unn[q],0.0,self.O_unn[q])
                    self.W_unn[q]=self.O_unn[q]
                    #self.W_unn[q]=np.dot(self.W_unn[q],W_nn.T)
            else:
                self.W_unn[q] = np.dot(self.W_unn[q],W_nn)
            #
            # adjust transformation for the occupied states
            self.W_unn[q] = self.W_unn[q]*mask
            for n in range(nbands):
                if (mask[n,n]==0.0):
                    self.W_unn[q,n,n]=1.0
            #
            # reset to unit matrix
            self.wfs.kpt_u[q].W_nn=np.eye(self.nbands)
            #
            # orthonormalize W
            self.W_unn[q]=ortho(self.W_unn[q])
            #            
                       

    def solve_poisson_charged(self,phi,rho,pos,phi0,rho0,
                              zero_initial_phi=False):
        #
        #
        #
        # monopole moment
        q1    = self.finegd.integrate(rho)/np.sqrt(4 * np.pi)
        q0    = self.finegd.integrate(rho0)/np.sqrt(4 * np.pi)
        q     = q1/q0
        #
        self.rho_n     = rho - q * rho0
        #
        if zero_initial_phi==True:
            phi[:]     = 0.0
        else:
            axpy(-q, phi0, phi) # phi -= q * self.phi_gauss
        #
        # Determine potential from neutral density using standard solver
        niter = self.psolver.solve_neutral(phi, self.rho_n)
        #
        # correct error introduced by removing monopole
        axpy(+q, phi0, phi)      # phi += q * self.phi_gauss
        #
        return niter

    def localize(self,blocks=[],nocc=0):
        #
        nbands = self.nbands
        nspins = self.nspins
        iterations = -1
        eps        = 1E-6
        #
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        if nocc==0:
            nocc=nbands
        #
        for u in myblocks:
            #
            # get the local index of the block u
            q=self.myblocks.index(u)
            #
            #
            #print 'SIC: localizing block ',u
            #
            # compute the Wannier integrals
            z_nnc    = self.gauss.get_wannier(self.wfs.kpt_u[q].psit_nG,
                                              self.nt_g0)
            value    = 0.0
            oldvalue = 0.0
            U_nn     = np.identity(nbands)
            #
            i = 0
            while i != iterations:
                value = localize(z_nnc, U_nn)
                #print i, value
                if value - oldvalue < eps:
                    break
                i += 1
                oldvalue = value
            #
            self.W_unn[q] = U_nn.T
        #
        return

    def init_unitary_transformation(self,blocks=[],rattle=0.01,localize=True):
        #
        test   =self.debug
        nbands =self.nbands
        nspins =self.nspins
        #
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        #
        # if localize
        if localize:
            self.localize(blocks)
        #
        # if rattle<=0: use a non-random perturbation
        if rattle<=0:
            #
            # debug output
            if test>5:
                print 'strength of perturbation  : ',-rattle
                print 'list of blocks            : ', blocks
            #
            # loop all blocks in block-list
            for u in myblocks:
                #
                # get the local index of the block u
                q=self.myblocks.index(u)
                #
                G_nn = np.zeros((nbands,nbands),dtype=complex)
                W_nn = np.zeros((nbands,nbands),dtype=self.dtype)
                w_n  = np.zeros((nbands),dtype=float)
                H_nn = np.zeros((nbands,nbands),dtype=complex)
                #
                # setup the generator for the case of real optimization
                # (purely complex generator)
                for n in range(nbands):
                    for m in range(nbands):
                        if n>m:
                            W_nn[n,m] = 0.5/(n+m)*rattle
                            W_nn[m,n] =-0.5/(n+m)*rattle
                G_nn  = 1j*(W_nn - W_nn.T.conj())
                #
                # in case of complex optimization the generator may have
                # additional real contributions to the matrixelements
                if self.optcmplx:
                    for n in range(nbands):
                        for m in range(nbands):
                            if n>m:
                                W_nn[n,m] = 0.5/(n+m)**2*rattle
                                W_nn[m,n] = 0.5/(n+m)**2*rattle
                    G_nn  = G_nn + (W_nn + W_nn.T.conj())
                #
                # setup matrix exponential of generator
                diagonalize(G_nn,w_n)
                #
                w_n = rattle*w_n
                #
                for n in range(nbands):
                    for m in range(nbands):
                        for k in range(nbands):
                            H_nn[n,m] = H_nn[n,m] +                        \
                                        (   cos(w_n[k]) + 1j*sin(w_n[k]))* \
                                        G_nn[k,m]*G_nn[k,n].conj()
                #
                # initialize the unitary transformation
                if self.blk_comm.rank==0:
                    if self.optcmplx:
                        self.W_unn[q] = np.dot(H_nn,self.W_unn[q])
                    else:
                        self.W_unn[q] = np.dot(H_nn.real,self.W_unn[q])
                else:
                    self.W_unn[q,:,:] = 0.0

                self.blk_comm.broadcast(self.W_unn[q], 0)

                if test>5:
                    print 'Initial transformation matrix W for block ',u
                    print self.W_unn[q]
        else:
            #
            # debug output
            if test>5:
                print 'strength of perturbation  : ',rattle
                print 'list of blocks            : ',blocks
            #
            # loop all blocks in block-list
            for u in myblocks:
                #
                # get the local index of the block u
                q=self.myblocks.index(u)
                #
                G_nn = np.zeros((nbands,nbands),dtype=complex)
                w_n  = np.zeros((nbands),dtype=np.float64)
                H_nn = np.zeros((nbands,nbands),dtype=complex)
                #
                # setup the generator for the case of real optimization
                # (purely complex generator)
                W_nn = np.random.rand(nbands,nbands)
                G_nn  = 1j*(W_nn - W_nn.T.conj())
                #
                # in case of complex optimization the generator may have
                # additional real contributions to the matrixelements
                if self.optcmplx:
                    W_nn = np.random.rand(nbands,nbands)
                    G_nn  = G_nn + (W_nn + W_nn.T.conj())
                #
                # setup matrix exponential of generator
                diagonalize(G_nn, w_n)
                #
                w_n = rattle*w_n
                #
                for n in range(nbands):
                    for m in range(nbands):
                        for k in range(nbands):
                            H_nn[n,m] = H_nn[n,m] +                        \
                                        (   cos(w_n[k]) + 1j*sin(w_n[k]))* \
                                        G_nn[k,m]*G_nn[k,n].conj()
                #
                # initialize the unitary transformation
                if self.blk_comm.rank==0:
                    if self.optcmplx:
                        self.W_unn[q] = np.dot(H_nn,self.W_unn[q].copy())
                    else:
                        self.W_unn[q] = np.dot(H_nn.real,self.W_unn[q].copy())
                else:
                    self.W_unn[q,:,:] = 0.0

                self.blk_comm.broadcast(self.W_unn[q], 0)

                if test>2:
                    print 'Initial transformation matrix W for block ',u
                    print self.W_unn[q]


def matrix_exponential(G_nn,dlt):

    """Computes the matrix exponential of an antihermitian operator

        U = exp(dlt*G)

    """
    ndim = G_nn.shape[1]
    w_n  = np.zeros((ndim),dtype=float)

    V_nn = np.zeros((ndim,ndim),dtype=complex)
    O_nn = np.zeros((ndim,ndim),dtype=complex)
    if G_nn.dtype==complex:
        V_nn =  1j*G_nn.real + G_nn.imag
    else:
        V_nn =  1j*G_nn.real

    diagonalize(V_nn,w_n)
    #
    O_nn  = np.diag(np.exp(1j*dlt*w_n))
    #print np.max(np.abs(dlt*w_n))
    #
    if G_nn.dtype==complex:
        U_nn = np.dot(V_nn.T.conj(),np.dot(O_nn,V_nn)).copy()
    else:
        U_nn = np.dot(V_nn.T.conj(),np.dot(O_nn,V_nn)).real.copy()
    #        
    return U_nn

def ortho(W):
    ndim = np.shape(W)[1]
    O = np.dot(W, W.T.conj())
    err = np.sum(np.abs(O - np.eye(ndim)))
    #print err
    if (err<1E-10):
        X = 1.5*np.eye(ndim) - 0.5*O    
    else:
        n = np.zeros(ndim,dtype=float)
        diagonalize(O,n)
        U = O.T.conj().copy()
        nsqrt = np.diag(1.0/np.sqrt(n))
        X = np.dot(np.dot(U, nsqrt), U.T.conj())
    O = np.dot(X, W)
    
    
    return O
