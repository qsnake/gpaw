"""SIC stuff - work in progress!"""

import numpy as np
import ase.units as units
import gpaw.mpi as mpi

from ase import *
from gpaw.utilities import pack, unpack
from gpaw.utilities.blas import axpy, gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XCFunctional
from gpaw.utilities.timing import Timer
from gpaw.poisson import PoissonSolver
from gpaw.sic_gauss import Gaussian

from gpaw.atom.generator import Generator, parameters
from gpaw.utilities import hartree
from math import pi,cos,sin,log10,exp,atan2

class SIC:
    def __init__(self, xcname='LDA',finegrid=False,coufac=1.0, excfac=1.0):
        """Self-Interaction Corrected (SIC) Functionals.

        nspins: int
            Number of spins.

        xcname: string
            Name of LDA/GGA functional which acts as
            a starting point for the construction of
            the self-interaction corrected functional

        """
        #
        # parameters
        self.coufac    = coufac   # coulomb coupling constant
        self.excfac    = excfac   # scaling factor for xc functional.
        self.optcmplx  = False    # complex optimization
        self.FineGrid  = finegrid # use fine-grid for coulomb/xc evaluation
        self.adderror  = False    # add unit-opt. residual to basis-residual
        self.init_rattle = 0.1    # perturbation to the canonic states
        self.virt_SIC  = False    # evaluate SIC for virtual orbitals
        self.parlayout = 1        # parallelization layout
        #   
        self.old_coul  = True
        self.inistep   = 0.45    # trial step length in unitary optimization
        self.uomaxres  = 5E-3    # target accuracy for unitary optimization
        self.uominres  = 1E-1    # minimum accuracy before unitary opt. starts
        self.uorelres  = 1E-2    # same, but relative to basis residual
        self.maxuoiter = 20      # maximum number of unitary opt. iterations
        self.maxlsiter = 30      # maximum number of line-search steps
        self.maxcgiter = 0       # maximum number of CG-iterations
        self.lsinterp  = False    # interpolate for minimum during line search
        #
        # debugging parameters
        self.units     = 27.21   # output units 1: in Hartree, 27.21: in eV
        self.test      = 3       # debug level        
        self.act_SIC   = True    # self-consistent SIC 
        self.use_paw   = False    # apply PAW corrections
        self.paw_matrix= False    # True
        self.paw_proj  = False
        #
        # initialization
        self.dtype     = None    # complex/float 
        self.nbands    = None    # total number of bands/orbitals
        self.nblocks   = None    # total number of blocks
        self.mybands   = None    # list of bands of node's responsibility
        self.myblocks  = None    # list of blocks of node's responsibility
        self.nspins    = None    # number of spins
        self.init_SIC  = True    # SIC functional has to be initialized?
        self.init_cou  = True    # coulomb solver has to be initialized?
        self.active_SIC= False   # SIC is activated
        self.UOsteps   = 0       # number of UO-steps
        self.ESI       = 0.0     # orbital dependent energy (SIC)
        self.RSI       = 0.0     # residual of unitary optimization
        #
        self.xcbasisname = xcname
        self.xcname      = xcname + '-SIC'
        self.xcbasis     = XCFunctional(self.xcbasisname, 2)
        self.gga         = self.xcbasis.gga
        self.mgga        = not True
        self.orbital_dependent = True
        self.hybrid      = 0.0
        self.uses_libxc  = self.xcbasis.uses_libxc
        self.gllb        = False 
        
    def set_non_local_things(self, density, hamiltonian, wfs, atoms,
                             energy_only=False):
 
        self.wfs         = wfs
        self.density     = density
        self.hamiltonian = hamiltonian
        self.atoms       = atoms
        #
        nbands    = wfs.nbands
        nkpt      = wfs.nibzkpts
        nspins    = hamiltonian.nspins
        nblocks   = nkpt*nspins
        mynblocks = len(wfs.kpt_u)
        #
        self.nbands      = nbands
        self.nspins      = nspins
        self.nblocks     = nblocks 
        #
        # make sure that we are doing a gamma-point only
        # or finite system calculation
        if nkpt != 1:
            if mpi.rank==0:
                print 'SIC only implemented for finite or Gamma-Point calculation'
                print '   number of bands/orbitals: ',nbands
                print '   number of k-points      : ',nkpt
                print '   number of spins         : ',nspins
                print '   number of blocks        : ',nblocks
            assert False
        #
        # check for periodicity in any dimension
        pbc = self.atoms.get_pbc()
        if pbc[0] or pbc[1] or pbc[2]:
            self.pbc      = pbc
            self.periodic = True
            self.old_coul = False
        else:
            self.pbc      = pbc
            self.periodic = False
        #
        # SIC always requires the spin-polarized variant of the
        # functional
        if nspins==1:
            self.xcbasis     = XCFunctional(self.xcbasisname, 1)
            self.xcsic       = XCFunctional(self.xcbasisname, 2)
        else:
            self.xcbasis     = XCFunctional(self.xcbasisname, 2)
            self.xcsic       = self.xcbasis
        #
        # check if complex optimization is toggled on
        if (self.optcmplx):
            self.dtype=complex
        else:
            self.dtype=float
        #
        # evaluate ODD-functional on coarse grid or on fine grid
        self.gd          = density.gd
        if self.FineGrid:
            self.finegd  = density.finegd
        else:
            self.finegd  = density.gd
        #
        # summarize
        if mpi.rank==0:
            print 'SIC: complex optimization   :',self.optcmplx
            print 'SIC: fine-grid integrations :',self.FineGrid
            print 'SIC: periodic boundaries    :',self.periodic
            
        if self.parlayout==1:
            if mpi.rank==0:
                print 'SIC: parallelization layout : blocks + space-partition'
            #
            # list of blocks stored on this node
            self.myblocks=[]
            for kpt in wfs.kpt_u:
                if not kpt.s in self.myblocks:
                    self.myblocks.append(kpt.s)
            #
            # full list of bands
            self.mybands     = range(self.nbands)
            
        elif self.parlayout==2:
            if mpi.rank==0:
                print 'SIC: parallelization layout : blocks + states'
            #
            # list of blocks stored on this node
            self.myblocks=[]
            for kpt in wfs.kpt_u:
                if not kpt.s in self.myblocks:
                    self.myblocks.append(kpt.s)
            #
            # list of bands stored on this node
            self.mybands=[]
            for n in range(nbands):
                if n%(mpi.size/nspins)==mpi.rank%(mpi.size/nspins):
                    self.mybands.append(n)

        else:
            if mpi.rank==0:
                print 'SIC: parallelization layout : no parallelization'
            self.myblocks    = range(self.nblocks)
            self.mybands     = range(self.nbands)
        #
        if self.myblocks==[]:
            print 'SIC: warning: nodes ',mpi.rank,' is idle throughout SIC evaluation'
            print '              bands : ',self.mybands
            print '              blocks: ',self.myblocks
        if self.mybands==[]:
            print 'SIC: warning: nodes ',mpi.rank,' is idle throughout SIC evaluation'
            print '              bands : ',self.mybands
            print '              blocks: ',self.myblocks

        #
        if (mpi.rank==0):
            print 'SIC:      CPU blocks     bands'
        mpi.world.barrier()
        for node in range(mpi.size):
            mpi.world.barrier()
            if (node==mpi.rank):
                print ('SIC:     %4d %s %s' % (mpi.rank, str(self.myblocks).ljust(10),
                                               str(self.mybands).ljust(40)))
        #
        mynbands         = len(self.mybands)
        mynblocks        = len(self.myblocks)
        #
        self.mynbands    = mynbands      
        self.mynblocks   = mynblocks
        #
        # list of nodes containing the real-space partitioning for the
        # current k-point (for now we do not mix k-points).
        nodes=self.gd.comm.get_members()
        self.blk_comm = mpi.world.new_communicator(nodes)
        #
        # initialize poisson solver for self-coulomb
        self.psolver     = PoissonSolver(nn=3, relax='J')
        self.psolver.set_grid_descriptor(self.finegd)
        self.psolver.initialize()
        #
        self.Sha         = 0.0
        self.Sxc         = 0.0
        self.Stot        = 0.0
        #
        # real-space representations of densities/WF and fields
        # on the coarse grid
        self.v_unG       = self.gd.empty((mynblocks,mynbands))
        self.v_cou_unG   = self.gd.empty((mynblocks,mynbands))
        self.phit_unG    = self.gd.empty((mynblocks,nbands))
        self.Htphit_unG  = self.gd.empty((mynblocks,nbands))
        self.nt_G        = self.gd.empty()
        
        # Projectors
        self.natoms = atoms.get_number_of_atoms()
        self.P_uani = [[0 for i in range(self.natoms)] \
                          for j in range(self.mynblocks)]
        #
        # utility fields on the fine grid
        # (unless coarse grid integrations are forces)
        self.nt_g        = self.finegd.empty()
        self.nt_g0       = self.finegd.empty()
        self.v_g0        = self.finegd.empty()
        self.e_g         = self.finegd.zeros()
        self.v_g         = self.finegd.zeros()
        self.v_cou_g     = self.finegd.zeros()
        #
        # occupation numbers and single-particle energies
        self.f_un        = np.zeros((nblocks,nbands),dtype=float)
        self.m_un        = np.zeros((nblocks,nbands),dtype=float)
        self.eps_un      = np.zeros((nblocks,nbands),dtype=float)
        self.Sha_un      = np.zeros((nblocks,nbands),dtype=float)
        self.Sxc_un      = np.zeros((nblocks,nbands),dtype=float)
        self.Stot_un     = np.zeros((nblocks,nbands),dtype=float)
        self.nocc_u      = np.zeros((nblocks),dtype=float)
        self.step_u      = np.ones((mynblocks),dtype=float)
        self.step_u      = self.step_u*self.inistep
        #
        # transformation from canonic to energy optimal states
        self.W_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.init_unitary_transformation(blocks=self.myblocks,
                                         rattle=self.init_rattle)
        #
        # constraint matrix
        self.L_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.K_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.V_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.O_unn       = np.zeros((mynblocks,nbands,nbands),dtype=self.dtype)
        self.Tmp_nn      = np.zeros((mynblocks,nbands),dtype=self.dtype)

        self.normK_u     = np.zeros((mynblocks),dtype=float)
        self.RSI_u       = np.zeros((mynblocks),dtype=float)
        self.ESI_u       = np.zeros((mynblocks),dtype=float)
        self.npart_u     = np.zeros((mynblocks),dtype=int)
        #
        # setup the timers
        self.timer       = self.hamiltonian.timer
        #
        # TEST stuff
        self.Rgrid       = None
        self.Rphase      = None
        self.gauss       = None
        self.mask        = None
        self.cell        = atoms.cell/Bohr
        self.rcell       = 2.0*pi*np.linalg.solve(self.cell.T, np.eye(3))
        self.pos_un      = np.zeros((3,nblocks,nbands),dtype=float)
        self.loc_un      = np.zeros((nblocks,nbands),dtype=float)
        self.rho_gauss   = None
        self.phi_gauss   = None
        self.rho_n       = None
        #
        if (self.periodic):
            self.masked_nt_g = self.finegd.empty()
            self.rho_n       = self.finegd.empty()
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
        return 0.0

    def adjust_non_local_residual(self, pR_G, dR_G, kpt, n):
        pass

    def get_non_local_force(self, kpt):
        return 0.0
    
    def get_non_local_energy(self, n_g=None, a2_g=None, e_LDAc_g=None,
                             v_LDAc_g=None, v_g=None, deda2_g=None):
        return 0.0
    
    def calculate_spinpaired(self, e_g, n_g, v_g, a2_g=None, deda2_g=None):
        #
        # the LDA/GGA part of the functional
        self.xcbasis.calculate_spinpaired(e_g, n_g, v_g, a2_g, deda2_g)
        #
        # orbital dependent components of the functional
        if n_g.ndim == 3:
            #
            self.calculate_sic_potentials()
            self.unitary_optimization()
            #
            # only node 0 of grid communicator writes the total SIC to
            # the grid-point (0,0,0)
            if self.finegd.comm.rank == 0:
                #assert e_g.ndim == 3
                if e_g.ndim == 3:
                    e_g[0, 0, 0] += self.Stot / self.finegd.dv
        

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                                a2_g=None,  aa2_g=None, ab2_g=None,
                                deda2_g=None, dedaa2_g=None, dedab2_g=None):
        self.xcbasis.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                        a2_g, aa2_g, ab2_g,
                                        deda2_g, dedaa2_g, dedab2_g)
        if na_g.ndim == 3:
            #
            self.calculate_sic_potentials()
            self.unitary_optimization()
            #
            # only one single node writes the total SIC to
            # the grid-point (0,0,0)
            if self.finegd.comm.rank == 0:
                # assert e_g.ndim == 3
                if e_g.ndim == 3:
                    e_g[0, 0, 0] += self.Stot / self.finegd.dv


    def calculate_sic_potentials(self,blocks=[]):
        #
        # check if wavefunctions have already been initialized
        # -> else exit with E_SI=0
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
        #
        # check if ODD-functional is active
        if self.wfs.eigensolver.error > self.uominres:
            if not self.active_SIC:
                return 0.0
        else:
            self.active_SIC = True
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
        test        = self.test
        unit        = self.units
        #
        # select blocks which need to be evaluated
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        mybands     = self.mybands
        #
        # compensate changes to the canonic basis throughout
        # subspace diagonalization
        #
        self.update_unitary_transformation(myblocks)
        #
        # update the energy optimal states
        self.update_optimal_states(myblocks)
        #
        # update the SIC potentials and energies
        self.update_potentials(myblocks,mybands)
        #
        # distribute SIC energies
        self.distribute_energies()
        #
        # print self-interaction energies
        if test > 3:
            self.write_energies(myblocks,mybands)
        #
        return self.Stot

    def calculate_sic_matrixelements(self,blocks=[]):        
        #
        # check if wavefunctions have already been initialized
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
        #
        # start the timer
        self.timer.start('SIC - matrixelements')
        #
        # check if SIC potentials have already been initialized
        if self.init_SIC:
            return 0.0
        #
        # select blocks which need to be evaluated
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
            
        mybands     = self.mybands
        #
        # some references to lengthy variable names
        wfs         = self.wfs
        setups      = self.wfs.setups
        density     = self.density
        hamiltonian = self.hamiltonian
        psolver     = self.hamiltonian.poisson
        nt_g        = self.nt_g
        nt_g0       = self.nt_g0
        v_g0        = self.v_g0
        nt_G        = self.nt_G
        v_cou_unG   = self.v_cou_unG
        v_unG       = self.v_unG
        #
        # action of the SIC potentials
        self.Htphit_unG   = self.v_unG*self.phit_unG
        #
        # evaluate symmetrized V_ij and Kappa_ij 
        for u in myblocks:
            #
            # get the local index of the block u 
            q=self.myblocks.index(u)
            #
            # calculate SIC matrix <Phi_i|V_j|Phi_j>
            gemm(self.gd.dv,self.phit_unG[q],self.Htphit_unG[q],0.0,self.V_unn[q],'c')
            self.gd.comm.sum(self.V_unn[q])
            #
            # add PAW corrections
            if self.paw_matrix:
                self.V_unn[q] -= self.paw_sic_matrix(q)

            # apply subspace mask
            for n in range(self.nbands):
                for m in range(self.nbands):
                    if ((n<self.npart_u[q] and m>=self.npart_u[q]) or
                        (n>=self.npart_u[q]  and m<self.npart_u[q])):
                        self.V_unn[q,n,m]=0.0
            #
            # symmetrization of V and kappa-Matrix
            self.K_unn[q] = 0.5*(self.V_unn[q] - self.V_unn[q].T.conj())
            self.V_unn[q] = 0.5*(self.V_unn[q] + self.V_unn[q].T.conj())
            #print 'SIC-potential matrix <Phi_i|V_j|Phi_j> of block ',q,' on CPU',mpi.rank
            #print self.K_unn[q]
            #
            # total norm of kappa-matrix
            self.normK_u[q] = np.sum(self.K_unn[q]*self.K_unn[q].conj())
        #
        # stop the timer
        self.timer.stop('SIC - matrixelements')

    def paw_sic_matrix(self, u):
        """Calculates the PAW correction for the 
        <Phi_n1 | V_n2 | Phi_n2> Hartree term"""
        
        if not self.use_paw:
            return 0
            
        wfs     = self.wfs
        natoms = self.natoms
        nbands  = wfs.nbands
        setups  = wfs.setups
        dV_nn = np.zeros((nbands,nbands))
    
        for a in range(natoms):
            P_ni = self.P_uani[u][a]
            M0_pp = setups[a].M0_pp
            #M0_pp = setups[a].M_pp
            for n in range(nbands):
                P_i = P_ni[n]
                D_ii = np.outer(P_i, P_i)
                D_p = pack(D_ii)
                dH_ii = 2*unpack(np.dot(M0_pp, D_p))
                dV_nn[:,n] = np.dot(P_ni, np.dot(dH_ii, P_i))
        return dV_nn


    def unitary_optimization(self):
        #
        test    =self.test
        myblocks=self.myblocks
        #
        # skip unitary optimization if initialization is not finished
        if self.init_SIC:
            return
        #
        # get the basis error from the eigensolver
        basiserror=self.wfs.eigensolver.error
        localerror=0.0
        #
        # check if ODD-functional is active
        if not self.active_SIC:
            return
        #
        # allocate temporary arrays
        U_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        O_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        #
        # prepare logging
        if test>2 and mpi.rank==0:
            print ("================  unitary optimization  ===================")
        #
        # loop all blocks
        for u in myblocks:
            #
            # get the local index of the block u 
            q=self.myblocks.index(u)
            #
            # logging
            if test>6:
                print 'CPU ',mpi.rank,' optimizing block ',u,\
                ' (local index ',q,')'
            #
            # skip blocks with 1 or 0 particles
            if self.npart_u[q]<=1:
                continue
            #
            # the unitary optimization iteration
            #
            epsstep  = 0.005  # 0.005
            dltstep  = 0.1   # 0.1
            #
            # the search direction (initially the gradient)
            D_nn  = self.W_unn[q].copy()
            D_old = 0.0*self.W_unn[q].copy()
            #
            # save the last energy
            #matrix_exponential(self.K_unn[q],U_nn,0.0)
            #matrix_multiply(U_nn,self.W_unn[q].copy(),self.W_unn[q]) 
            self.update_optimal_states([u],rotate_only=True)
            E0=self.update_potentials([u])
            self.calculate_sic_matrixelements([u])
            ESI_init = E0
            optstep  = 0.0
            Gold     = 0.0
            cgiter   = 0
            #
            for iter in range(self.maxuoiter):
                #
                # create a copy of the initial unitary transformation
                # and orbital dependent energies
                W_old    = self.W_unn[q].copy()
                K_old    = self.K_unn[q].copy()
                ESI_old  = E0
                lsiter   = 0
                lsmethod = 'undefined'
                #
                #
                # length of gradient at current configuration
                G0   = np.trace(np.dot(self.K_unn[q].T,self.K_unn[q]))
                G0cg = G0
                #
                #
                if (Gold!=0.0):
                    beta = G0/Gold
                    D_nn[:,:] = self.K_unn[q][:,:] + beta*D_old
                    G0        = np.trace(np.dot(D_nn.T,D_nn.T))
                    step = epsstep/sqrt(abs(G0))
                else:
                    D_nn[:,:] = self.K_unn[q][:,:]
                    step = epsstep/sqrt(G0)
                #
                updated  = False
                minimum  = False
                failed   = True
                #
                # the "infinitesimal" trial step
                if (epsstep!=0.0):
                    #
                    # infinitesimal steepest descent 
                    #matrix_exponential(self.K_unn[q],U_nn,step)
                    matrix_exponential(D_nn,U_nn,step)
                    matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                    self.update_optimal_states([u],rotate_only=True)
                    E1 = self.update_potentials([u])
                    self.calculate_sic_matrixelements([u])
                    #
                    # length of the gradient at the new position
                    #G1 = np.trace(np.dot(K_old.T,self.K_unn[q]))
                    G1 = np.trace(np.dot(D_nn.T,self.K_unn[q]))
                    #
                    # compute the optimal step size
                    optstep = step*G0/(G0-G1)
                    #
                    #print 'trial step: ',step,optstep,E1-E0,G0,G1
                    #
                    # decide on the method for stepping
                    if (optstep > 0.0):
                        #matrix_exponential(K_old,U_nn,optstep)
                        matrix_exponential(D_nn,U_nn,optstep)
                        matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                        self.update_optimal_states([u],rotate_only=True)
                        E1=self.update_potentials([u])
                        if (E1<ESI_old or True):
                            self.calculate_sic_matrixelements([u])
                            lsiter   = 0
                            lsmethod = 'convex'
                            updated  = True
                            minimum  = True
                            failed   = False
                            ESI      = E1
                        else:
                            optstep  = 0.0
                            self.K_unn[q][:]=K_old[:]
                    else:
                        optstep  = 0.0
                        self.K_unn[q][:]=K_old[:]
                    #
                if (optstep==0.0):
                    #
                    # we are in the concave region, just follow the gradient
                    lsmethod='concave'
                    #
                    ###matrix_exponential(self.K_unn[q],U_nn,0.0)
                    ##matrix_exponential(D_nn,U_nn,0.0)
                    ##matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                    #self.W_unn[q,:]=W_old[:]
                    #self.update_optimal_states([u],rotate_only=True)
                    #E0=self.update_potentials([u])
                    #
                    step    = dltstep/epsstep*step
                    #while (1==1):
                    #    matrix_exponential(self.K_unn[q],U_nn,step)
                    #    matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                    #    self.update_optimal_states([u],rotate_only=True)
                    #    E1 = self.update_potentials([u])
                    #    if (E1<E0+dE):
                    #        Gold = -1.0
                    #        G    = (E1-E0)/step
                    #        break
                    #    step = 0.5*step
                    #    print step,E1-E0,dE
                    #
                    #matrix_exponential(self.K_unn[q],U_nn,step)
                    matrix_exponential(D_nn,U_nn,step)
                    matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                    self.update_optimal_states([u],rotate_only=True)
                    E1 = self.update_potentials([u])
                    #
                    #
                    if (E1>E0 and False):
                        optstep   = 0.0#epsstep
                        updated   = False
                        minimum   = False
                        failed    = False
                        lsmethod  = 'failed'
                        maxlsiter = -1
                    else:
                        maxlsiter = self.maxlsiter
                        
                    G       = (E1-E0)/step
                    step0   = 0.0
                    step1   = step
                    step2   = 2*step
                    #
                    for lsiter in range(maxlsiter):
                        #
                        # energy at the new position
                        #matrix_exponential(K_old,U_nn,step2)
                        matrix_exponential(D_nn,U_nn,step2)
                        matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
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
                            step0   = step1
                            step1   = step2
                            step2   = step2 + step/(max(abs(G),1.0))
                            E0=E1
                            E1=E2
                #
                if (cgiter!=0):
                    lsmethod = lsmethod + ' CG'
                #
                if (cgiter>=self.maxcgiter):
                    Gold   = 0.0
                    cgiter = 0
                else:
                    Gold        = G0cg
                    cgiter      = cgiter + 1
                    D_old[:,:]  = D_nn[:,:]
                #
                # update the energy and matrixelements of V and Kappa
                # and accumulate total residual of unitary optimization
                if (not updated):
                    #matrix_exponential(K_old,U_nn,optstep)
                    if (optstep==0.0):
                        self.W_unn[q,:] =W_old[:] 
                        #matrix_exponential(D_nn,U_nn,optstep)
                        #matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                        self.update_optimal_states([u],rotate_only=True)
                        self.calculate_sic_matrixelements([u])
                        failed = True
                        ESI=E0
                    else:
                        matrix_exponential(D_nn,U_nn,optstep)
                        matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                        self.update_optimal_states([u],rotate_only=True)
                        ESI=self.update_potentials([u])
                        self.calculate_sic_matrixelements([u])
                        failed = False

                E0=ESI
                #
                # orthonormalize the energy optimal orbitals
                self.W_unn[q] = ortho(self.W_unn[q])
                #
                # 
                self.RSI_u[q]=self.normK_u[q]
                #
                # logging
                dE2=max(abs(ESI-ESI_old),1.0E-16)
                dE =max(abs(ESI-ESI_init),1.0E-16)
                K =sqrt(max(self.RSI_u[q],1.0E-16))
                if test>2:
                    if self.finegd.comm.rank == 0:
                        print(" UO-iter %3i : %10.5f  %5.1f %5.1f %6.3f %3i %s" %
                              (iter+1,ESI*self.units,log10(dE2),log10(K),
                               optstep,lsiter+1,lsmethod))
            
                if K<basiserror*self.uorelres or K<self.uomaxres or failed :
                    localerror = localerror + K**2
                    break
            
        if test>2 and mpi.rank==0:
            print ("============  finished unitary optimization  ==============")
        if test>1 and mpi.rank==0:
            print (" initial ODD-energy : %10.5f" %
                  ( ESI_init*self.units))
            print (" final   ODD-energy : %10.5f %5.1f %5.1f" %
                  (  ESI*self.units,log10(dE),log10(K)))
        #
        # add residual of unitary optimization to basis error to
        # avoid premature exit of basis optimization
        if self.adderror:
            self.wfs.eigensolver.error = sqrt(0.5*(basiserror**2 + localerror))
                
    def add_non_local_terms(self, psit_nG, Htpsit_nG, u):
        #
        if self.init_SIC:
            return 
        #
        if not self.act_SIC:
            return
        #
        # check if ODD-functional is active
        if not self.active_SIC:
            return
        #
        # start the timer
        self.timer.start('SIC - basis action')
        #
        q=self.myblocks.index(u)
        #
        nbands=psit_nG.shape[0]
        U_nn = np.zeros((nbands,self.nbands),dtype=self.dtype)
        #V_nn = np.zeros((nbands,self.nbands),dtype=psit_nG.dtype)
        #
        # project psit_nG to the energy optimal states
        gemm(self.gd.dv,self.phit_unG[q],psit_nG,0.0,U_nn,'c')
        #print U_nn
        #gemm(self.wfs.gd.dv,psit_nG,self.phit_unG[q],0.0,U_nn,'c')
        #OLD:self.density.gd.comm.sum(U_nn)
        self.gd.comm.sum(U_nn)
        #
        # NEW:
        # action of the orbital dependent potentials on the
        # energy optimal states
        self.Htphit_unG[q]   = self.v_unG[q]*self.phit_unG[q]
        #
        # action of the unified potential on the canonic states
        gemm(1.0,self.Htphit_unG[q],U_nn,1.0,Htpsit_nG)
        # OLD:
        #gemm(1.0,self.V_unn[q],U_nn,0.0,V_nn)
        ##
        ## accumulate action of the orbital dependent operator
        #gemm(1.0,self.phit_unG[q],V_nn,1.0,Htpsit_nG)
        #
        self.timer.stop('SIC - basis action')


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
            

    def distribute_energies(self):
        #
        #
        f_un    = self.f_un
        eps_un  = self.eps_un
        Sha_un  = self.Sha_un
        Sxc_un  = self.Sxc_un
        Stot_un = self.Stot_un
        #
        # accumulate orbital dependent energies 
        self.wfs.kpt_comm.sum(f_un)
        self.wfs.kpt_comm.sum(eps_un)
        self.wfs.kpt_comm.sum(Sha_un)
        self.wfs.kpt_comm.sum(Sxc_un)
        self.wfs.kpt_comm.sum(Stot_un)
        #
        # accumulate total SIC energies
        self.Sha  = (Sha_un*f_un).sum()
        self.Sxc  = (Sxc_un*f_un).sum()
        self.Stot = (Stot_un*f_un).sum()

    def write_energies(self,blocks=[],bands=[]):
        #
        units   = self.units
        nblocks = self.nblocks
        nbands  = self.nbands
        eps_un  = self.eps_un
        f_un    = self.f_un
        pos_un  = self.pos_un
        loc_un  = self.loc_un
        Sha_un  = self.Sha_un
        Sxc_un  = self.Sxc_un
        Stot_un = self.Stot_un
        Sha     = self.Sha
        Sxc     = self.Sxc
        Stot    = self.Stot
        #
        pos     = np.ndarray((3),dtype=float)
        #
        # write the header
        if (mpi.rank==0):
            print ("=============  self-interaction corrections ===============")
            print (" blk orb   eps occ :      S_cou        S_xc      S_tot     ") 
            print ("-----------------------------------------------------------")
            #
            # loop all states
            for u in range(nblocks):
                for n in range(nbands):
                    #
                    if (self.periodic):
                        pos = pos_un[:,u,n]
                    else:
                        pos = np.dot(self.cell.T,pos_un[:,u,n])
                    #
                    print ("%3d %3d %8.3f %5.2f : %8.3f  %8.3f %8.3f   %7.2f %7.2f %7.2f  %7.2f" %
                          (u,n,eps_un[u,n]*units,f_un[u,n],
                           Sha_un[u,n]*units,Sxc_un[u,n]*units,
                           Stot_un[u,n]*units,
                           pos[0],pos[1],pos[2],loc_un[u,n]))
            #
            # write the footer
            print ("-----------------------------------------------------------")
            print ("          total        : %10.5f  %10.5f %10.5f" %
                   (Sha*units,Sxc*units,Stot*units))
            print ("===========================================================")

    def update_position(self,dens):
        #
        #
        # phase fields have to be initialized?
        if (self.Rgrid==None):
            #
            self.Rgrid  = self.wfs.gd.get_grid_point_coordinates()
            self.Rphase = np.zeros(np.shape(self.Rgrid),dtype=complex)
            R = self.Rgrid
            P = self.Rphase
        
            for i in range(3):
                if (self.atoms.pbc[i]):
                    #
                    # periodic dimension
                    self.Rphase[i,:] = np.exp(1j*(
                        self.rcell[i,0]*self.Rgrid[0,:] +\
                        self.rcell[i,1]*self.Rgrid[1,:] +\
                        self.rcell[i,2]*self.Rgrid[2,:]))
                else:
                    #
                    # finite dimension 
                    self.Rphase[i,:] = \
                        self.rcell[i,0]*self.Rgrid[0,:] +\
                        self.rcell[i,1]*self.Rgrid[1,:] +\
                        self.rcell[i,2]*self.Rgrid[2,:]
        #
        # calculate the direct cordinates x_i of the positions of the
        # orbitals (r_0 = \sum x_i R_i)
        cpos = np.zeros((3),dtype=complex)
        rpos = np.zeros((3),dtype=float)
        for i in range(3):
            #
            cpos[i] = self.density.gd.integrate(self.Rphase[i,:]*dens[:])
            #
            if (self.atoms.pbc[i]):
                if (abs(cpos[i])>1E-12):
                    rpos[i] = atan2(cpos[i].imag,cpos[i].real)/(2.0*pi)
                    if (rpos[i]<0.0):
                        rpos[i] = rpos[i]+1.0
                else:
                    rpos[i] = 0.5
            else:
                rpos[i] = cpos[i].real/(2.0*pi)
            #
        #
        # return the direct coordinates of the positions
        # (cartesian positions are evaluated during output only)
        return rpos
        #
        # calculate the positions of the orbitals
        # return np.dot(self.cell.T,rpos)
        #
        
    def update_masked_density(self,dens,mdens,pos):
        #
        fac = -0.5/self.maskwidth**2
        nx  = dens.shape[0]
        ny  = dens.shape[1]
        nz  = dens.shape[2]
        maskx = np.ones((nx),dtype=float)
        masky = np.ones((ny),dtype=float)
        maskz = np.ones((nz),dtype=float)
        dx  = 1.0/nx
        dy  = 1.0/ny
        dz  = 1.0/nz
        #
        # setup the mask in the three spatial directions
        if (self.atoms.pbc[0]):
            for ix in range(nx):
                x   = ix*dx
                dlt = min(min(abs(x-pos[0]),abs(x-pos[0]+1)),abs(x-pos[0]-1))
                if (dlt > 0.25):
                    maskx[ix] = exp(fac*dlt**2)
        
        if (self.atoms.pbc[1]):
            for ix in range(ny):
                x   = ix*dy
                dlt = min(min(abs(x-pos[1]),abs(x-pos[1]+1)),abs(x-pos[1]-1))
                if (dlt > 0.25):
                    masky[ix] = exp(fac*dlt**2)
                
        if (self.atoms.pbc[2]):
            for ix in range(nz):
                x   = ix*dz
                dlt = min(min(abs(x-pos[2]),abs(x-pos[2]+1)),abs(x-pos[2]-1))
                if (dlt > 0.25):
                    maskz[ix] = exp(fac*dlt**2)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    mdens[ix,iy,iz] = dens[ix,iy,iz]*\
                                      maskx[ix]*masky[iy]*maskz[iz]
        #
        
    def update_potentials(self,blocks=[],bands=[]):
        #
        #
        # check if wavefunctions have already been initialized
        # -> else exit with E_SI=0
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
        
        self.timer.start('SIC - potentials')
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
        test        = self.test
        unit        = self.units
        Sha_un      = self.Sha_un
        Sxc_un      = self.Sxc_un
        Stot_un     = self.Stot_un
        nocc_u      = self.nocc_u
        m_nt_g      = self.masked_nt_g
        #
        # select blocks which need to be evaluated
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        mybands      = self.mybands
        #
        Sha_un[:,:]  = 0.0
        Sxc_un[:,:]  = 0.0
        Stot_un[:,:] = 0.0
        nocc_u[:]    = 0.0
        #
        # loop specified blocks and bands
        for u in myblocks:
            #
            # get the local index of the block u (which is
            q=self.myblocks.index(u)
            #
            #print 'updating potentials',mpi.rank,q,u
            #
            for n in mybands:
                #
                # initialize temporary variables
                # ------------------------------------------------------------
                f        = self.f_un[u,n]
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
                nt_G     = np.abs(self.phit_unG[q,n])**2  
                #
                # interpolate density on the fine grid
                # ------------------------------------------------------------
                if self.FineGrid:
                    Nt = self.gd.integrate(nt_G)
                    density.interpolator.apply(nt_G, nt_g)
                    Ntfine = self.finegd.integrate(nt_g)
                    nt_g  *= Nt / Ntfine
                else:
                    nt_g[:] = nt_G[:] 
                self.timer.stop('rho')
                #
                # compute the masked density and its effect on the
                # norm of density
                # ------------------------------------------------------------
		if (self.periodic or not self.old_coul):
                    #
		    self.timer.start('localization masks')
		    #
                    if (self.gauss==None):
                        self.gauss     = Gaussian(self.finegd,self.cell,self.pbc)
                        self.rho_gauss = np.ndarray(nt_g.shape,dtype=float)
                        self.phi_gauss = np.ndarray(nt_g.shape,dtype=float)
                        self.mask      = np.ndarray(nt_g.shape,dtype=float)
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
                    Sxc_un[u,n] = 0.0
                else:
                    self.xcsic.calculate_spinpolarized(e_g, nt_g, v_g, nt_g0, v_g0)
                    Sxc_un[u,n] = -self.excfac*self.finegd.integrate(e_g)
                    v_g[:]     *= -self.excfac
                #
                self.timer.stop('Exc')
                #
                # self-interaction correction for U_Hartree
                # ------------------------------------------------------------
                self.timer.start('Hartree')
                if self.coufac==0.0 or noSIC:
                    Sha_un[u,n] = 0.0
                else:
                    #
                    # use the last coulomb potential as initial guess
                    # for the coulomb solver and transform to the fine grid
                    if self.FineGrid:
                        density.interpolator.apply(v_cou_unG[q,n], v_cou_g)
                    else:
                        v_cou_g[:]=v_cou_unG[q,n][:]
                    #
                    # initialize the coulomb solver (if necessary) and
                    # solve the poisson equation
                    if (self.old_coul):
                        #
                        psolver.solve(v_cou_g, nt_g, charge=1, #eps=2E-10,
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
                    v_g[:]      = v_g[:] - self.coufac * v_cou_g[:]
                    #
                    # add PAW corrections to the self-Hartree-Energy
                    Sha_un[u,n] -= self.coufac*self.paw_sic_hartree_energy(u,n)
                #
                self.timer.stop('Hartree')
                #
                # apply the localization mask to the potentials
                # and set potential to zero for metallic and unoccupied states
                # -------------------------------------------------------------
                if (self.periodic):
                    v_g[:] = f*v_g[:]*self.mask[:]
                #
                # restrict to the coarse-grid
                # -------------------------------------------------------------
                if self.FineGrid:
                    hamiltonian.restrictor.apply(v_g    , v_unG[q,n])
                    hamiltonian.restrictor.apply(v_cou_g, v_cou_unG[q,n])
                else:
                    v_unG[q,n][:]     = v_g[:]
                    v_cou_unG[q,n][:] = v_cou_g[:]
                #
                #
                # accumulate total SIC-energies and number of occupied states
                # in block
                # -------------------------------------------------------------
                Stot_un[u,n] = Sxc_un[u,n] + Sha_un[u,n]
                nocc_u[q]    = nocc_u[q]   + f/(3-self.nspins)
                #
            self.npart_u[q] = int(nocc_u[q]+0.5)
        #
        # SIC potentials are now initialized
        self.init_SIC=False
        self.init_cou=False
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
        self.timer.stop('SIC - potentials')
        #
        # return the total orbital dependent energy for the
        # updated states
        return (Stot_un*self.f_un).sum()

    def paw_sic_hartree_energy(self, u, n):
        """Calculates the PAW corrections for the SIC Hartree energy for band n"""

        if not self.use_paw:
            return 0

        natoms = self.natoms
        wfs         = self.wfs
        setups      = self.wfs.setups
        dE_a = np.zeros(natoms)
        for a in range(natoms):
            P_i = self.P_uani[u][a][n]
            D_ii = np.outer(P_i, P_i)
            M0_pp = setups[a].M0_pp
            D_p = pack(D_ii)
            dE_a[a] = np.dot(D_p, np.dot(M0_pp, D_p))

        return dE_a.sum()

        
        
    def update_optimal_states(self,blocks=[],rotate_only=False):
        #
        test    = self.test
        nbands  = self.nbands
        nblocks = self.nblocks
        
        #
        self.timer.start('SIC - state update')
        #
        # update the grid representation (and projectors)
        # of the energy optimal orbitals
        for u in blocks:
            #
            # get the local index of the block u 
            q=self.myblocks.index(u)
            #print 'updating states ',q
            #
            # write transformation matrix W 
            if test>4:
                print 'Transformation matrix W for block ',u
                print self.W_unn[q]
            #
            # calculate the energy optimal orbitals |Phi> = W|Psi>
            if self.optcmplx:
                gemm(1.0,(self.wfs.kpt_u[q].psit_nG+0j).copy(),self.W_unn[q],0.0,self.phit_unG[q])
            else:
                gemm(1.0,self.wfs.kpt_u[q].psit_nG,self.W_unn[q],0.0,self.phit_unG[q])
            #
            # apply W to projectors
            natoms = self.natoms
            if self.use_paw:
                P_ani = self.wfs.kpt_u[q].P_ani
                if self.paw_proj:            
                    for a in range(natoms):
                        self.P_uani[q][a] = np.zeros(P_ani[a].shape)
                        gemm(1.0,P_ani[a].copy(),self.W_unn[q],0.0,self.P_uani[q][a])
            
            # check overlap matrix of orbitals |Phi>
            if test>5:
                gemm(self.gd.dv,self.phit_unG[q],self.phit_unG[q],0.0,self.O_unn[q],'c')
                self.gd.comm.sum(self.O_unn[q])
                print 'Overlap matrix <Phi_i|Phi_j> for block ',q
                print self.O_unn[q]
                
        if rotate_only:
            self.timer.stop('SIC - state update')
            return
        #
        # single particle energies and occupation factors mapped from the
        # canonic orbitals
        if type(self.wfs.kpt_u[0].eps_n)==None:
            self.eps_un = np.zeros((nblocks,nbands),dtype=float)
        else:
            self.eps_un = np.zeros((nblocks,nbands),dtype=float)
            for u in blocks:
                q=self.myblocks.index(u)
                self.eps_un[u]=self.wfs.kpt_u[q].eps_n
        #
        if type(self.wfs.kpt_u[0].f_n)==None:
            self.f_un = np.ones((nblocks,nbands),dtype=float)  
        else:
            self.f_un = np.zeros((nblocks,nbands),dtype=float)
            for u in blocks:
                q=self.myblocks.index(u)
                self.f_un[u] = self.wfs.kpt_u[q].f_n
                
        self.timer.stop('SIC - state update')


    def update_unitary_transformation(self,blocks=[]):
        #
        test    = self.test
        nbands  = self.nbands
        #
        # compensate for the changes to the orbitals due to
        # last subspace diagonalization
        for u in blocks:
            #
            # get the local index of the block u (which is
            q=self.myblocks.index(u)
            #
            if self.optcmplx:
                self.Tmp_nn = self.wfs.kpt_u[q].W_nn + 1j*0.0
            else:
                self.Tmp_nn = self.wfs.kpt_u[q].W_nn
            #
            # account for changes to the canonic states
            # during diagonalization of the unitary invariant hamiltonian
            gemm(1.0,self.Tmp_nn,self.W_unn[q],0.0,self.O_unn[q])
            self.W_unn[q]=self.O_unn[q].copy()
            #
            # reset to unit matrix
            self.wfs.kpt_u[q].W_nn=np.eye(self.nbands)

    def solve_poisson_charged(self,phi,rho,pos,phi0,rho0,
                              zero_initial_phi=False):
        #
        #
        #
        # monopole moment
        q1    = self.finegd.integrate(rho)/np.sqrt(4 * pi)
        q0    = self.finegd.integrate(rho0)/np.sqrt(4 * pi)
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


    def init_unitary_transformation(self,blocks=[],rattle=0.01):
        #
        test   =self.test
        nbands =self.nbands
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
            for u in blocks:
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
                        self.W_unn[q] = H_nn
                    else:
                        self.W_unn[q] = H_nn.real
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
            for u in blocks:
                #
                # get the local index of the block u
                q=self.myblocks.index(u)
                #
                G_nn = np.zeros((nbands,nbands),dtype=complex)
                w_n  = np.zeros((nbands),dtype=float)
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
                        self.W_unn[q] = H_nn
                    else:
                        self.W_unn[q] = H_nn.real
                else:
                    self.W_unn[q,:,:] = 0.0

                self.blk_comm.broadcast(self.W_unn[q], 0)
           
                if test>5:
                    print 'Initial transformation matrix W for block ',u
                    print self.W_unn[q]


def matrix_exponential(G_nn,U_nn,dlt):

    """Computes the matrix exponential of an antihermitian operator

        U = exp(dlt*G)

    """
    ndim = G_nn.shape[1]
    w_n  = np.zeros((ndim),dtype=float)
    
    V_nn = np.zeros((ndim,ndim),dtype=complex)
    if G_nn.dtype==complex:
        V_nn =  1j*G_nn.real - G_nn.imag
    else:
        V_nn =  1j*G_nn.real 

    diagonalize(V_nn,w_n)
    #
    for n in range(ndim):
        for m in range(ndim):
            U_nn[n,m] = 0.0
            for k in range(ndim):
                U_nn[n,m] = U_nn[n,m] +                             \
                            (cos(dlt*w_n[k]) + 1j*sin(dlt*w_n[k]))* \
                            V_nn[k,m]*V_nn[k,n].conj()


def matrix_multiply(A_nn,B_nn,C_nn):

    """Product of two sqare matrices

       one day in future this should be substituted by gemm

    """
    ndim = A_nn.shape[1]
            
    for n in range(ndim):
        for m in range(ndim):
            C_nn[n,m] = 0.0
            for k in range(ndim):
                C_nn[n,m] = C_nn[n,m] + A_nn[n,k]*B_nn[k,m]

def ortho(W):
    O = np.dot(W, W.T.conj())
    n = np.zeros(np.shape(W)[1])
    diagonalize(O,n)
    U = O.T.copy()
    nsqrt = np.diag(1/np.sqrt(n))
    X = np.dot(np.dot(U, nsqrt), U.T.conj())
    return np.dot(X, W)
