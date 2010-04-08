"""SIC stuff - work in progress!"""

import numpy as np
import ase.units as units
import gpaw.mpi as mpi

from ase import *
from gpaw.utilities import pack
from gpaw.utilities.blas import axpy, gemm
from gpaw.utilities.lapack import diagonalize
from gpaw.xc_functional import XCFunctional

from gpaw.atom.generator import Generator, parameters
from gpaw.utilities import hartree
from math import pi,cos,sin,log10

class SIC:
    def __init__(self, nspins=1, xcname='LDA',
                 coufac=1.0, excfac=1.0):
        """Self-Interaction Corrected (SIC) Functionals.

        nspins: int
            Number of spins.

        xcname: string
            Name of LDA/GGA functional which acts as
            a starting point for the construction of
            the SIC functional

        """

        self.nspins = nspins
        self.xcbasisname = xcname
        self.xcname      = xcname + '-SIC'
        #
        # SIC are always spin-polarized
        if nspins==1:
            self.xcbasis     = XCFunctional(self.xcbasisname, 1)
            self.xcsic       = XCFunctional(self.xcbasisname, 2)
            #self.xcsic       = XCFunctional('LDAx', 2)
        else:
            self.xcbasis     = XCFunctional(self.xcbasisname, 2)
            self.xcsic       = self.xcbasis
        #
        self.gga = self.xcbasis.gga
        self.mgga = not True
        self.orbital_dependent = True
        self.hybrid = 0.0
        self.uses_libxc = self.xcbasis.uses_libxc
        self.gllb = False
        #
        # weight factors for coulomb SIC and E_xc
        self.coufac    = coufac  # coulomb coupling constant
        self.excfac    = excfac  # scaling factor for exchange-correlation funct.
        self.optcmplx  = False   # complex optimization
        self.adderror  = False   # add unit-optimization residual to basis-residual
        
        self.dtype     = None    # complex/float 
        self.nbands    = None    # total number of bands/orbitals
        self.nblocks   = None    # total number of blocks
        self.mybands   = None    # list of bands of node's responsibility
        self.myblocks  = None    # list of blocks of node's responsibility
        
        self.inistep   = 1.0     # trial step length in unitary optimization
        self.nstep     = 4       # increment factor after succesful ls-step
                                 # fac=(1+1/n)
        self.mstep     = 2       # decrement factor after failed line-search step
                                 # fac=(1+1/n)^{-m}
        self.uomaxres  = 1E-4    # target accuracy for unitary optimization residual
        self.uorelres  = 1E-1     # same, but relative to basis residual
        self.maxuoiter = 10      # maximum number of unitary optimization iterations
        self.maxlsiter = 10      # maximum number of line-search steps
        
        self.units     = 27.21   # output units 1: in Hartree, 27.21: in eV
        #self.units     = 1       # output units 1: in Hartree, 27.21: in eV
        self.test      = 3       # debug level
        self.init_rattle =0.01  # perturbation to the canonic states
        self.ESI       = 0.0     # orbital dependent energy (SIC)
        self.RSI       = 0.0     # residual of unitary optimization
        
        self.init_SIC  = True    # SIC functional has to be initialized?
        self.init_cou  = True    # coulomb solver has to be initialized?
        self.act_SIC   = True    # self-consistent SIC
        self.virt_SIC  = False    # evaluate SIC for virtual orbitals
        self.parlayout = 1       # parallelization layout
        
        
    def set_non_local_things(self, density, hamiltonian, wfs, atoms,
                             energy_only=False):

        nbands    = wfs.nbands
        nkpt      = wfs.nibzkpts
        nspins    = hamiltonian.nspins
        nblocks   = nkpt*nspins
        mynblocks = len(wfs.kpt_u)
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
        
        self.nbands  =nbands
        self.nspins  =nspins
        self.nblocks =nblocks 
        
        if (self.optcmplx):
            self.dtype=complex
        else:
            self.dtype=float

        if mpi.rank==0:
            print 'SIC: complex optimization   :',self.optcmplx
            
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
        # list of nodes containing the real-space partitioning for the current k-point
        # (for now we do not mix k-points).
        nodes=density.gd.comm.get_members()
        self.blk_comm = mpi.world.new_communicator(nodes)
        #
        mynbands         = len(self.mybands)
        mynblocks        = len(self.myblocks)
        #
        self.mynbands    = mynbands      
        self.mynblocks   = mynblocks
        self.gd          = density.gd
        self.finegd      = density.finegd
        self.wfs         = wfs
        self.density     = density
        self.hamiltonian = hamiltonian
        self.atoms       = atoms
        #
        self.Sha         = 0.0
        self.Sxc         = 0.0
        self.Stot        = 0.0
        #
        # real-space representations of densities/WF and fields
        self.v_unG       = self.gd.empty((mynblocks,mynbands))
        self.v_cou_unG   = self.gd.empty((mynblocks,mynbands))
        self.phit_unG    = self.gd.empty((mynblocks,nbands))
        self.Htphit_unG  = self.gd.empty((mynblocks,nbands))
        #
        # utility fields
        self.nt_G        = self.gd.empty()
        self.nt_g        = self.finegd.empty()
        self.nt_g0       = self.finegd.empty()
        self.v_g0        = self.finegd.empty()
        self.e_g         = self.finegd.zeros()
        self.v_g         = self.finegd.zeros()
        self.v_cou_g     = self.finegd.zeros()
        #
        # occupation numbers and single-particle energies
        self.f_un        = np.zeros((nblocks,nbands),dtype=float)
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
        self.init_unitary_transformation(blocks=self.myblocks,rattle=self.init_rattle)
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
            #
            # only node 0 of grid communicator writes the total SIC to
            # the grid-point (0,0,0)
            if self.finegd.comm.rank == 0:
                    assert e_g.ndim == 3
                    e_g[0, 0, 0] += self.Stot / self.finegd.dv

        # calculate SIC matrix 
        self.calculate_sic_matrixelements()
        
        # unitary optimization
        self.unitary_optimization()

    def calculate_spinpolarized(self, e_g, na_g, va_g, nb_g, vb_g,
                                a2_g=None,  aa2_g=None, ab2_g=None,
                                deda2_g=None, dedaa2_g=None, dedab2_g=None):
        self.xcbasis.calculate_spinpolarized(e_g, na_g, va_g, nb_g, vb_g,
                                        a2_g, aa2_g, ab2_g,
                                        deda2_g, dedaa2_g, dedab2_g)
        if na_g.ndim == 3:
            #
            self.calculate_sic_potentials()
            #
            # only one single node writes the total SIC to
            # the grid-point (0,0,0)
            if self.finegd.comm.rank == 0:
                    assert e_g.ndim == 3
                    e_g[0, 0, 0] += self.Stot / self.finegd.dv
            
        # calculate SIC matrix
        self.calculate_sic_matrixelements()

        # unitary optimization
        self.unitary_optimization()

    def calculate_sic_potentials(self,blocks=[]):
        #
        # check if wavefunctions have already been initialized
        # -> else exit with E_SI=0
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
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
        self.write_energies(myblocks,mybands)
        #
        # return total orbital dependent energy
        #if 0 in myblocks and 0 in mybands:
        return self.Stot
        #else:
        #    return 0.0

    def calculate_sic_matrixelements(self,blocks=[]):        
        #
        # check if wavefunctions have already been initialized
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
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
            gemm(wfs.gd.dv,self.phit_unG[q],self.Htphit_unG[q],0.0,self.V_unn[q],'c')
            self.density.gd.comm.sum(self.V_unn[q])
            #
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
        # allocate temporary arrays
        U_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        O_nn  = np.zeros((self.nbands,self.nbands),dtype=self.W_unn.dtype)
        #
        # prepare logging
        if test>0 and mpi.rank==0:
            print ("================  unitary optimization  ===================")
        #
        # loop all blocks
        for u in myblocks:
            #
            # get the local index of the block u 
            q=self.myblocks.index(u)
            #
            # logging
            if test>3:
                print 'CPU ',mpi.rank,' optimizing block ',u,' (local index ',q,')'
            #
            # set initial step-width
            step=self.step_u[q]
            #
            # the unitary optimization iteration
            for iter in range(self.maxuoiter):
                #
                # create a copy of the initial unitary transformation
                # and orbital dependent energies
                W_old    = self.W_unn[q].copy()
                W_min    = self.W_unn[q].copy()
                #
                self.update_optimal_states([u],rotate_only=True)
                ESI_old  = self.update_potentials([u])
                #print ESI_old,u
                #
                ESI_min  = ESI_old        # minimum energy found along the line search
                step_min = self.step_u[q] # step length where the min. was found
                descent  = False          # have we achived an energy descent?
                iterV    = 0              # orb. dep. potentials last eval. in iter #
                iterE    = 0              # energies last evaluated in iter #
                K        = 0.0
                #
                # perform a line-search
                for lsiter in range(self.maxlsiter):
                    #
                    dN=0.0
                    #
                    # apply unitary transformation to matrix W
                    #print self.K_unn
                    matrix_exponential(self.K_unn[q],U_nn,step)
                    matrix_multiply(U_nn,W_old.copy(),self.W_unn[q])
                    #print U_nn
                    #
                    # check change in norm
                    matrix_multiply(self.W_unn[q],self.W_unn[q].T.conj(),O_nn)
                    O_nn=O_nn-np.eye(self.nbands)
                    dN=dN+sqrt(np.sum(O_nn*O_nn.conj()))
                    #
                    # compute the orbital dependent energy
                    self.update_optimal_states([u],rotate_only=True)
                    ESI   = self.update_potentials([u])
                    iterV = lsiter
                    #
                    #print u,q,ESI,ESI_min
                    if ESI<ESI_min:
                        #
                        # increase steplength
                        step    = step*(1.0 + 1.0/self.nstep)
                        step_min= step
                        ESI_min = ESI
                        iterE   = lsiter
                        W_min   = self.W_unn[q].copy()
                        descent  = True
                    else:
                        #
                        if not descent:
                            #
                            # decrease steplength
                            step=step*pow(1.0 + 1.0/self.nstep,-self.mstep)
                            descent=False
                        else:
                            #
                            # we already found a minimum, so we can exit...
                            break
                    #
                    if test>3:
                        if self.finegd.comm.rank == 0:
                            print 'lsiter :',lsiter,ESI,step,descent
                    #
                #
                # bad luck: line search did not find any lower energy for
                # the tested step-sizes.
                # --> just revert to the initial unitary transformation, exit
                # unitary optimization and hope that the changes due to the
                # optimization of the basis will cure the problem within the
                # next iteration cycle. Still if this is not the case the
                # step-size will be reduced within each unsuccessful call
                # which should finally yield an energy gain in the limit step->0
                if not descent:
                    self.W_unn[q] = W_old.copy()
                    ESI=ESI_old
                    break
                #
                # select the optimal step-size
                step          = step_min
                ESI           = ESI_min
                self.W_unn[q] = W_min.copy()
                #
                # if the potentials belong to a different step-length -> update
                if iterV != iterE:
                    self.update_optimal_states([u],rotate_only=True)
                    ESI   = self.update_potentials([u])
                #
                # update the matrixelements of V and Kappa
                # and accumulate total residual of unitary optimization
                self.calculate_sic_matrixelements([u])
                self.RSI_u[q]=self.normK_u[q]
                #
                # logging
                if test>1:
                    dE=max(abs(ESI-ESI_old),1.0E-16)
                    dN=max(abs(dN),1.0E-16)
                    K =sqrt(max(self.RSI_u[q],1.0E-16))
                    if self.finegd.comm.rank == 0:
                        print(" UO-iter %3i : %10.5f  %5.1f %5.1f %5.1f %12.4f %3i" %
                              (iter+1,ESI*self.units,log10(dE),log10(K),
                               log10(abs(dN)),step,lsiter))
            
                if K<basiserror*self.uorelres or K<self.uomaxres:
                    localerror = localerror + K**2
                    break
            #
            # save step for next run
            self.step_u[q]  = step
            
        if test>0 and mpi.rank==0:
            print ("============  finished unitary optimization  ==============")
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
        q=self.myblocks.index(u)
        #
        nbands=psit_nG.shape[0]
        U_nn = np.zeros((nbands,self.nbands),dtype=self.dtype)
        V_nn = np.zeros((nbands,self.nbands),dtype=psit_nG.dtype)
        #
        # project psit_nG to the energy optimal states
        gemm(self.wfs.gd.dv,self.phit_unG[q],psit_nG,0.0,U_nn,'c')
        self.density.gd.comm.sum(U_nn)
        gemm(1.0,self.V_unn[q],U_nn,0.0,V_nn)
        #
        # accumulate action of the orbital dependent operator
        gemm(1.0,self.phit_unG[q],V_nn,1.0,Htpsit_nG)


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
        Sha_un  = self.Sha_un
        Sxc_un  = self.Sxc_un
        Stot_un = self.Stot_un
        Sha     = self.Sha
        Sxc     = self.Sxc
        Stot    = self.Stot
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
                    print ("%3d %3d %8.3f %5.2f : %10.5f  %10.5f %10.5f" %
                          (u,n,eps_un[u,n]*units,f_un[u,n],
                           Sha_un[u,n]*units,Sxc_un[u,n]*units,
                           Stot_un[u,n]*units))
            #
            # write the footer
            print ("-----------------------------------------------------------")
            print ("          total        : %10.5f  %10.5f %10.5f" %
                   (Sha*units,Sxc*units,Stot*units))
            print ("===========================================================")


    def update_potentials(self,blocks=[],bands=[]):
        #
        #
        # check if wavefunctions have already been initialized
        # -> else exit with E_SI=0
        if self.wfs.kpt_u[0].psit_nG is None:
            return 0.0
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
        #
        #
        # select blocks which need to be evaluated
        if blocks==[]:
            myblocks    = self.myblocks
        else:
            myblocks    = self.myblocks and blocks
        #
        mybands     = self.mybands
        #
        Sha_un[:,:] = 0.0
        Sxc_un[:,:] = 0.0
        Stot_un[:,:] = 0.0
        nocc_u[:]    = 0.0
        #
        # loop specified blocks and bands
        for u in myblocks: 
            #
            # get the local index of the block u (which is
            q=self.myblocks.index(u)
            #
            #print 'updating potentials',q
            #
            for n in mybands:
                #
                # initialize temporary variables
                # ------------------------------------------------------------
                f        = self.f_un[u,n]
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
                nt_G     = np.abs(self.phit_unG[q,n])**2
                #
                # interpolate density on the fine grid
                # ------------------------------------------------------------
                # notes: the total norm has to be conserved explicitly by
                #        renormalization of the interpolated density
                #        to avoid errors due to the interpolation. One
                #        furthermore also expects to have a positive semi-
                #        definit propability of finding a particle (density).
                #        But interpolation sucks and does not necessarily
                #        maintain such fundamental properties :-(
                #        However, this must be corrected somewhere else, as
                #        else the xc-potential would contain NaNs
                #        (already for LDA exchange only)
                #
                Nt = density.gd.integrate(nt_G)
                density.interpolator.apply(nt_G, nt_g)
                #np.maximum(nt_g,0.0)
                Ntfine = density.finegd.integrate(nt_g)
                nt_g  *= Nt / Ntfine
                #
                # self-interaction correction for E_xc
                # ------------------------------------------------------------
                if self.excfac==0.0 or noSIC:
                    Sxc_un[u,n] = 0.0
                else:
                    self.xcsic.calculate_spinpolarized(e_g, nt_g, v_g, nt_g0, v_g0)
                    #Sxc_un[u,n] = -self.excfac*e_g.ravel().sum() * self.finegd.dv
                    Sxc_un[u,n] = -self.excfac*density.finegd.integrate(e_g)
                    v_g[:]     *= -f*self.excfac
                #
                # self-interaction correction for U_Hartree
                # ------------------------------------------------------------
                if self.coufac==0.0 or noSIC:
                    Sha_un[u,n] = 0.0
                else:
                    #
                    # use the last coulomb potential as initial guess
                    # for the coulomb solver and transform to the fine grid
                    density.interpolator.apply(v_cou_unG[q,n], v_cou_g)
                    #
                    # initialize the coulomb solver (if necessary) and
                    # solve the poisson equation
                    psolver.solve(v_cou_g, nt_g, charge=1, 
                                  zero_initial_phi=self.init_cou)
                    #
                    # compose the energy density and add potential
                    # contributions to orbital dependent potential.
                    e_g         = nt_g * v_cou_g
                    Sha_un[u,n] = -0.5*self.coufac*density.finegd.integrate(e_g)
                    v_g[:]      = v_g[:] - self.coufac * f * v_cou_g[:]
                #
                # restrict to the coarse-grid
                # -------------------------------------------------------------
                hamiltonian.restrictor.apply(v_g    , v_unG[q,n])
                hamiltonian.restrictor.apply(v_cou_g, v_cou_unG[q,n])
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
        # return the total orbital dependent energy for the
        # updated states
        #print Stot_un
        return (Stot_un*self.f_un).sum()
        
        
    def update_optimal_states(self,blocks=[],rotate_only=False):
        #
        test    = self.test
        nbands  = self.nbands
        nblocks = self.nblocks
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
            # check overlap matrix of orbitals |Phi>
            if test>5:
                gemm(self.wfs.gd.dv,self.phit_unG[q],self.phit_unG[q],0.0,self.O_unn[q],'c')
                self.density.gd.comm.sum(self.O_unn[q])
                print 'Overlap matrix <Phi_i|Phi_j> for block ',q
                print self.O_unn[q]
                
        if rotate_only:
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

