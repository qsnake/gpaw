# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements time propagators for time-dependent density 
functional theory calculations."""

import Numeric as num
import BasicLinearAlgebra


###############################################################################
# Propagator
###############################################################################
class Propagator:
    """Time propagator
    
    The Propagator-class is the VIRTUAL base class for all propagators.
    
    """
    def __init__(self, td_density, td_hamiltonian, td_overlap):
        """Create the Propagator-object.
        
        ================ ====================================================
        Parameters:
        ================ ====================================================
        td_density       the time-dependent density
        td_hamiltonian   the time-dependent hamiltonian
        td_overlap       the time-dependent overlap operator
        ================ ====================================================
        
        """ 
        raise('Error in Propagator: Propagator is virtual. ' \
                  'Use derived classes.' )
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        
    def propagate(self, kpt_up, kpt_dn, wf_up, wf_dn, time, time_step):
        """Propagate spin up and down wavefunctions. 
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        kpt_up      k-point of spin up wavefunctions
        kpt_dn      k-point of spin down wavefunctions
        wf_up       list of spin up wavefunctions (kpt_u.psit_nG[])
        wf_dn       list of spin down wavefunctions (kpt_d.psit_nG[])
        time        the current time
        time_step   the time step
        =========== ==========================================================
        
        """ 
        raise "Error in Propagator: Member function propagate is virtual."



###############################################################################
# ExplicitCrankNicolson
###############################################################################
class ExplicitCrankNicolson(Propagator):
    """Explicit Crank-Nicolson propagator
    
    Crank-Nicolson propagator, which approximates the time-dependent 
    Hamiltonian to be unchanged during one iteration step.
    
    (S(t) + .5 dt H(t) / hbar) psi(t+dt) = (S(t) - .5 dt H(t) / hbar) psi(t)
    
    """
    
    def __init__(self, td_density, td_hamiltonian, td_overlap, solver):
        """Create ExplicitCrankNicolson-object.
        
        ================ =====================================================
        Parameters:
        ================ =====================================================
        td_density       the time-dependent density
        td_hamiltonian   the time-dependent hamiltonian
        td_overlap       the time-dependent overlap operator
        solver           solver for linear equations
        ================ =====================================================
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = BasicLinearAlgebra.BLAS()
        
        self.hpsit = None
        self.spsit = None
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def propagate(self, kpt_up, kpt_dn, wf_up, wf_dn, time, time_step):
        """Propagate spin up and down wavefunctions. 
        
        =========== =========================================================
        Parameters:
        =========== =========================================================
        kpt_up      k-point of spin up wavefunctions
        kpt_dn      k-point of spin down wavefunctions
        wf_up       list of spin up wavefunctions (kpt_u.psit_nG[])
        wf_dn       list of spin down wavefunctions (kpt_d.psit_nG[])
        time        the current time
        time_step   the time step
        =========== =========================================================
        
        """
        self.time_step = time_step
        self.td_density.update()
        self.td_hamiltonian.update(self.td_density.get_density(),time)
        self.td_overlap.update()
        if ( self.hpsit == None ):
            self.hpsit = num.zeros(wf_up[0].shape, num.Complex)
        if ( self.spsit == None ):
            self.spsit = num.zeros(wf_up[0].shape, num.Complex)
        
        for psit in wf_up:
            self.kpt = kpt_up
            self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
            self.td_overlap.apply(self.kpt, psit, self.spsit)
            
            #psit[:] = self.spsit - .5J * self.hpsit * time_step
            psit[:] = self.spsit
            self.blas.zaxpy(-.5j * self.time_step, self.hpsit, psit)
            
            # A x = b
            psit[:] = self.solver.solve(self,psit,psit)
            
        for psit in wf_dn:
            self.kpt = kpt_dn
            self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
            self.td_overlap.apply(self.kpt, psit, self.spsit)
            # psit[:] = self.spsit - .5J * self.hpsit * time_step
            psit[:] = self.spsit
            self.blas.zaxpy(-.5j * self.time_step, self.hpsit, psit)
            
            # A x = b
            psit[:] = self.solver.solve(self,psit,psit)
        
    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunction.
        
        =========== ===================================
        Parameters:
        =========== ===================================
        psi         the known wavefunction
        psin        the result
        =========== ===================================
        
        """
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit)
        self.td_overlap.apply(self.kpt, psi, self.spsit)
        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit 
        psin[:] = self.spsit
        self.blas.zaxpy(.5j * self.time_step, self.hpsit, psin)



###############################################################################
# SemiImpicitCrankNicolson
###############################################################################
class SemiImplicitCrankNicolson(Propagator):
    """Semi-implicit Crank-Nicolson propagator
    
    Crank-Nicolson propagator, which first approximates the time-dependent 
    Hamiltonian to be unchanged during one iteration step to predict future 
    wavefunctions. Then the approximations for the future wavefunctions are
    used to approximate the Hamiltonian at the middle of the time step.
    
    (S(t) + .5 dt H(t) / hbar) psi_t(t+dt) = (S(t) - .5 dt H(t) / hbar) psi(t)
    (S(t) + .5 dt H(t+dt/2) / hbar) psi(t+dt) 
    = (S(t) - .5 dt H(t+dt/2) / hbar) psi(t)
    
    """
    
    def __init__(self, td_density, td_hamiltonian, td_overlap, solver):
        """Create SemiImplicitCrankNicolson-object.
        
        ================ =====================================================
        Parameters:
        ================ =====================================================
        td_density       the time-dependent density
        td_hamiltonian   the time-dependent hamiltonian
        td_overlap       the time-dependent overlap operator
        solver           a solver for linear equations
        ================ =====================================================
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = BasicLinearAlgebra.BLAS()
        
        self.twf_up = None
        self.twf_dn = None
        
        self.hpsit = None
        self.spsit = None
        
        
    def propagate(self, kpt_up, kpt_dn, wf_up, wf_dn, time, time_step):
        """Propagate spin up and down wavefunctions.
        
        =========== =========================================================
        Parameters:
        =========== =========================================================
        kpt_up      k-point of spin up wavefunctions
        kpt_dn      k-point of spin down wavefunctions
        wf_up       list of spin up wavefunctions (kpt_u.psit_nG[])
        wf_dn       list of spin down wavefunctions (kpt_d.psit_nG[])
        time        the current time
        time_step   the time step
        =========== =========================================================
        
        """
        # temporary wavefunctions
        if ( self.twf_up == None ):
            self.twf_up = num.array(wf_up, num.Complex)
        else:
            self.twf_up[:] = wf_up
        if ( self.twf_dn == None ):
            self.twf_dn = num.array(wf_dn, num.Complex)
        else:
            self.twf_dn[:] = wf_dn
        
        if ( self.hpsit == None ):
            self.hpsit = num.zeros(wf_up[0].shape, num.Complex)
        if ( self.spsit == None ):
            self.spsit = num.zeros(wf_up[0].shape, num.Complex)
        
        self.time_step = time_step
        self.kpt_up = kpt_up
        self.kpt_dn = kpt_dn
        
        # rho(t)
        self.td_density.update()
        # H(t)
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        # S(t)
        self.td_overlap.update()
        
        # predict
        self.solve_propagation_equation(wf_up, wf_dn, time_step)
        
        # rho(t+dt)
        self.td_density.update()
        # H(t+dt/2)
        self.td_hamiltonian.half_update( self.td_density.get_density(), \
                                         time + time_step )
        # S(t+dt/2)
        self.td_overlap.half_update()
        
        wf_up[:] = self.twf_up[:]
        wf_dn[:] = self.twf_dn[:]
        
        # correct
        self.solve_propagation_equation(wf_up, wf_dn, time_step)
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def solve_propagation_equation(self, wf_up, wf_dn, time_step):
        
        for psit in wf_up:
            self.kpt = self.kpt_up
            self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
            self.td_overlap.apply(self.kpt, psit, self.spsit)

            #psit[:] = self.spsit - .5J * self.hpsit * time_step
            psit[:] = self.spsit
            self.blas.zaxpy(-.5j * self.time_step, self.hpsit, psit)
            
            # A x = b
            psit[:] = self.solver.solve(self,psit,psit)
            
        for psit in wf_dn:
            self.kpt = self.kpt_dn
            self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
            self.td_overlap.apply(self.kpt, psit, self.spsit)
            # psit[:] = self.spsit - .5J * self.hpsit * time_step
            psit[:] = self.spsit
            self.blas.zaxpy(-.5j * self.time_step, self.hpsit, psit)
            
            # A x = b
            psit[:] = self.solver.solve(self,psit,psit)
            
            
    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunction.
        
        =========== ===================================
        Parameters:
        =========== ===================================
        psi         the known wavefunction
        psin        the result
        =========== ===================================
        
        """
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit)
        self.td_overlap.apply(self.kpt, psi, self.spsit)
        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit
        psin[:] = self.spsit
        self.blas.zaxpy(.5j * self.time_step, self.hpsit, psin)



###############################################################################
# SelfConsistentCrankNicolson
###############################################################################
class SelfConsistentCrankNicolson(Propagator):
    """Self-consistent Crank-Nicolson propagator
    
    (S(t) + .5 dt H(t+dt/2) / hbar) psi(t+dt) 
          = (S(t) - .5 dt H(t+dt/2) / hbar) psi(t)
    
    """
    
    def __init__(self, td_density, td_hamiltonian, td_overlap, solver):
        """Create SelfConsistentCrankNicolson-object.
        
        ================ =====================================================
        Parameters:
        ================ =====================================================
        td_density       the time-dependent density
        td_hamiltonian   the time-dependent hamiltonian
        td_overlap       the time-dependent overlap operator
        solver           a solver for linear equations
        ================ =====================================================
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = BasicLinearAlgebra.BLAS()
        
        self.twf_up = None
        self.twf_dn = None
        
        self.hpsit = None
        self.spsit = None
        
        self.vt_sG = None
        
        
    def propagate(self, kpt_up, kpt_dn, wf_up, wf_dn, time, time_step, scf_iterations = 4, debug = 0):
        """Propagate spin up and down wavefunctions.
        
        =========== =========================================================
        Parameters:
        =========== =========================================================
        kpt_up      k-point of spin up wavefunctions
        kpt_dn      k-point of spin down wavefunctions
        wf_up       list of spin up wavefunctions (kpt_u.psit_nG[])
        wf_dn       list of spin down wavefunctions (kpt_d.psit_nG[])
        time        the current time
        time_step   the time step
        =========== =========================================================
        
        """
        # temporary wavefunctions
        if ( self.twf_up == None ):
            self.twf_up = num.array(wf_up, num.Complex)
        else:
            self.twf_up[:] = wf_up
        if ( self.twf_dn == None ):
            self.twf_dn = num.array(wf_dn, num.Complex)
        else:
            self.twf_dn[:] = wf_dn
        
        if ( self.hpsit == None ):
            self.hpsit = num.zeros(wf_up[0].shape, num.Complex)
        if ( self.spsit == None ):
            self.spsit = num.zeros(wf_up[0].shape, num.Complex)
            
        if ( self.vt_sG == None ):
            self.vt_sG = \
                num.zeros( self.td_hamiltonian.hamiltonian.vt_sG.shape,
                           num.Float )
        
        self.time_step = time_step
        self.kpt_up = kpt_up
        self.kpt_dn = kpt_dn
        
        # rho(t)
        self.td_density.update()
        # H(t)
        self.td_hamiltonian.update( self.td_density.get_density(), \
                                    time + time_step / 2 )
        # S(t)
        self.td_overlap.update()
        
        # predict
        self.solve_propagation_equation(wf_up, wf_dn, time_step)
        
        self.vt_sG[:] = self.td_hamiltonian.hamiltonian.vt_sG
        
        for i in range(scf_iterations):
            self.td_hamiltonian.hamiltonian.vt_sG[:] = self.vt_sG
            
            # rho(t+dt)
            self.td_density.update()
            # H(t+dt/2)
            self.td_hamiltonian.half_update( self.td_density.get_density(), \
                                             time + time_step / 2 )
            # S(t+dt/2)
            self.td_overlap.half_update()
            
            if ( debug ):
                print 'H', num.vdot( (self.td_hamiltonian.hamiltonian.vt_sG - self.vt_sG), (self.td_hamiltonian.hamiltonian.vt_sG - self.vt_sG) )
            
            wf_up[:] = self.twf_up[:]
            wf_dn[:] = self.twf_dn[:]
        
            # correct
            self.solve_propagation_equation(wf_up, wf_dn, time_step)
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def solve_propagation_equation(self, wf_up, wf_dn, time_step):
        
        for psit in wf_up:
            self.kpt = self.kpt_up
            self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
            self.td_overlap.apply(self.kpt, psit, self.spsit)

            #psit[:] = self.spsit - .5J * self.hpsit * time_step
            psit[:] = self.spsit
            self.blas.zaxpy(-.5j * self.time_step, self.hpsit, psit)
            
            # A x = b
            psit[:] = self.solver.solve(self,psit,psit)
            
        for psit in wf_dn:
            self.kpt = self.kpt_dn
            self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
            self.td_overlap.apply(self.kpt, psit, self.spsit)
            # psit[:] = self.spsit - .5J * self.hpsit * time_step
            psit[:] = self.spsit
            self.blas.zaxpy(-.5j * self.time_step, self.hpsit, psit)
            
            # A x = b
            psit[:] = self.solver.solve(self,psit,psit)
            
            
    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunction.
        
        =========== ===================================
        Parameters:
        =========== ===================================
        psi         the known wavefunction
        psin        the result
        =========== ===================================
        
        """
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit)
        self.td_overlap.apply(self.kpt, psi, self.spsit)
        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit
        psin[:] = self.spsit
        self.blas.zaxpy(.5j * self.time_step, self.hpsit, psin)
