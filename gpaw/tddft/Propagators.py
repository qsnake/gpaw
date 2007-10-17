# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements time propagators for time-dependent density 
functional theory calculations."""

import Numeric as num
import gpaw.tddft.BasicLinearAlgebra

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
        raise RuntimeError( 'Error in Propagator: Propagator is virtual. '
                            'Use derived classes.' )
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        
    def propagate(self, kpt_u, time, time_step):
        """Propagate spin up and down wavefunctions. 
        
        =========== ==========================================================
        Parameters:
        =========== ==========================================================
        kpt_u       K-points (= spins, contains wavefunctions)
        time        the current time
        time_step   the time step
        =========== ==========================================================
        
        """ 
        raise RuntimeError( 'Error in Propagator: '
                            'Member function propagate is virtual.' )


###############################################################################
# ExplicitCrankNicolson
###############################################################################
class ExplicitCrankNicolson(Propagator):
    """Explicit Crank-Nicolson propagator
    
    Crank-Nicolson propagator, which approximates the time-dependent 
    Hamiltonian to be unchanged during one iteration step.
    
    (S(t) + .5 dt H(t) / hbar) psi(t+dt) = (S(t) - .5 dt H(t) / hbar) psi(t)
    
    """
    
    def __init__(self, td_density, td_hamiltonian, td_overlap, solver, blas):
        """Create ExplicitCrankNicolson-object.
        
        ================ =====================================================
        Parameters:
        ================ =====================================================
        td_density       the time-dependent density
        td_hamiltonian   the time-dependent hamiltonian
        td_overlap       the time-dependent overlap operator
        solver           solver for linear equations
        blas             basic linear algebra subroutines
        ================ =====================================================
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = blas
        
        self.hpsit = None
        self.spsit = None
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def propagate(self, kpt_u, time, time_step):
        """Propagate spin up and down wavefunctions. 
        
        =========== =========================================================
        Parameters:
        =========== =========================================================
        kpt_u       K-points (= spins, contains wavefunctions)
        time        the current time
        time_step   the time step
        =========== =========================================================
        
        """
        self.time_step = time_step
        self.td_density.update()
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        self.td_overlap.update()
        if self.hpsit is None:
            self.hpsit = self.blas.zeros(kpt_u[0].psit_nG[0].shape, num.Complex)
        if self.spsit is None:
            self.spsit = self.blas.zeros(kpt_u[0].psit_nG[0].shape, num.Complex)
        
        # loop over k-points (spins)
        for kpt in kpt_u:
            self.kpt = kpt
            # loop over wavefunctions of this k-point (spin)
            for psit in kpt.psit_nG:
                self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
                self.td_overlap.apply(self.kpt, psit, self.spsit)
            
                #psit[:] = self.spsit - .5J * self.hpsit * time_step
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
# DummyDensity
###############################################################################
class DummyDensity:
    """Implements dummy (= does nothing) density for AbsorptionKick."""
    def update(self):
        pass
        
    def get_density(self):
        return None

###############################################################################
# AbsorptionKick
###############################################################################
class AbsorptionKick(ExplicitCrankNicolson):
    """Absorption kick propagator
    
    Absorption kick propagator::

      (S(t) + .5 dt p.r / hbar) psi(0+) = (S(t) - .5 dt p.r / hbar) psi(0-)

    where ``|p| = (eps e / hbar)``, and eps is field strength, e is elementary 
    charge.
    
    """
    
    def __init__(self, abs_kick_hamiltonian, td_overlap, solver, blas):
        """Create AbsorptionKick-object.
        
        ===================== =================================================
        Parameters:
        ===================== =================================================
        abs_kick_hamiltonian  the absorption kick hamiltonian
        td_overlap            the time-dependent overlap operator
        solver                solver for linear equations
        blas                  basic linear algebra subroutines
        ===================== =================================================
        
        """
        self.td_density = DummyDensity()
        self.td_hamiltonian = abs_kick_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = blas
        
        self.hpsit = None
        self.spsit = None


    def kick(self, kpt_u):
        """Excite all possible frequencies.""" 
        #print "Absorption kick iterations = ", self.td_hamiltonian.iterations
        #print " (. = 10 iterations)"
        for l in range(self.td_hamiltonian.iterations):
            self.propagate(kpt_u, 0, 1.0)
        #    if ( ((l+1) % 10) == 0 ): 
        #        print ".",
        #        sys.stdout.flush()
        #print ""
        

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
    
    def __init__(self, td_density, td_hamiltonian, td_overlap, solver, blas):
        """Create SemiImplicitCrankNicolson-object.
        
        ================ =====================================================
        Parameters:
        ================ =====================================================
        td_density       the time-dependent density
        td_hamiltonian   the time-dependent hamiltonian
        td_overlap       the time-dependent overlap operator
        solver           a solver for linear equations
        blas             basic linear algebra subroutines
        ================ =====================================================
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = blas
        
        self.twf = None        
        self.hpsit = None
        self.spsit = None
        
        
    def propagate(self, kpt_u, time, time_step):
        """Propagate spin up and down wavefunctions.
        
        =========== =========================================================
        Parameters:
        =========== =========================================================
        kpt_u       K-points (= spins, contains wavefunctions)
        time        the current time
        time_step   the time step
        =========== =========================================================
        
        """
        # temporary wavefunctions
        if self.twf is None:
            self.twf = []
            for kpt in kpt_u:
                twf = [
                    self.blas.array(psit, num.Complex)
                    for psit in kpt.psit_nG
                    ]
                self.twf.append(twf)
        else:
            for u in range(len(kpt_u)):
                self.twf[u][:] = kpt_u[u].psit_nG
        
        if self.hpsit is None:
            self.hpsit = self.blas.zeros(kpt_u[0].psit_nG[0].shape, num.Complex)
        if self.spsit is None:
            self.spsit = self.blas.zeros(kpt_u[0].psit_nG[0].shape, num.Complex)
        
        self.time_step = time_step
        
        # rho(t)
        self.td_density.update()
        # H(t)
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        # S(t)
        self.td_overlap.update()
        
        # predict
        self.solve_propagation_equation(kpt_u, time_step)
        
        # rho(t+dt)
        self.td_density.update()
        # H(t+dt/2)
        self.td_hamiltonian.half_update( self.td_density.get_density(),
                                         time + time_step )
        # S(t+dt/2)
        self.td_overlap.half_update()

        # propagate psit(t), not psit(t+dt), in correct
        for u in range(len(kpt_u)):
            kpt_u[u].psit_nG[:] = self.twf[u]
        
        # correct
        self.solve_propagation_equation(kpt_u, time_step)
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def solve_propagation_equation(self, kpt_u, time_step):
        
        for kpt in kpt_u:
            self.kpt = kpt
            for psit in kpt.psit_nG:
                self.td_hamiltonian.apply(self.kpt, psit, self.hpsit)
                self.td_overlap.apply(self.kpt, psit, self.spsit)

                #psit[:] = self.spsit - .5J * self.hpsit * time_step
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



