# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements time propagators for time-dependent density 
functional theory calculations."""

import Numeric as num
import BasicLinearAlgebra


# Propagator
class Propagator:
    """Propagator
    
    The Propagator-class is the VIRTUAL base class for all propagators.
    
    """
    def __init__(self, td_density, td_hamiltonian, td_overlap):
        """Create the Propagator-object.
        
        Propagator(td_density, td_hamiltonian, td_overlap)
        
        Parameters:
        =====================================================================
        td_density     = the time-dependent density
        td_hamiltonian = the time-dependent hamiltonian
        td_overlap     = the time-dependent overlap operator
        =====================================================================
        """ 
        raise('Error in Propagator: Propagator is virtual. ' \
                  'Use derived classes.' )
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        
    def propagate(self, kpt_up, kpt_dn, wf_up, wf_dn, time, time_step):
        """propagate(kpt_up, kpt_dn, wf_up, wf_dn, time, time_step)
        
        Propagate spin up and down wavefunctions. 
        
        Parameters:
        =====================================================================
        kpt_up    = k-point of spin up wavefunctions
        kpt_dn    = k-point of spin down wavefunctions
        wf_up     = list of spin up wavefunctions (kpt_u.psit_nG[])
        wf_dn     = list of spin down wavefunctions (kpt_d.psit_nG[])
        time      = the current time
        time_step = the time step
        =====================================================================
        """ 
        raise "Error in Propagator: Member function propagate is virtual."


# ExplicitCrankNicolson
class ExplicitCrankNicolson(Propagator):
    """Explicit Crank-Nicolson propagator
    
    Crank-Nicolson propagator, which approximates the time-dependent 
    Hamiltonian to be unchanged during one iteration step.
    
    ( S(t) + .5 dt H(t) / hbar ) psi(t+dt) =
       ( S(t) - .5 dt H(t) / hbar ) psi(t)
    
    """
    
    def __init__(self, td_density, td_hamiltonian, td_overlap, solver):
        """Create ExplicitCrankNicolson-object.
        
        ExplicitCrankNicolson(td_density, td_hamiltonian, td_overlap, solver)
        
        Parameters:
        =====================================================================
        td_density     = the time-dependent density
        td_hamiltonian = the time-dependent hamiltonian
        td_overlap     = the time-dependent overlap operator
        solver         = solver for linear equations
        =====================================================================
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.blas = BasicLinearAlgebra.BLAS()
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def propagate(self, kpt_up, kpt_dn, wf_up, wf_dn, time, time_step):
        """propagate(kpt_up, kpt_dn, wf_up, wf_dn, time, time_step)
        
        Propagate spin up and down wavefunctions. 
        
        Parameters:
        =====================================================================
        kpt_up    = k-point of spin up wavefunctions
        kpt_dn    = k-point of spin down wavefunctions
        wf_up     = list of spin up wavefunctions (kpt_u.psit_nG[])
        wf_dn     = list of spin down wavefunctions (kpt_d.psit_nG[])
        time      = the current time
        time_step = the time step
        =====================================================================
        """ 
        self.time_step = time_step
        self.td_density.update()
        self.td_hamiltonian.update(self.td_density.get_density(),time) 
        self.td_overlap.update()
        self.hpsit = num.zeros(wf_up[0].shape, num.Complex)
        self.spsit = num.zeros(wf_dn[0].shape, num.Complex)
                
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
        
        dot(psi, psin)
        
        Parameters:
        =====================================================================
        psi  = the known wavefunction
        psin = the result
        =====================================================================
        """
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit)
        self.td_overlap.apply(self.kpt, psi, self.spsit)
        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit 
        psin[:] = self.spsit
        self.blas.zaxpy(.5j * self.time_step, self.hpsit, psin)
