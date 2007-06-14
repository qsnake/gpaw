# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements a class for (true) time-dependent density 
functional theory calculations."""

import Numeric as num

import BiCGStab
import Propagators
from TimeDependentVariablesAndOperators import \
    TimeDependentHamiltonian, TimeDependentOverlap, \
    TimeDependentDensity


class TDDFT:
    """ Time-dependent density functional theory
    
    This class is the core class of the time-dependent density functional 
    theory implementation and is the only class which user has to utilize.
    """
    
    def __init__(self, paw, td_potential = None, kpt_up = None, kpt_dn = None,\
                     propagator='EXCN', solver='BiCGStab', tolerance=1e-10):
        """Create TDDFT-object.
        
        TDDFT(paw, td_potential = None, propagator='EXCN', 
              solver='BiCGStab, tolerance=1e-10)
              
        Parameters:
        =====================================================================
        paw          = the PAW-object from a time-independent (the ground state)
                       calculation
        td_potential = the time-dependent potential
        kpt_up       = spin up k-point   (if None, paw.kpt_u[0])
        kpt_dn       = spin down k-point (if None, paw.kpt_u[1])
        propagator   = the name of the time propagator
        solver       = the name of the iterative linear equations solver 
        tolerance    = tolerance for the solver
        =====================================================================
        
        Note: Use about ???10^-3 - 10^-4??? tighter tolerance for PAW.
        """
        
        # Convert PAW-object to complex
        paw.totype(num.Complex);
        
        # Set initial time 
        self.time = 0.
        
        # Time-dependent variables and operators
        self.td_potential = td_potential
        self.td_hamiltonian = \
            TimeDependentHamiltonian( paw.pt_nuclei, \
                                          paw.hamiltonian, \
                                          td_potential )
        self.td_overlap = TimeDependentOverlap(paw.pt_nuclei)
        self.td_density = TimeDependentDensity(paw)
        
        # Solver for linear equations
        if (solver == 'BiCGStab'):
            self.solver = BiCGStab.BiCGStab(tolerance=tolerance)
        else:
            raise('Error in TDDFT: Solver %s not supported. '\
                      'Only BiCGStab is currently supported.' % (solver) )
        
        # Time propagator
        if ( propagator == 'EXCN' ):
            self.propagator = Propagators.ExplicitCrankNicolson(self.td_density,self.td_hamiltonian,self.td_overlap,self.solver)
        else:
            raise('Error in TDDFT: Time propagator %s not supported. '\
                      'Only EXCN (Explicit Crank-Nicolson) is currently'\
                      'supported.' % (propagator) )
        
        # Wavefunctions and k-points
        if ( kpt_up != None ):
            self.kpt_up = kpt_up
        else:
            self.kpt_up = paw.kpt_u[0]
            
        if ( kpt_dn != None ):
            self.kpt_dn = kpt_dn
        else:
            self.kpt_dn = paw.kpt_u[1]
            
        self.wf_up = self.kpt_up.psit_nG
        self.wf_dn = self.kpt_dn.psit_nG
        
        
    def propagate(self, time_step, iterations=1):
        """Propagates wavefunctions.
        
        propagate(time_step, iterations=1)
        
        Parameters:
        =====================================================================
        time_step = time step
        iterations = iterations
        =====================================================================
        """
        for i in range(iterations):
            self.propagator.propagate(self.kpt_up,self.kpt_dn, self.wf_up,self.wf_dn, self.time, time_step)
            self.time += time_step
