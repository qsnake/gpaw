# Copyright (c) 2007 Lauri Lehtovaara

"""This module implements time propagators for time-dependent density 
functional theory calculations."""

import sys

import numpy as npy

from gpaw.utilities.blas import axpy

from gpaw.mpi import rank

# Multivector ZAXPY: a x + y => y
def multi_zaxpy2(a,x,y, nvec):
    for i in range(nvec):
        axpy(a*(1+0J), x[i], y[i])


###############################################################################
# Propagator
###############################################################################
class Propagator:
    """Time propagator
    
    The Propagator-class is the VIRTUAL base class for all propagators.
    
    """
    def __init__(self, td_density, td_hamiltonian, td_overlap):
        """Create the Propagator-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            the time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            the time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        
        """ 
        raise RuntimeError( 'Error in Propagator: Propagator is virtual. '
                            'Use derived classes.' )
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap

        
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions once. 
        
        Parameters
        ----------
        kpt_u: Kpoint
            K-point
        time: float
            the current time
        time_step: float
            the time step
        
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
    
    def __init__( self, td_density, td_hamiltonian, td_overlap, 
                  solver, preconditioner, gd, timer):
        """Create ExplicitCrankNicolson-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (/wavefunction) grid descriptor
        timer: Timer
            timer
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.preconditioner = preconditioner
        self.gd = gd
        self.timer = timer
        
        self.hpsit = None
        self.spsit = None
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions. 
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points
        time: float
            the current time
        time_step: float
            time step
        
        """
        self.timer.start('Update time-dependent operators')

        self.time_step = time_step
        self.td_density.update()
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        self.td_overlap.update()

        if self.hpsit is None:
            self.hpsit = self.gd.zeros(n=len(kpt_u[0].psit_nG), dtype=complex)
        if self.spsit is None:
            self.spsit = self.gd.zeros(n=len(kpt_u[0].psit_nG), dtype=complex)

        self.timer.stop('Update time-dependent operators')
        
        # loop over k-points (spins)
        for kpt in kpt_u:
            self.kpt = kpt
            self.timer.start('Apply time-dependent operators')
            self.td_hamiltonian.apply(self.kpt, kpt.psit_nG, self.hpsit)
            self.td_overlap.apply(self.kpt, kpt.psit_nG, self.spsit)
            self.timer.stop('Apply time-dependent operators')

            #psit[:] = self.spsit - .5J * self.hpsit * time_step
            kpt.psit_nG[:] = self.spsit            
            multi_zaxpy2(-.5j * self.time_step, self.hpsit, kpt.psit_nG, 
                          len(kpt.psit_nG))

            # A x = b
            self.solver.solve(self, kpt.psit_nG, kpt.psit_nG)


    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunction.

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result

        """
        self.timer.start('Apply time-dependent operators')
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit)
        self.td_overlap.apply(self.kpt, psi, self.spsit)
        self.timer.stop('Apply time-dependent operators')

        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit 
        psin[:] = self.spsit
        multi_zaxpy2(.5j * self.time_step, self.hpsit, psin,
                      len(psi))


    #  M psin = psi, where M = T (kinetic energy operator)
    def apply_preconditioner(self, psi, psin):
        """Solves preconditioner equation.

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result
        
        """
        self.timer.start('Apply TDDFT preconditioner')
        if self.preconditioner is not None:
            self.preconditioner.apply(self.kpt, psi, psin)
        else:
            psin[:] = psi
        self.timer.stop('Apply TDDFT preconditioner')


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
    
    def __init__(self, abs_kick_hamiltonian, td_overlap, solver, preconditioner, gd, timer):
        """Create AbsorptionKick-object.
        
        Parameters
        ----------
        abs_kick_hamiltonian: AbsorptionKickHamiltonian
            the absorption kick hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (wavefunction) grid descriptor
        timer: Timer
            timer

        """
        self.td_density = DummyDensity()
        self.td_hamiltonian = abs_kick_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.preconditioner = preconditioner
        self.gd = gd
        self.timer = timer

        self.hpsit = None
        self.spsit = None


    def kick(self, kpt_u):
        """Excite all possible frequencies.
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points

        """ 

        if rank == 0:
            print "Kick iterations = ", self.td_hamiltonian.iterations

        for l in range(self.td_hamiltonian.iterations):
            self.propagate(kpt_u, 0, 1.0)
            if rank == 0:
                print '.',
                sys.stdout.flush()
        if rank == 0:
            print ''
        

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
    
    def __init__( self, td_density, td_hamiltonian, td_overlap, 
                  solver, preconditioner, gd, timer ):
        """Create SemiImplicitCrankNicolson-object.
        
        Parameters
        ----------
        td_density: TimeDependentDensity
            the time-dependent density
        td_hamiltonian: TimeDependentHamiltonian
            the time-dependent hamiltonian
        td_overlap: TimeDependentOverlap
            the time-dependent overlap operator
        solver: LinearSolver
            solver for linear equations
        preconditioner: Preconditioner
            preconditioner for linear equations
        gd: GridDescriptor
            coarse (wavefunction) grid descriptor
        timer: Timer
            timer
        
        """
        self.td_density = td_density
        self.td_hamiltonian = td_hamiltonian
        self.td_overlap = td_overlap
        self.solver = solver
        self.preconditioner = preconditioner
        self.gd = gd
        self.timer = timer
        
        self.tmp_psit_nG = None
        self.hpsit = None
        self.spsit = None
        
        
    def propagate(self, kpt_u, time, time_step):
        """Propagate wavefunctions once.
        
        Parameters
        ----------
        kpt_u: List of Kpoints
            K-points
        time: float
            the current time
        time_step: float
            time step
        """
        # temporary wavefunctions
        
        if self.tmp_psit_nG is None:
            self.tmp_psit_nG = []
            for kpt in kpt_u:
                self.tmp_psit_nG.append( self.gd.empty( n=len(kpt.psit_nG),
                                                        dtype=complex ) )

        # copy current wavefunctions to temporary variable
        for u in range(len(kpt_u)):
            self.tmp_psit_nG[u][:] = kpt_u[u].psit_nG
        
        if self.hpsit is None:
            self.hpsit = self.gd.zeros(len(kpt_u[0].psit_nG), dtype=complex)
        if self.spsit is None:
            self.spsit = self.gd.zeros(len(kpt_u[0].psit_nG), dtype=complex)
        
        self.time_step = time_step


        self.timer.start('Update time-dependent operators')

        # rho(t)
        self.td_density.update()
        # H(t)
        self.td_hamiltonian.update(self.td_density.get_density(), time)
        # S(t)
        self.td_overlap.update()

        self.timer.stop('Update time-dependent operators')

        # predict
        self.solve_propagation_equation(kpt_u, time_step)

        self.timer.start('Update time-dependent operators')

        # rho(t+dt)
        self.td_density.update()
        # H(t+dt/2)
        self.td_hamiltonian.half_update( self.td_density.get_density(),
                                         time + time_step )
        # S(t+dt/2)
        self.td_overlap.half_update()

        self.timer.stop('Update time-dependent operators')

        # propagate psit(t), not psit(t+dt), in correct
        for u in range(len(kpt_u)):
            kpt_u[u].psit_nG[:] = self.tmp_psit_nG[u]
        
        # correct
        self.solve_propagation_equation(kpt_u, time_step)
        
        
    # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
    def solve_propagation_equation(self, kpt_u, time_step):
        
        for kpt in kpt_u:
            self.kpt = kpt
            self.timer.start('Apply time-dependent operators')
            self.td_hamiltonian.apply(self.kpt, kpt.psit_nG, self.hpsit)
            self.td_overlap.apply(self.kpt, kpt.psit_nG, self.spsit)
            self.timer.stop('Apply time-dependent operators')

            #psit[:] = self.spsit - .5J * self.hpsit * time_step
            kpt.psit_nG[:] = self.spsit
            multi_zaxpy2(-.5j * self.time_step, self.hpsit, kpt.psit_nG,
                          len(kpt.psit_nG))

            # A x = b
            self.solver.solve(self, kpt.psit_nG, kpt.psit_nG)
            
            
    # ( S + i H dt/2 ) psi
    def dot(self, psi, psin):
        """Applies the propagator matrix to the given wavefunctions.
        
        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result
        
        """
        self.timer.start('Apply time-dependent operators')
        self.td_hamiltonian.apply(self.kpt, psi, self.hpsit)
        self.td_overlap.apply(self.kpt, psi, self.spsit)
        self.timer.stop('Apply time-dependent operators')

        #  psin[:] = self.spsit + .5J * self.time_step * self.hpsit
        psin[:] = self.spsit
        multi_zaxpy2(.5j * self.time_step, self.hpsit, psin,
                     len(psi))


    #  M psin = psi, where M = T (kinetic energy operator)
    def apply_preconditioner(self, psi, psin):
        """Applies preconditioner.
        
        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result
        
        """
        self.timer.start('Solve TDDFT preconditioner')
        if self.preconditioner is not None:
            self.preconditioner.apply(self.kpt, psi, psin)
        else:
            psin[:] = psi
        self.timer.stop('Solve TDDFT preconditioner')
