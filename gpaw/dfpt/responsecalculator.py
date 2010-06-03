"""This module implements a linear response calculator class."""

__all__ = ["ResponseCalculator"]

import numpy as np

from gpaw.transformers import Transformer
from gpaw.poisson import PoissonSolver, FFTPoissonSolver
# from gpaw.mixer import BaseMixer
from gpaw.dfpt.mixer import BaseMixer

from gpaw.dfpt.sternheimeroperator import SternheimerOperator
from gpaw.dfpt.scipylinearsolver import ScipyLinearSolver
from gpaw.dfpt.preconditioner import ScipyPreconditioner


class ResponseCalculator:
    """This class is a calculator for the sc density variation.

    From the given perturbation, the set of coupled equations for the
    first-order density response is solved self-consistently.
    
    """
    
    def __init__(self, calc, perturbation, poisson_solver=None, **kwargs):
        """Store calculator etc.

        Parameters
        ----------
        calc: Calculator
            Calculator instance containing a ground-state calculation.
        perturbation: Perturbation
            Class implementing the perturbing potential. Must provide an
            ``apply`` member function that knows how to apply the perturbation
            to a (set of) state vector(s).
            
        """
        
        # Make sure that localized functions are initialized
        calc.set_positions()
        atoms = calc.get_atoms()
        
        self.calc = calc
        self.perturbation = perturbation

        if hasattr(perturbation, 'solve_poisson'):
            self.solve_poisson = perturbation.solve_poisson
        else:
            assert poisson_solver is not None, "No Poisson solver given"

            self.poisson = poisson_solver
            self.solve_poisson = self.poisson.solve_neutral
            
        # Get list of k-points
        self.kpt_u = self.calc.wfs.kpt_u

        # Store grid-descriptors
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd

        # dtype for ground-state wave-functions
        self.gs_dtype = calc.wfs.dtype
        # dtype for the perturbing potential
        self.dtype = perturbation.dtype
        
        # Grid transformer -- convert array from coarse to fine grid
        self.interpolator = Transformer(self.gd, self.finegd, nn=3,
                                        dtype=self.dtype, allocate=False)
        # Grid transformer -- convert array from fine to coarse grid
        self.restrictor = Transformer(self.finegd, self.gd, nn=3,
                                      dtype=self.dtype, allocate=False)

        # Wave-function derivative
        self.psit1_unG = None
        # Sternheimer operator
        self.sternheimer_operator = None
        # Krylov solver
        self.linear_solver = None

        # Number of occupied bands
        self.nbands = max(1, self.calc.wfs.nvalence/2)
        assert self.nbands <= calc.wfs.nbands
                                  
        self.initialized = False
        
    def initialize(self, tolerance_sternheimer=1.0e-5, use_pc=False):
        """Make the object ready for a calculation."""

        hamiltonian = self.calc.hamiltonian
        wfs = self.calc.wfs

        # Initialize interpolator and restrictor
        self.interpolator.allocate()
        self.restrictor.allocate()
        
        # Initialize mixer
        # weight = 1 -> no metric is used
        self.mixer = BaseMixer(beta=0.4, nmaxold=5, weight=1)
        self.mixer.initialize_metric(self.gd)
        
        # Linear operator in the Sternheimer equation
        self.sternheimer_operator = \
            SternheimerOperator(hamiltonian, wfs, self.gd, dtype=self.gs_dtype)

        # Preconditioner for the Sternheimer equation
        if use_pc:
            pc = ScipyPreconditioner(self.gd,
                                     self.sternheimer_operator.project,
                                     dtype=self.gs_dtype)
        else:
            pc = None

        # Temp ??
        self.pc = pc
        # Linear solver for the solution of Sternheimer equation            
        self.linear_solver = ScipyLinearSolver(tolerance=tolerance_sternheimer,
                                               preconditioner=pc)

        self.initialized = True
        
    def __call__(self, maxiter=1000, tolerance_sc=1.0e-4,
                 tolerance_sternheimer=1e-5):
        """Calculate linear density response.

        Parameters
        ----------
        maxiter: int
            Maximum number of iterations in the self-consistent evaluation of
            the density variation
        tolerance_sc: float
            Tolerance for the self-consistent loop measured in terms of
            integrated absolute change of the density derivative between two
            iterations
        tolerance_sternheimer: float
            Tolerance for the solution of the Sternheimer equation -- passed to
            the ``LinearSolver``
            
        """

        assert self.initialized, ("Linear response calculator "
                                  "not initizalized.")

            
        # Reset mixer
        self.mixer.reset()
        
        # List the variations of the wave-functions
        self.psit1_unG = [self.gd.zeros(n=self.nbands, dtype=self.gs_dtype)
                          for kpt in self.kpt_u]

        ############## Remove this when time comes ###########################
        components = ['x','y','z']
        atoms = self.calc.get_atoms()
        symbols = atoms.get_chemical_symbols()
        print "Atom index: %i" % self.perturbation.a
        print "Atomic symbol: %s" % symbols[self.perturbation.a]
        print "Component: %s" % components[self.perturbation.v]
        ######################################################################
        
        for iter in range(maxiter):
            print     "iter:%3i\t" % iter,
            print     "Calculating wave function variations"            
            if iter == 0:
                self.first_iteration()
            else:
                norm = self.iteration()
                print "abs-norm: %6.3e\t" % norm,
                # The density is complex !!!!!!
                print "integrated density response: %5.2e" % \
                      self.gd.integrate(self.nt1_G)
        
                if norm < tolerance_sc:
                    print ("self-consistent loop converged in %i iterations"
                           % iter)
                    break
                
            if iter == maxiter-1:
                print     ("self-consistent loop did not converge in %i "
                           "iterations" % (iter+1))
                
        return self.nt1_G.copy(), self.psit1_unG
    
    def first_iteration(self):
        """Perform first iteration of sc-loop."""

        self.wave_function_variations()
        self.nt1_G = self.density_response()
        self.mixer.mix(self.nt1_G, [])

    def iteration(self):
        """Perform iteration."""

        # Update variation in the effective potential
        v1_G = self.effective_potential_variation()
        # Update wave function variations
        self.wave_function_variations(v1_G)
        # Update density
        self.nt1_G = self.density_response()
        # Mix
        self.mixer.mix(self.nt1_G, [])
        norm = self.mixer.get_charge_sloshing()

        return norm

    def effective_potential_variation(self):
        """Calculate variation in the effective potential."""

        # Get phases for the transformation between grids from the perturbation
        phase_cd = self.perturbation.phase_cd
        
        # Calculate new effective potential
        nt1_g = self.finegd.zeros(dtype=self.dtype)
        self.interpolator.apply(self.nt1_G, nt1_g, phases=phase_cd)

        # Hartree part
        vHXC1_g = self.finegd.zeros(dtype=self.dtype)
        self.solve_poisson(vHXC1_g, nt1_g)

        # XC part - fix this in the xc_functional.py file !!!!
        density = self.calc.density
        nt_g_ = density.nt_g.ravel()
        vXC1_g = self.finegd.zeros(dtype=float)
        vXC1_g.shape = nt_g_.shape
        hamiltonian = self.calc.hamiltonian
        hamiltonian.xcfunc.calculate_fxc_spinpaired(nt_g_, vXC1_g)
        vXC1_g.shape = nt1_g.shape
        vHXC1_g += vXC1_g * nt1_g

        # Transfer to coarse grid
        v1_G = self.gd.zeros(dtype=self.dtype)
        self.restrictor.apply(vHXC1_g, v1_G, phases=phase_cd)
        
        return v1_G
    
    def wave_function_variations(self, v1_G=None):
        """Calculate variation in the wave-functions.

        Parameters
        ----------
        v1_G: ndarray
            Variation of the local effective potential (Hartree + XC)

        """

        # Calculate wave-function variations for all k-points.
        for kpt in self.kpt_u:

            k = kpt.k
            print "k-point %2.1i" % k
            psit_nG = kpt.psit_nG[:self.nbands]
            psit1_nG = self.psit1_unG[k]

            # Right-hand side of Sternheimer equation
            rhs_nG = self.gd.zeros(n=self.nbands, dtype=self.gs_dtype)
            # k
            self.perturbation.apply(psit_nG, rhs_nG, kpt=kpt)

            if self.pc is not None:
                # k+q
                self.pc.set_phases(kpt.phase_cd)
                
            # Loop over all valence-bands
            for n in range(self.nbands):

                # Get view of the Bloch function and its variation
                psit_G = psit_nG[n]
                psit1_G = psit1_nG[n]
                rhs_G = -1 * rhs_nG[n]
                
                # Update k-point and band index in SternheimerOperator
                self.sternheimer_operator.set_blochstate(n, k)

                # Rhs of Sternheimer equation
                if v1_G is not None:
                    rhs_G -= v1_G * psit_G
                
                self.sternheimer_operator.project(rhs_G)

                print "\tBand %2.1i -" % n,
                iter, info = self.linear_solver.solve(self.sternheimer_operator,
                                                      psit1_G, rhs_G)

                if info == 0:
                    print "linear solver converged in %i iterations" % iter
                elif info > 0:
                    assert info == 0, ("linear solver did not converge in "
                                       "maximum number (=%i) of iterations"
                                       % iter)
                else:
                    assert info == 0, ("linear solver failed to converge")
                
    def density_response(self):
        """Calculate density response from variation in the wave-functions."""

        # Note - density might be complex
        nt1_G = self.gd.zeros(dtype=self.dtype)
        
        for kpt in self.kpt_u:

            # The occupations includes the weight of the k-points
            f_n = kpt.f_n[:self.nbands]
            psit_nG = kpt.psit_nG[:self.nbands]
            psit1_nG = self.psit1_unG[kpt.k]

            for n, f in enumerate(f_n):
                # NOTICE: this relies on the automatic down-cast of the complex
                # array on the rhs to a real array when the lhs is real !!
                # Factor 2 for time-reversal symmetry
                nt1_G += 2 * f * (psit_nG[n].conjugate() * psit1_nG[n])

        return nt1_G
