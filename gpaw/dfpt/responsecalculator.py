"""This module implements a linear response calculator class."""

__all__ = ["ResponseCalculator"]

import numpy as np

from gpaw.transformers import Transformer
# from gpaw.mixer import BaseMixer
from gpaw.dfpt.mixer import BaseMixer

from gpaw.dfpt.sternheimeroperator import SternheimerOperator
from gpaw.dfpt.scipylinearsolver import ScipyLinearSolver
from gpaw.dfpt.preconditioner import ScipyPreconditioner


class ResponseCalculator:
    """This class is a calculator for the sc density variation.

    From the given perturbation, the set of coupled equations for the
    first-order density response is solved self-consistently.

    Parameters
    ----------
    max_iter: int
        Maximum number of iterations in the self-consistent evaluation of
        the density variation
    tolerance_sc: float
        Tolerance for the self-consistent loop measured in terms of
        integrated absolute change of the density derivative between two
        iterations
    tolerance_sternheimer: float
        Tolerance for the solution of the Sternheimer equation -- passed to
        the ``LinearSolver``
    beta: float (0 < beta < 1)
        Mixing coefficient
    nmaxold: int
        Length of history for the mixer.
    weight: int
        Weight for the mixer metric (=1 -> no metric used).
        
    """

    parameters = {'verbose':               True,
                  'max_iter':              100,
                  'max_iter_krylov':       100,
                  'tolerance_sc':          1.0e-4,
                  'tolerance_sternheimer': 1e-5,
                  'use_pc':                True,
                  'beta':                  0.4,
                  'nmaxold':               3,
                  'weight':                50
                  }
    
    def __init__(self, calc, wfs, perturbation, kpointdescriptor,
                 poisson_solver=None, **kwargs):
        """Store calculator etc.

        Parameters
        ----------
        calc: Calculator
            Calculator instance containing a ground-state calculation
            (calc.set_positions must have been called before this point!).
        wfs: WaveFunctions
            Class taking care of wave-functions, projectors, k-point related
            quantities and symmetries.
        perturbation: Perturbation
            Class implementing the perturbing potential. Must provide an
            ``apply`` member function implementing the multiplication of the 
            perturbing potential to a (set of) state vector(s).
            
        """
        
        # Store ground-state quantities
        self.hamiltonian = calc.hamiltonian
        self.density = calc.density

        self.wfs = wfs
        self.perturbation = perturbation

        # Get list of k-point containers
        self.kpt_u = wfs.kpt_u
        self.kd = kpointdescriptor

        # Poisson solver
        if hasattr(perturbation, 'solve_poisson'):
            self.solve_poisson = perturbation.solve_poisson
        else:
            assert poisson_solver is not None, "No Poisson solver given"

            self.poisson = poisson_solver
            self.solve_poisson = self.poisson.solve_neutral
       
        # Store grid-descriptors
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd

        # dtype for ground-state wave-functions
        self.gs_dtype = calc.wfs.dtype
        # dtype for the perturbing potential and density
        self.dtype = perturbation.dtype
        
        # Grid transformer -- convert array from coarse to fine grid
        self.interpolator = Transformer(self.gd, self.finegd, nn=3,
                                        dtype=self.dtype, allocate=False)
        # Grid transformer -- convert array from fine to coarse grid
        self.restrictor = Transformer(self.finegd, self.gd, nn=3,
                                      dtype=self.dtype, allocate=False)

        # Sternheimer operator
        self.sternheimer_operator = None
        # Krylov solver
        self.linear_solver = None

        # Phases for transformer objects - since the perturbation determines
        # the form of the density response this is obtained from the
        # perturbation in the ``__call__`` member function below.
        self.phase_cd = None

        # Array attributes
        self.nt1_g = None
        self.nt1_G = None
        self.vH1_g = None
        
        # Number of occupied bands
        nvalence = calc.wfs.nvalence
        self.nbands = nvalence/2 + nvalence%2
        assert self.nbands <= calc.wfs.nbands
                                  
        self.initialized = False

        self.parameters = {}
        self.set(**kwargs)

    def __call__(self):
        """Calculate density derivative."""

        assert self.initialized, ("Linear response calculator "
                                  "not initizalized.")

        # Parameters
        p = self.parameters
        max_iter = p['max_iter']
        tolerance = p['tolerance_sc']
        
        # Reset mixer
        self.mixer.reset()
        # Reset wave-functions
        self.wfs.reset()

        # Set phase attribute for Transformer objects
        self.phase_cd = self.perturbation.get_phase_cd()
        
        for iter in range(max_iter):

            if iter == 0:
                self.first_iteration()
            else:
                print "iter:%3i\t" % iter,
                norm = self.iteration()
                print "abs-norm: %6.3e\t" % norm,
                print ("integrated density response (abs): % 5.2e (%5.2e) "
                       % (self.gd.integrate(self.nt1_G.real), 
                          self.gd.integrate(np.absolute(self.nt1_G))))
                       
                if norm < tolerance:
                    print ("self-consistent loop converged in %i iterations"
                           % iter)
                    break
                
            if iter == max_iter-1:
                print     ("self-consistent loop did not converge in %i "
                           "iterations" % (iter+1))
                
        return self.nt1_G.copy()
    
    def set(self, **kwargs):
        """Set parameters for calculation."""

        # Check for legal input parameters
        for key, value in kwargs.items():

            if not key in ResponseCalculator.parameters:
                raise TypeError("Unknown keyword argument: '%s'" % key)

        # Insert default values if not given
        for key, value in ResponseCalculator.parameters.items():

            if key not in kwargs:
                kwargs[key] = value

        self.parameters.update(kwargs)
            
    def initialize(self, spos_ac):
        """Make the object ready for a calculation."""

        # Parameters
        p = self.parameters
        beta = p['beta']
        nmaxold = p['nmaxold']
        weight = p['weight']
        use_pc = p['use_pc']
        tolerance_sternheimer = p['tolerance_sternheimer']
        max_iter_krylov = p['max_iter_krylov']
        
        # Initialize WaveFunctions attribute
        self.wfs.initialize(spos_ac)
        
        # Initialize interpolator and restrictor
        self.interpolator.allocate()
        self.restrictor.allocate()
        
        # Initialize mixer
        # weight = 1 -> no metric is used
        self.mixer = BaseMixer(beta=beta, nmaxold=nmaxold,
                               weight=weight, dtype=self.dtype)
        self.mixer.initialize_metric(self.gd)
        
        # Linear operator in the Sternheimer equation
        self.sternheimer_operator = \
            SternheimerOperator(self.hamiltonian, self.wfs, self.gd,
                                dtype=self.gs_dtype)

        # Preconditioner for the Sternheimer equation
        if p['use_pc']:
            pc = ScipyPreconditioner(self.gd,
                                     self.sternheimer_operator.project,
                                     dtype=self.gs_dtype)
        else:
            pc = None

        # Temp ??
        self.pc = pc
        # Linear solver for the solution of Sternheimer equation            
        self.linear_solver = ScipyLinearSolver(tolerance=tolerance_sternheimer,
                                               preconditioner=pc,
                                               max_iter=max_iter_krylov)

        self.initialized = True

    def first_iteration(self):
        """Perform first iteration of sc-loop."""

        self.wave_function_variations()
        self.density_response()
        self.mixer.mix(self.nt1_G, [], phase_cd=self.phase_cd)
        self.interpolate_density()

        #XXX Temp - in order to see the Hartree potential after 1'st iteration
        v1_G = self.effective_potential_variation()
        
    def iteration(self):
        """Perform iteration."""

        # Update variation in the effective potential
        vHXC1_G = self.effective_potential_variation()
        # Update wave function variations
        self.wave_function_variations(vHXC1_G=vHXC1_G)
        # Update density
        self.density_response()
        # Mix - supply phase_cd here for metric inside the mixer
        self.mixer.mix(self.nt1_G, [], phase_cd=self.phase_cd)
        norm = self.mixer.get_charge_sloshing()

        self.interpolate_density()
       
        return norm

    def interpolate_density(self):
        """Interpolate density derivative onto the fine grid."""

        self.nt1_g = self.finegd.zeros(dtype=self.dtype)
        self.interpolator.apply(self.nt1_G, self.nt1_g, phases=self.phase_cd)
        
    def effective_potential_variation(self):
        """Calculate derivative of the effective potential (Hartree + XC)."""

        # Hartree part
        vHXC1_g = self.finegd.zeros(dtype=self.dtype)
        self.solve_poisson(vHXC1_g, self.nt1_g)
        # Store for evaluation of second order derivative
        self.vH1_g = vHXC1_g.copy()
        
        # XC part - fix this in the xc_functional.py file !!!!
        nt_g = self.density.nt_g
        nt_g_ = nt_g.ravel()
        vXC1_g = self.finegd.zeros(dtype=float)
        vXC1_g.shape = nt_g_.shape
        hamiltonian = self.hamiltonian
        hamiltonian.xcfunc.calculate_fxc_spinpaired(nt_g_, vXC1_g)
        vXC1_g.shape = self.nt1_g.shape
        vHXC1_g += vXC1_g * self.nt1_g

        # Transfer to coarse grid
        vHXC1_G = self.gd.zeros(dtype=self.dtype)
        self.restrictor.apply(vHXC1_g, vHXC1_G, phases=self.phase_cd)
        
        return vHXC1_G
    
    def wave_function_variations(self, vHXC1_G=None):
        """Calculate variation in the wave-functions.

        Parameters
        ----------
        v1_G: ndarray
            Variation of the local effective potential (Hartree + XC).

        """

        verbose = self.parameters['verbose']

        if verbose:
            print "Calculating wave function variations"

        if self.perturbation.has_q():
            q_c = self.perturbation.get_q()
            kplusq_k = self.kd.find_k_plus_q(q_c)
        else:
            kplusq_k = None

        # XXX Temp
        # self.rhs_nG = self.gd.zeros(n=self.nbands, dtype=self.gs_dtype)
            
        # Calculate wave-function variations for all k-points.
        for kpt in self.kpt_u:

            k = kpt.k

            if verbose:
                print "k-point %2.1i" % k
            
            # Index of k+q vector
            if kplusq_k is None:
                kplusq = k
                kplusqpt = kpt
            else:
                kplusq = kplusq_k[k]
                kplusqpt = self.kpt_u[kplusq]

            # Ground-state and first-order wave-functions
            psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # Update the SternheimerOperator
            self.sternheimer_operator.set_kplusq(kplusq)
            
            # Right-hand side of Sternheimer equations
            rhs_nG = self.gd.zeros(n=self.nbands, dtype=self.gs_dtype)
            
            # k and k+q
            # XXX should only be done once but maybe too cheap to bother ??
            self.perturbation.apply(psit_nG, rhs_nG, self.wfs, k, kplusq)
            
            if self.pc is not None:
                # k+q
                self.pc.set_kpt(kplusqpt)
                
            # Loop over occupied bands
            for n in range(self.nbands):

                # Get view of the Bloch function and its variation
                psit_G  = psit_nG[n]
                psit1_G = psit1_nG[n]

                # Rhs of Sternheimer equation                
                rhs_G = -1 * rhs_nG[n]

                if vHXC1_G is not None:
                    rhs_G -= vHXC1_G * psit_G
                    
                # Update k-point index and band index in SternheimerOperator
                self.sternheimer_operator.set_blochstate(n, k)
                self.sternheimer_operator.project(rhs_G)

                # XXX Temp
                #if k == 0:
                #    self.rhs_nG[n] = rhs_G.copy()
                
                if verbose:
                    print "\tBand %2.1i -" % n,

                iter, info = self.linear_solver.solve(self.sternheimer_operator,
                                                      psit1_G, rhs_G)
                
                if info == 0:
                    if verbose: 
                        print "linear solver converged in %i iterations" % iter
                elif info > 0:
                    assert False, ("linear solver did not converge in maximum "
                                   "number (=%i) of iterations" % iter)
                else:
                    assert False, ("linear solver failed to converge")

                
    def density_response(self):
        """Calculate density response from variation in the wave-functions."""

        # Note, density might be complex
        self.nt1_G = self.gd.zeros(dtype=self.dtype)

        for kpt in self.kpt_u:

            # The occupations includes the weight of the k-points
            f_n = kpt.f_n
            # Use weight of k-point instead of occupation -- spin degeneracy is
            # included in the weight
            w = kpt.weight
            # Wave functions
            psit_nG = kpt.psit_nG
            psit1_nG = kpt.psit1_nG

            # for n in range(self.nbands):
            for n, f in enumerate(f_n):
                # NOTICE: this relies on the automatic down-cast of the complex
                # array on the rhs to a real array when the lhs is real !!
                # Factor 2 for time-reversal symmetry
                self.nt1_G += 2 * w * psit_nG[n].conjugate() * psit1_nG[n]
                #XXX
                ## self.nt1_G += f * (psit_nG[n].conjugate() * psit1_nG[n] +
                ##                    psit1_nG[n].conjugate() * psit_nG[n])

