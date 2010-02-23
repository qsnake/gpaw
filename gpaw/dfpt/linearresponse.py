import numpy as np

import ase.units as units

from gpaw.utilities import unpack

from sternheimeroperator import SternheimerOperator
from linearsolver import LinearSolver
from scipylinearsolver import ScipyLinearSolver

__all__ = ["LinearResponse"]

class LinearResponse:
    """Evaluate linear density response to perturbations in external potential.

    The class provides an implementation of density-functional perturbation
    theory (DFPT^1) in which the static response to perturbations in the external
    potential is calculated in a self-consistent manner from the occupied
    states of the unperturbed system.

    References
    ----------
    1) Rev. Mod. Phys. 73, 515 (2001)
    
    """
    
    def __init__(self, calc):
        """Store calculator and init the LinearResponse calculator."""
        
        # init positions
        calc.set_positions()
        self.calc = calc

        # Store grids
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd

        # wave function derivative
        self.psit1_unG = None
        # effective potential derivative
        self.vIont1_G = None

        self.sternheimer_operator = None
        self.linear_solver = None

        # 1) phonon
        # 2) constant E-field
        # 3) ----
        self.perturbation = None
        
    def calculate_derivative(self, atom, cartesian, eps):
        """Derivate of the local PAW potential wrt an atomic displacement.

        Parameters
        ----------
        atom: int
            Index of the atom
        cartesian: int
            Cartesian component of the displacement
        eps: float
            Magnitude (in Bohr) of the atomic displacement used in the
            finite-difference evaluation of the derivative."""
        
        a = atom
        v = cartesian
        # Array for the derivative of the local part of the PAW potential
        Vloc1t_g = self.finegd.zeros()
        
        # Contributions from compensation charges (ghat) and local potential
        # (vbar)
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar

        # Atomic displacements in scaled coordinates
        eps_s = eps/self.gd.cell_cv[v,v]
        
        # grid for density derivative of ions
        ghat1_g = self.finegd.zeros()

        # Calculate finite-difference derivatives
        spos_ac = self.calc.atoms.get_scaled_positions()
        
        dict_ghat = ghat.dict(zero=True)
        dict_vbar = vbar.dict(zero=True)

        dict_ghat[a] = -1 * self.calc.density.Q_aL[a]
        dict_vbar[a] -= 1.

        spos_ac[a, v] -= eps_s
        ghat.set_positions(spos_ac)
        ghat.add(ghat1_g, dict_ghat)
        vbar.set_positions(spos_ac)
        vbar.add(Vloc1t_g, dict_vbar)

        dict_ghat[a] *= -1
        dict_vbar[a] *= -1
            
        spos_ac[a, v] += 2 * eps_s
        ghat.set_positions(spos_ac)
        ghat.add(ghat1_g, dict_ghat)
        vbar.set_positions(spos_ac)
        vbar.add(Vloc1t_g, dict_vbar)

        # Return to initial positions
        spos_ac[a, v] -= eps_s
        ghat.set_positions(spos_ac)
        vbar.set_positions(spos_ac)

        # Solve Poisson's eq. for the potential from the compensation charges
        hamiltonian = self.calc.hamiltonian
        ps = hamiltonian.poisson
        Vghat1_g = self.finegd.zeros()
        ps.solve(Vghat1_g, ghat1_g)

        Vloc1t_g += Vghat1_g
        
        # Convert change in the potential to a derivative
        d = 2 * eps
        Vloc1t_g /= d
        
        # Transfer to coarse grid
        Vloc1t_G = self.gd.zeros()
        hamiltonian.restrictor.apply(Vloc1t_g, Vloc1t_G)

        self.Vloc1t_g = Vloc1t_g.copy() / d
        self.ghat1_g = ghat1_g / d
        self.Vghat1_g = Vghat1_g.copy() / d
        
        return Vloc1t_G

    def calculate_nonlocal_derivative(self, atom, cartesian, eps):
        """Derivate of the non-local PAW potential wrt an atomic displacement.

        Parameters
        ----------
        atom: int
            Index of the atom
        cartesian: int
            Cartesian component of the displacement
        eps: float
            Magnitude (in Bohr) of the atomic displacement used in the
            finite-difference evaluation of the derivative."""
        
        a = atom
        v = cartesian
        nbands = self.calc.wfs.nvalence/2

        hamiltonian = self.calc.hamiltonian
        
        # Array for the derivative of the non-local part of the PAW potential
        pt1_ng = self.finegd.zeros(n=nbands)
        
        # Projectors on the atom
        pt = self.calc.wfs.pt #lfc
        P_ni = self.calc.wfs.kpt_u[0].P_ani[a][:nbands]
        dH_ii = unpack(hamiltonian.dH_asp[a][0])

        # Does the order matter here ??????
        M_ni = np.dot(P_ni, dH_ii)
        
        # Atomic displacements in scaled coordinates
        eps_s = eps/self.gd.cell_cv[v,v]
        
        # Calculate finite-difference derivatives
        spos_ac = self.calc.atoms.get_scaled_positions()
        
        dict_pt = pt.dict(shape=(nbands,), zero=True)
        print dict_pt[a].shape
        print M_ni.shape
        dict_pt[a] -= M_ni

        spos_ac[a, v] -= eps_s
        pt.set_positions(spos_ac)
        pt.add(pt1_ng, dict_pt)

        dict_pt[a] *= -1
            
        spos_ac[a, v] += 2 * eps_s
        pt.set_positions(spos_ac)
        pt.add(pt1_ng, dict_pt)

        # Return to initial positions
        spos_ac[a, v] -= eps_s
        pt.set_positions(spos_ac)
        
        # Convert change to a derivative
        d = 2 * eps
        pt1_ng /= d
        
        # Transfer to coarse grid
        pt1_nG = self.gd.zeros(n=nbands)
        print pt1_nG.shape
        print pt1_ng.shape

        hamiltonian.restrictor.apply(pt1_ng, pt1_nG)
        stop
        
        return pt1_nG
    
    def calculate_response(self, a, c, alpha = 0.4,
                           eps = 0.01/units.Bohr,
                           tolerance_sc = 1e-5,
                           tolerance_sternheimer = 1e-5):
        """Calculate linear density response for given q-vector.

        Implementation of q != 0 to be done!
        
        """

        components = ['x','y','z']
        atoms = self.calc.get_atoms()
        symbols = atoms.get_chemical_symbols()
        print "Atom index: %i" % a
        print "Atomic symbol: %s" % symbols[a]
        print "Component: %s" % components[c]
        
        hamiltonian = self.calc.hamiltonian
        wfs = self.calc.wfs
        kpt_u = self.calc.wfs.kpt_u
        num_kpts = len(kpt_u)
        num_occ_bands = self.calc.wfs.nvalence/2
        
        # Linear solver for the solution of Sternheimer equation
        self.linear_solver = ScipyLinearSolver(tolerance = tolerance_sternheimer)
        # Linear operator in the Sternheimer equation
        self.sternheimer_operator = SternheimerOperator(hamiltonian, wfs, self.gd)
        # List for storing the variations in the wave-functions
        self.psit1_unG = np.array([[self.gd.zeros() for j in range(num_occ_bands)]
                                   for i in range(num_kpts)],dtype=float)

        # Variation of the local part of the pseudo-potential
        self.Vloc1t_G = self.calculate_derivative(a, c, eps)
        
        for iter in range(100):
            if iter == 0:
                self.first_iteration()
            else:
                print "iter:%3.i\t" % iter,
                norm = self.iteration(iter, alpha)
                print "\t\tabs-norm: %6.3e\t" % norm,
                print "integrated density response: %5.2e" % \
                      self.gd.integrate(self.nt1_G)
        
                if norm < tolerance_sc:
                    print "self-consistent loop converged in %i iterations" \
                          % iter
                    break

        return self.nt1_G.copy()
    

    def first_iteration(self):
        """Perform first iteration of sc-loop."""

        self.wave_function_variations(self.Vloc1t_G)
        self.nt1_G = self.density_response()

    def iteration(self, iter, alpha):
        """Perform iteration.

        Parameters
        ----------
        iter: int
            Iteration number
        alpha: float
            Linear mixing parameter

        """

        # Copy old density
        nt1_G_old = self.nt1_G.copy()
        # Update variation in the effective potential
        v1_G = self.effective_potential_variation()
        # Update wave function variations
        self.wave_function_variations(v1_G)
        # Update density
        nt1_G = self.density_response()
        # Mix
        self.nt1_G = alpha * nt1_G + (1. - alpha) * nt1_G_old
        # Integrated absolute density change
        norm = self.gd.integrate(np.abs(self.nt1_G - nt1_G_old))

        return norm

    def effective_potential_variation(self):
        """Calculate variation in the effective potential."""
        
        # Calculate new effective potential
        density = self.calc.density
        nt1_g = self.finegd.zeros()
        density.interpolator.apply(self.nt1_G, nt1_g)
        hamiltonian = self.calc.hamiltonian
        ps = hamiltonian.poisson
        # Hartree part
        vHXC1_g = self.finegd.zeros()
        ps.solve(vHXC1_g, nt1_g)
        # XC part
        nt_g_ = density.nt_g.ravel()
        vXC1_g = self.finegd.zeros()
        vXC1_g.shape = nt_g_.shape
        hamiltonian.xcfunc.calculate_fxc_spinpaired(nt_g_, vXC1_g)
        vXC1_g.shape = nt1_g.shape
        vHXC1_g += vXC1_g * nt1_g
        # Transfer to coarse grid
        v1_G = self.gd.zeros()
        hamiltonian.restrictor.apply(vHXC1_g, v1_G)
        # Add ionic part
        v1_G += self.Vloc1t_G
        # self.v1_G = v1_G.copy()

        return v1_G
    
    def wave_function_variations(self, v1_G):
        """Calculate variation in the wave-functions.

        Parameters
        ----------
        v1_G: ndarray
            Variation of the effective potential (PS + Hartree + XC)

        """

        nvalence = self.calc.wfs.nvalence
        kpt_u = self.calc.wfs.kpt_u

        for kpt in kpt_u:

            psit_nG = kpt.psit_nG[:nvalence/2]
            psit1_nG = self.psit1_unG[kpt.k]
            
            for n in range(nvalence/2):

                psit_G = psit_nG[n]
                psit1_G = psit1_nG[n]

                # rhs of Sternheimer equation
                rhs_G = -1. * v1_G * psit_G
                # Update k-point and band index in SternheimerOperator
                self.sternheimer_operator.set_blochstate(kpt.k, n)
                self.sternheimer_operator.project(rhs_G)
                print "Solving Sternheimer -",
                iter, info = self.linear_solver.solve(self.sternheimer_operator,
                                                      psit1_G, rhs_G)
                if info == 0:
                    print "Linear solver converged in %i iterations" % iter
                elif info > 0:
                    print ("Linear solver did not converge in %i iterations" %
                           iter)
                    assert info == 0
                else:
                    print "Linear solver failed" 
                    assert info == 0
                    
    def density_response(self):
        """Calculate density response from variation in the wave-functions."""

        nt1_G = self.gd.zeros()
    
        nvalence = self.calc.wfs.nvalence
        kpt_u = self.calc.wfs.kpt_u

        for kpt in kpt_u:

            psit_nG = kpt.psit_nG[:nvalence/2]
            psit1_nG = self.psit1_unG[kpt.k]

            for psit_G, psit1_G in zip(psit_nG, psit1_nG):

                nt1_G += 4 * psit_G * psit1_G

        return nt1_G
    




