"""This module implements a phonon perturbation."""

__all__ = ["PhononPerturbation"]

from math import sqrt, pi

import numpy as np
import numpy.linalg as la

from gpaw.utilities import unpack, unpack2
from gpaw.transformers import Transformer
from gpaw.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.lfc import LocalizedFunctionsCollection as LFC

from gpaw.dfpt.perturbation import Perturbation


class PhononPerturbation(Perturbation):
    """Implementation of phonon-related terms for the Sternheimer equation."""
    
    def __init__(self, calc, **kwargs):
        """Store useful objects, e.g. lfc's for the various atomic functions.
            
        Depending on whether the system is periodic or finite, Poisson's equation
        is solved with FFT or multigrid techniques, respectively.
       
        """

        self.calc = calc
        
        # Use same q-point grid as the k-point grid of the ground-state
        # calculation 
        # self.qpts_u = self.calc.wfs.kpt_u
        
        # Boundary conditions
        pbc_c = calc.atoms.get_pbc()
        
        if np.all(pbc_c == False):
            self.gamma = True
            self.dtype = float
            self.phase_cd = None            
            # Multigrid Poisson solver
            self.poisson = PoissonSolver()
        else:
            # For now do Gamma calculation (wrt q-vector!)
            if True: #len(self.qpts_u) == 1:
                self.gamma = True
                # Modified to test the implementation with complex quantities
                self.dtype = complex
                self.gamma = False
                self.ibzk_qc = np.array(((0, 0, 0)), dtype=float)
                # Phase factors for the transformation between fine and coarse grids
                # Look in wavefunction.py for q != 0
                self.phase_cd = np.ones((3, 2), dtype=complex)        
            else:
                self.gamma = False
                self.dtype = complex
                # Get k-points -- only temp, I need q-vectors; maybe the same ???
                self.ibzk_qc = self.calc.get_ibz_k_points()
                
            # FFT Poisson solver
            self.poisson = FFTPoissonSolver(dtype=self.dtype)

        # Use existing ghat and vbar instances -- in case of periodic BC's
        # set the k-points and update (see ``initialize`` member function)
        self.ghat = calc.density.ghat
        self.vbar = calc.hamiltonian.vbar

        # Projectors on the atoms
        self.pt = self.calc.wfs.pt
        
        # Store grid-descriptors
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd
        
        # Grid transformer -- convert array from fine to coarse grid
        self.restrictor = Transformer(self.finegd, self.gd, nn=3,
                                      dtype=self.dtype, allocate=False)

        # Atom, cartesian coordinate and q-vector of the perturbation
        self.a = None
        self.v = None
        
        # Gamma-point calculation
        if self.gamma:
            self.q = -1
        else:
            # Modified to test the implementation with complex quantities
            self.q = 0 #None

        # Coefficients needed for the non-local part of the perturbation
        self.P_ani = None
        self.dP_aniv = None

    def initialize(self):
        """Prepare the various attributes for a calculation."""

        if not self.gamma:

            # Get scaled atomic positions
            spos_ac = self.calc.atoms.get_scaled_positions()
            
            # Set q-vectors and update 
            self.ghat.set_k_points(self.ibzk_qc)
            self.ghat._update(spos_ac)
            # Set q-vectors and update 
            self.vbar.set_k_points(self.ibzk_qc)
            self.vbar._update(spos_ac)

            # Phase factor exp(i*q*r) needed to obtian the periodic part of lfc
            coor_vg = self.finegd.get_grid_point_coordinates()
            cell_cv = self.finegd.cell_cv
            # Convert to scaled coordinates
            scoor_cg = np.dot(la.inv(cell_cv), coor_vg.swapaxes(0, -2))
            scoor_cg = scoor_cg.swapaxes(1,-2)
            # Phase factor
            phase_qg = np.exp(2j * pi *
                              np.dot(self.ibzk_qc, scoor_cg.swapaxes(0,-2)))
            self.phase_qg = phase_qg.swapaxes(1, -2)
            
        # Setup the Poisson solver -- to be used on the fine grid
        self.poisson.set_grid_descriptor(self.finegd)
        self.poisson.initialize()
        # Grid transformer
        self.restrictor.allocate()
        # Calculate coefficients needed for the non-local part of the PP
        self.calculate_dP_aniv()

    def get_dtype(self):
        """Return dtype for the phonon perturbation."""

        return self.dtype
    
    def set_perturbation(self, a, v):
        """Set atom and cartesian coordinate of the perturbation.

        Parameters
        ----------
        a: int
            Index of the atom.
        v: int 
            Cartesian component (0, 1 or 2) of the atomic displacement.
            
        """

        self.a = a
        self.v = v
        # Invalidate calculated quantities

    def set_q(self, q):
        """Set the index of the q-vector of the perturbation."""

        self.q = q
        # Check that the index is in allowed range
        # Update self.phase_cd attribute
        
    def solve_poisson(self, phi_g, rho_g):
        """Solve Poisson's equation for a Bloch-type charge distribution.

        More to come here ...
        
        Parameters
        ----------
        phi_g: GridDescriptor
            Grid for the solution of Poissons's equation.
        rho_g: GridDescriptor
            Grid with the charge distribution.

        """

        assert phi_g.shape == rho_g.shape == self.phase_qg.shape[-3:], \
               ("Arrays have incompatible shapes.")
        assert self.q is not None, ("q-vector not set")
        
        # Solve Poisson's eq. for the potential from the periodic part of the
        # compensation charge derivative

        if self.gamma: # Gamma point calculation wrt the q-vector
            # NOTICE: solve_neutral
            self.poisson.solve_neutral(v1_g, ghat1_g)  
        else:
            # Divide out the phase factor to get the periodic part 
            rho_g /= self.phase_qg[self.q]
            # Solve Poisson's equation for the periodic part of the potential
            # NOTICE: solve_neutral
            self.poisson.solve_neutral(phi_g, rho_g)  
            # Return to Bloch form
            phi_g *= self.phase_qg[self.q]
            
    def calculate_dP_aniv(self, kpt=None):
        """Coefficients for the derivative of the non-local part of the PP.

        Parameters
        ----------
        kpt: k-point
            K-point of the Bloch state on which the non-local potential acts
            upon

        The calculated coefficients are the following (except for an overall
        sign of -1; see ``derivative`` method of the ``lfc`` class)::

                     /        a*           
          dP_aniv =  | dG dPhi  (G) Psi (G) ,
                     /        iv       n

        where::
                       
              a        d       a
          dPhi  (G) =  ---  Phi (G) .
              iv         a     i
                       dR

        """

        # Projectors on the atoms
        pt = self.pt
        nbands = self.calc.wfs.nvalence/2
        # Integration dict
        dP_aniv = pt.dict(nbands, derivative=True)
        # Wave functions
        psit_nG = self.calc.wfs.kpt_u[0].psit_nG[:nbands]
        # Integrate with derivative of projectors
        pt.derivative(psit_nG, dP_aniv)
        # Store the coefficients
        self.dP_aniv = dP_aniv

    def calculate_derivative(self):
        """Derivate of the local PAW potential wrt an atomic displacements.

        The local part of the PAW potential has contributions from the
        compensation charges (``ghat``) and a spherical symmetric atomic
        potential (``vbar``).
        
        """

        assert self.a is not None
        assert self.v is not None
        assert self.q is not None
        
        a = self.a
        v = self.v
        
        # LFC's
        ghat = self.ghat
        vbar = self.vbar
        # Expansion coefficients for the ghat functions
        Q_aL = ghat.dict(zero=True)
        # Remember sign convention for add_derivative method
        Q_aL[a] = -1 * self.calc.density.Q_aL[a]

        # Grid for derivative of compensation charges
        ghat1_g = self.finegd.zeros(dtype=self.dtype)
        ghat.add_derivative(a, v, ghat1_g, Q_aL, q=self.q)
        # ghat.add(ghat1_g, Q_aL, q=self.q)
    
        # Solve Poisson's eq. for the potential from the periodic part of the
        # compensation charge derivative
        v1_g = self.finegd.zeros(dtype=self.dtype)
        self.solve_poisson(v1_g, ghat1_g)
        
        # Store potential from the compensation charge
        self.vghat1_g = v1_g.copy()
        
        # Add derivative of vbar - sign convention in add_derivative method
        c_ai = vbar.dict(zero=True)
        c_ai[a][0] = -1.
        vbar.add_derivative(a, v, v1_g, c_axi=c_ai, q=self.q)

        # Store potential for the evaluation of the energy derivative
        self.v1_g = v1_g.copy()

        # Transfer to coarse grid
        v1_G = self.gd.zeros(dtype=self.dtype)
        self.restrictor.apply(v1_g, v1_G, self.phase_cd)

        # This is a phase factor exp(iq*r) times a lattice periodic function
        return v1_G
  
    def calculate_nonlocal_derivative(self, kpt):
        """Derivate of the non-local PAW potential wrt an atomic displacement.

        Parameters
        ----------
        kpt: KPoint
            k-point of the Bloch function being operated on.

        Remember to generalize this to the case of complex wave-functions !!!!!
        
        """

        assert self.a is not None
        assert self.v is not None
        assert self.dP_aniv is not None
        
        a = self.a
        v = self.v
        
        # Projectors on the atoms
        pt = self.pt
        
        nbands = self.calc.wfs.nvalence/2
        hamiltonian = self.calc.hamiltonian

        # <p_a^i | psi_n >
        P_ni = self.calc.wfs.kpt_u[kpt.k].P_ani[a][:nbands]
        # <dp_av^i | psi_n > - remember the sign convention of the calculated
        # derivatives 
        dP_ni = -1 * self.dP_aniv[a][...,v]

        # Array for the derivative of the non-local part of the PAW potential
        vnl1_nG = self.gd.zeros(n=nbands, dtype=self.dtype)
        
        # Expansion coefficients for the projectors on atom a
        dH_ii = unpack(hamiltonian.dH_asp[a][0])

        # The derivative of the non-local PAW potential has two contributions
        # 1) Sum over projectors
        c_ni = np.dot(dP_ni, dH_ii)
        c_ani = pt.dict(shape=nbands, zero=True)
        c_ani[a] = c_ni
        pt.add(vnl1_nG, c_ani)

        # 2) Sum over derivatives of the projectors
        dc_ni = np.dot(P_ni, dH_ii)
        dc_ani = pt.dict(shape=nbands, zero=True)
        # Take care of sign of derivative in the coefficients
        dc_ani[a] = -1 * dc_ni

        pt.add_derivative(a, v, vnl1_nG, dc_ani)
     
        return vnl1_nG

    def calculate_derivative_old(self):
        """Derivate of the local PAW potential wrt an atomic displacement."""

        assert self.a is not None
        assert self.v is not None

        eps = 1e-5/units.Bohr
        
        a = self.a
        v = self.v
        # Array for the derivative of the local part of the PAW potential
        v1_g = self.finegd.zeros()
        
        # Contributions from compensation charges (ghat) and local potential
        # (vbar)
        ghat = self.ghat
        vbar = self.vbar

        # Atomic displacements in scaled coordinates
        eps_s = eps/self.gd.cell_cv[v,v]
        
        # grid for derivative of compensation charges
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
        vbar.add(v1_g, dict_vbar)

        dict_ghat[a] *= -1
        dict_vbar[a] *= -1
            
        spos_ac[a, v] += 2 * eps_s
        ghat.set_positions(spos_ac)
        ghat.add(ghat1_g, dict_ghat)
        vbar.set_positions(spos_ac)
        vbar.add(v1_g, dict_vbar)

        # Return to initial positions
        spos_ac[a, v] -= eps_s
        ghat.set_positions(spos_ac)
        vbar.set_positions(spos_ac)

        # Convert changes to a derivatives
        d = 2 * eps
        v1_g /= d
        ghat1_g /= d

        # Solve Poisson's eq. for the potential from the compensation charges
        hamiltonian = self.calc.hamiltonian
        ps = hamiltonian.poisson
        vghat1_g = self.finegd.zeros()
        ps.solve(vghat1_g, ghat1_g)

        v1_g += vghat1_g
       
        # Transfer to coarse grid
        v1_G = self.gd.zeros()
        hamiltonian.restrictor.apply(v1_g, v1_G)

        # Store potentials for the evaluation of the energy derivative
        self.v1_g = v1_g.copy()
        self.vghat1_g = vghat1_g.copy()
        
        return v1_G
    
    def calculate_nonlocal_derivative_old(self, kpt):
        """Derivate of the non-local PAW potential wrt an atomic displacement.

        Remember to generalize this to the case of complex wave-functions !!!!!
        
        """

        assert self.a is not None
        assert self.v is not None
        assert self.dP_aniv is not None
        
        a = self.a
        v = self.v
        
        # Projectors on the atoms
        pt = self.pt
        
        nbands = self.calc.wfs.nvalence/2
        hamiltonian = self.calc.hamiltonian

        # <p_a^i | psi_n >
        P_ni = self.calc.wfs.kpt_u[0].P_ani[a][:nbands]
        # <dp_av^i | psi_n > - remember the sign convention of the calculated
        # derivatives 
        dP_ni = -1 * self.dP_aniv[a][...,v]

        # Array for the derivative of the non-local part of the PAW potential
        vnl1_nG = self.gd.zeros(n=nbands)
        
        # Expansion coefficients for the projectors on atom a
        dH_ii = unpack(hamiltonian.dH_asp[a][0])

        # The derivative of the non-local PAW potential has two contributions
        # 1) Sum over projectors
        c_ni = np.dot(dP_ni, dH_ii)
        c_ani = pt.dict(shape=nbands, zero=True)
        c_ani[a] = c_ni
        pt.add(vnl1_nG, c_ani)

        # 2) Sum over derivatives of the projectors
        vnl1_temp_nG = self.gd.zeros(n=nbands)
        dc_ni = np.dot(P_ni, dH_ii)
        dc_ani = pt.dict(shape=nbands, zero=True)
        dc_ani[a] = dc_ni

        # Finite-difference derivative: dp = (p(+eps) - p(-eps)) / 2*eps
        # Atomic displacements in scaled coordinates
        eps = 1e-5/units.Bohr
        eps_s = eps/self.gd.cell_cv[v,v]
        spos_ac = self.calc.atoms.get_scaled_positions()
        
        dc_ani[a] *= -1
        spos_ac[a, v] -= eps_s
        pt.set_positions(spos_ac)
        pt.add(vnl1_temp_nG, dc_ani)

        dc_ani[a] *= -1
        spos_ac[a, v] += 2 * eps_s
        pt.set_positions(spos_ac)
        pt.add(vnl1_temp_nG, dc_ani)
    
        # Return to initial positions
        spos_ac[a, v] -= eps_s
        pt.set_positions(spos_ac)
        
        # Convert change to a derivative
        d = 2 * eps
        vnl1_temp_nG /= d
        vnl1_nG += vnl1_temp_nG
        
        return vnl1_nG
