"""This module implements a phonon perturbation."""

__all__ = ["PhononPerturbation"]

from math import sqrt, pi

import numpy as np

import ase.units as units

from gpaw.utilities import unpack, unpack2
from gpaw.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.lfc import LocalizedFunctionsCollection as LFC

from gpaw.dfpt.perturbation import Perturbation


class PhononPerturbation(Perturbation):
    """Implementation of phonon-related terms for the Sternheimer equation."""
    
    def __init__(self, calc):
        """Store useful objects, e.g. lfc's for the various atomic functions.

        Parameters
        ----------
        calc: calculator
        
        Depending on whether the system is periodic or not, Poisson's equation
        is solved with FFT and multigrid techniques, respectively.
       
        """

        self.calc = calc
        # Boundary conditions and Poisson solver
        pbc = calc.atoms.get_pbc()
        if np.all(pbc == False):
            self.gamma = True
            self.dtype = float
            self.poisson = PoissonSolver()
        else:
            self.gamma = False
            self.dtype = complex
            self.poisson = FFTPoissonSolver(dtype=complex)

        # Use existing ghat and vbar instances -- will be replaced by periodic
        # versions in the ``initialize`` method in case of periodic BC's
        # Compensation charge functions
        self.ghat = calc.density.ghat
        # Local potential
        self.vbar = calc.hamiltonian.vbar

        # Projectors on the atoms
        self.pt = self.calc.wfs.pt
        
        # Store grid-descriptors
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd

        # Atom, cartesian coordinate and q-vector of the perturbation
        self.a = None
        self.v = None
        if self.gamma:
            self.q = -1
        else:
            self.q = None

        # Coefficients needed for the non-local part of the perturbation
        self.P_ani = None
        self.dP_aniv = None

    def initialize(self):
        """Prepare the various attributes for a calculation."""

        # Replace ghat and vbar attributes with periodic versions 
        if not self.gamma:
            # Get setups
            setups = self.calc.hamiltonian.setups
            # Get k-points -- only temp, I need q-vectors; maybe the same ???
            ibzk_qc = self.calc.get_ibz_k_points()
            # Get scaled atomic positions
            spos_ac = self.calc.atoms.get_scaled_positions()
            
            # Replace ghat attribute with Bloch periodic version
            self.ghat = LFC(self.finegd,
                            [setup.ghat_l for setup in setups],
                            integral=sqrt(4 * pi))
            # Set q-vectors and update 
            self.ghat.set_k_points(ibzk_qc)
            self.ghat.set_positions(spos_ac)

            # Replace vbar attribute with Bloch periodic version
            self.vbar = LFC(self.finegd,
                            [[setup.vbar] for setup in setups])
            # Set q-vectors and update 
            self.vbar.set_k_points(ibzk_qc)
            self.vbar.set_positions(spos_ac)
        
        # Setup Poisson solver
        self.poisson.set_grid_descriptor(self.finegd)
        self.poisson.initialize()
        # Calculate coefficients needed for the non-local part of the PP
        self.calculate_dP_aniv()
        
    def set_perturbation(self, a, v):
        """Set atom and cartesian coordinate of the perturbation.

        Parameters
        ----------
        a: int
            Index of the atom
        v: int 
            Cartesian component (0, 1 or 2) of the atomic displacement
            
        """

        self.a = a
        self.v = v
        # Invalidate calculated quantities
        # - local part of pseudo-potential and the like

    def set_q(self, q):
        """Set the index of the q-vector of the perturbation."""

        self.q = q
        
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
        """Derivate of the local PAW potential wrt an atomic displacement.

        The local part of the PAW potential has contributions from the
        compensation charges (ghat) and a spherical symmetric atomic potential
        (vbar).
        
        """

        assert self.a is not None
        assert self.v is not None
        assert self.q is not None
        
        a = self.a
        v = self.v
        q = self.q
        
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
        # Solve Poisson's eq. for the potential from the compensation charges
        hamiltonian = self.calc.hamiltonian
        # ps = hamiltonian.poisson
        v1_g = self.finegd.zeros(dtype=self.dtype)
        #self.poisson.solve(v1_g, ghat1_g)
        #return v1_g
        # Store potential from the compensation charge
        self.vghat1_g = v1_g.copy()
        
        # Add derivative of vbar - sign convention in add_derivative method
        c_ai = vbar.dict(zero=True)
        c_ai[a][0] = -1
        vbar.add_derivative(a, v, v1_g, c_axi=c_ai, q=self.q)
        return v1_g
        # Transfer to coarse grid
        v1_G = self.gd.zeros(dtype=self.dtype)
        hamiltonian.restrictor.apply(v1_g, v1_G)

        # Store potential for the evaluation of the energy derivative
        self.v1_g = v1_g.copy()
        
        return v1_G
  
    def calculate_nonlocal_derivative(self, kpt):
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
