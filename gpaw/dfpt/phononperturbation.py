"""This module implements a phonon perturbation."""

__all__ = ["PhononPerturbation"]

import numpy as np

import ase.units as units

from gpaw.utilities import unpack, unpack2

from gpaw.dfpt.perturbation import Perturbation

class PhononPerturbation(Perturbation):
    """Implementation of phonon-related terms for the Sternheimer equation."""
    
    def __init__(self, calc, eps = 1e-5/units.Bohr):
        """Store useful objects, e.g. lfc for the various atomic functions.

        Parameters
        ----------
        eps: float
            Displacement (in Bohr) of atoms for finite-difference evaluations
            of the derivatives of localized functions

        """

        self.calc = calc
        self.eps = eps
        
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
        self.q = None

        # Coefficients needed for the non-local part of the perturbation
        self.dP_aniv = None
        self.P_ani = None
        
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
        """Set the q-vector of the perturbation."""

        self.q = q
        
    def calculate_derivative(self):
        """Derivate of the local PAW potential wrt an atomic displacement."""

        assert self.a is not None
        assert self.v is not None
        
        a = self.a
        v = self.v
        # Array for the derivative of the local part of the PAW potential
        v1_g = self.finegd.zeros()
        
        # Contributions from compensation charges (ghat) and local potential
        # (vbar)
        ghat = self.ghat
        vbar = self.vbar

        # Atomic displacements in scaled coordinates
        eps_s = self.eps/self.gd.cell_cv[v,v]
        
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
        d = 2 * self.eps
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

    def calculate_dP_aniv(self, kpt):
        """Coefficients for the derivative of the non-local part of the PP.

        Parameters
        ----------
        kpt: k-point
            K-point of the Bloch state on which the non-local potential acts
            upon

        The calculated coefficients are the following:
        
                     /                a*
          dP_aniv =  | dG Psi (G) dPhi  (G) ,
                     /       n        iv

        where
                       
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
    
    def calculate_nonlocal_derivative(self, kpt):
        """Derivate of the non-local PAW potential wrt an atomic displacement."""

        a = self.a
        v = self.v

        self.calculate_dP_aniv(kpt)
        
        # Projectors on the atoms
        pt = self.pt
        
        nbands = self.calc.wfs.nvalence/2
        hamiltonian = self.calc.hamiltonian

        # <p_a^i | psi_n >
        P_ni = self.calc.wfs.kpt_u[0].P_ani[a][:nbands]
        # <dp_av^i | psi_n >
        dP_ni = self.dP_aniv[a][...,v]

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
        eps_s = self.eps/self.gd.cell_cv[v,v]
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
        d = 2 * self.eps
        vnl1_temp_nG /= d
        
        vnl1_nG += vnl1_temp_nG
        
        return vnl1_nG
