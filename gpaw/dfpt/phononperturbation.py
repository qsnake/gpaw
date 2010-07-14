"""This module implements a phonon perturbation."""

__all__ = ["PhononPerturbation"]

from math import sqrt, pi

import numpy as np
import numpy.linalg as la

from gpaw.utilities import unpack, unpack2
from gpaw.transformers import Transformer
from gpaw.lfc import LocalizedFunctionsCollection as LFC

# To be removed
from gpaw.dfpt.poisson import PoissonSolver, FFTPoissonSolver

from gpaw.dfpt.perturbation import Perturbation
from gpaw.dfpt.kpointcontainer import KPointContainer


class PhononPerturbation(Perturbation):
    """Implementation of a phonon perturbation.

    This class implements the change in the effective potential due to a
    displacement of an atom ``a`` in direction ``v`` with wave-vector ``q``.
    The action of the perturbing potential on a state vector is implemented in
    the ``apply`` member function.
    
    """
    
    def __init__(self, calc, gamma, ibzq_qc=None, poisson_solver=None, **kwargs):
        """Store useful objects, e.g. lfc's for the various atomic functions.
            
        Depending on whether the system is periodic or finite, Poisson's equation
        is solved with FFT or multigrid techniques, respectively.
       
        """

        self.gamma = gamma
        self.ibzq_qc = ibzq_qc
        self.poisson = poisson_solver
        
        if gamma:
            self.dtype = float
            self.phase_cd = None #np.ones((3, 2), dtype=complex)
        else:
            self.dtype = complex
            self.phase_qcd = [kpt.phase_cd for kpt in calc.wfs.kpt_u]
            
        # dtype for ground-state wave-functions (used for the projectors)
        self.gs_gamma = calc.wfs.gamma
        # k-points for the wave-functions also needed for the projectors
        self.ibzk_qc = calc.get_ibz_k_points()

        # Temp solution - should be given as argument to the init method
        if poisson_solver is None:
            
            # Boundary conditions
            pbc_c = calc.atoms.get_pbc()

            if np.all(pbc_c == False):
                # Multigrid Poisson solver
                self.poisson = PoissonSolver()
            else:
                # FFT Poisson solver
                self.poisson = FFTPoissonSolver(dtype=self.dtype)
      
        # Store grid-descriptors
        self.gd = calc.density.gd
        self.finegd = calc.density.finegd

        # Steal setups for the lfc's
        setups = calc.wfs.setups

        # Store projector coefficients
        self.dH_asp = calc.hamiltonian.dH_asp.copy()
        
        # Localized functions:
        # core corections
        self.nct = LFC(self.gd, [[setup.nct] for setup in setups],
                       integral=[setup.Nct for setup in setups])
        # compensation charges
        self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups])        
        # self.ghat = LFC(self.finegd, [setup.ghat_l for setup in setups],
        #                 integral=sqrt(4 * pi))
        # vbar potential
        self.vbar = LFC(self.finegd, [[setup.vbar] for setup in setups])

        # Expansion coefficients for the compensation charges
        self.Q_aL = calc.density.Q_aL.copy()
        
        # Grid transformer -- convert array from fine to coarse grid
        self.restrictor = Transformer(self.finegd, self.gd, nn=3,
                                      dtype=self.dtype, allocate=False)

        # Atom, cartesian coordinate and q-vector of the perturbation
        self.a = None
        self.v = None
        
        # q-vector of the perturbation
        if self.gamma:
            self.q = -1
        else:
            self.q = None

    def initialize(self, spos_ac):
        """Prepare the various attributes for a calculation."""

        # Set positions on LFC's
        self.nct.set_positions(spos_ac)
        self.ghat.set_positions(spos_ac)
        self.vbar.set_positions(spos_ac)

        if not self.gamma:
            
            # Set q-vectors and update
            self.ghat.set_k_points(self.ibzq_qc)
            self.ghat._update(spos_ac)
            # Set q-vectors and update
            self.vbar.set_k_points(self.ibzq_qc)
            self.vbar._update(spos_ac)

            # Phase factor exp(iq.r) needed to obtian the periodic part of lfcs
            coor_vg = self.finegd.get_grid_point_coordinates()
            cell_cv = self.finegd.cell_cv
            # Convert to scaled coordinates
            scoor_cg = np.dot(la.inv(cell_cv), coor_vg.swapaxes(0, -2))
            scoor_cg = scoor_cg.swapaxes(1,-2)
            # Phase factor
            phase_qg = np.exp(2j * pi *
                              np.dot(self.ibzq_qc, scoor_cg.swapaxes(0,-2)))
            self.phase_qg = phase_qg.swapaxes(1, -2)

        #XXX To be removed from this class !!
        # Setup the Poisson solver -- to be used on the fine grid
        self.poisson.set_grid_descriptor(self.finegd)
        self.poisson.initialize()

        # Grid transformer
        self.restrictor.allocate()

    def get_phase_cd(self):
        """Overwrite base class member function."""

        return self.phase_cd
    
    def has_q(self):
        """Overwrite base class member function."""

        return (not self.gamma)

    def get_q(self):
        """Return q-vector."""

        assert not self.gamma, "Gamma-point calculation."
        
        return self.ibzq_qc[self.q]
    
    def set_perturbation(self, a, v):
        """Set atom and cartesian coordinate of the perturbation.

        Parameters
        ----------
        a: int
            Index of the atom.
        v: int 
            Cartesian component (0, 1 or 2) of the atomic displacement.
            
        """

        assert self.q is not None
        
        self.a = a
        self.v = v

        self.calculate_local_potential()

    def set_q(self, q):
        """Set the index of the q-vector of the perturbation."""

        assert not self.gamma, "Gamma-point calculation"
        
        self.q = q

        # Update phases and Poisson solver
        self.phase_cd = self.phase_qcd[q]
        self.poisson.set_q(self.ibzq_qc[q])

        # Invalidate calculated quantities
        # - local part of perturbing potential        
        self.v1_G = None
        
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

        #assert phi_g.shape == rho_g.shape == self.phase_qg.shape[-3:], \
        #       ("Arrays have incompatible shapes.")
        assert self.q is not None, ("q-vector not set")
        
        # Solve Poisson's eq. for the potential from the periodic part of the
        # compensation charge derivative

        # Gamma point calculation wrt the q-vector
        if self.gamma: 
            # NOTICE: solve_neutral
            self.poisson.solve_neutral(phi_g, rho_g)  
        else:
            # Divide out the phase factor to get the periodic part
            rhot_g = rho_g/self.phase_qg[self.q]

            # Solve Poisson's equation for the periodic part of the potential
            # NOTICE: solve_neutral
            self.poisson.solve_neutral(phi_g, rhot_g)

            # Return to Bloch form
            phi_g *= self.phase_qg[self.q]

    def apply(self, psi_nG, y_nG, wfs, k, kplusq):
        """Apply perturbation to unperturbed wave-functions.

        Parameters
        ----------
        psi_nG: ndarray
            Set of grid vectors to which the perturbation is applied.
        y_nG: ndarray
            Output vectors.
        k: int
            Index of the k-point for the vectors.
        kplusq: int
            Index of the k+q vector.
        calculate_projector_coef: bool
            Use existing coefficients when True.
            
        """

        assert self.a is not None
        assert self.v is not None
        assert self.q is not None
        assert psi_nG.ndim in (3, 4)
        assert tuple(self.gd.n_c) == psi_nG.shape[-3:]

        if psi_nG.ndim == 3:
            y_nG += psi_nG * self.v1_G
        else:
            for x_G, y_G in zip(psi_nG, y_nG):
                y_G += x_G * self.v1_G

        self.apply_nonlocal_potential(psi_nG, y_nG, wfs, k, kplusq)

    def calculate_local_potential(self):
        """Derivate of the local potential wrt an atomic displacements.

        The local part of the PAW potential has contributions from the
        compensation charges (``ghat``) and a spherical symmetric atomic
        potential (``vbar``).
        
        """

        assert self.a is not None
        assert self.v is not None
        assert self.q is not None
        
        a = self.a
        v = self.v
        
        # Expansion coefficients for the ghat functions
        Q_aL = self.ghat.dict(zero=True)
        # Remember sign convention for add_derivative method
        Q_aL[a] = -1 * self.Q_aL[a]

        # Grid for derivative of compensation charges
        ghat1_g = self.finegd.zeros(dtype=self.dtype)
        self.ghat.add_derivative(a, v, ghat1_g, c_axi=Q_aL, q=self.q)
        
        # Solve Poisson's eq. for the potential from the periodic part of the
        # compensation charge derivative
        v1_g = self.finegd.zeros(dtype=self.dtype)
        self.solve_poisson(v1_g, ghat1_g)
        
        # Store potential from the compensation charge
        self.vghat1_g = v1_g.copy()
        
        # Add derivative of vbar - sign convention in add_derivative method
        c_ai = self.vbar.dict(zero=True)
        c_ai[a][0] = -1.
        self.vbar.add_derivative(a, v, v1_g, c_axi=c_ai, q=self.q)

        # Store potential for the evaluation of the energy derivative
        self.v1_g = v1_g.copy()
        # self.v1_g = ghat1_g.copy()
        
        # Transfer to coarse grid
        v1_G = self.gd.zeros(dtype=self.dtype)
        self.restrictor.apply(v1_g, v1_G, phases=self.phase_cd)

        self.v1_G = v1_G

    def apply_nonlocal_potential(self, psi_nG, y_nG, wfs, k, kplusq):
        """Derivate of the non-local PAW potential wrt an atomic displacement.

        Parameters
        ----------
        k: int
            Index of the k-point being operated on.
        kplusq: int
            Index of the k+q vector.
            
        """

        assert self.a is not None
        assert self.v is not None
        assert psi_nG.ndim in (3,4)
        assert tuple(self.gd.n_c) == psi_nG.shape[-3:]
        
        if psi_nG.ndim == 3:
            n = 1
        else:
            n = psi_nG.shape[0]
            
        a = self.a
        v = self.v
        
        P_ani = wfs.kpt_u[k].P_ani
        dP_aniv = wfs.kpt_u[k].dP_aniv
        pt = wfs.pt
        
        # < p_a^i | Psi_nk >
        P_ni = P_ani[a]
        # < dp_av^i | Psi_nk > - remember the sign convention of the derivative
        dP_ni = -1 * dP_aniv[a][...,v]

        # Expansion coefficients for the projectors on atom a
        dH_ii = unpack(self.dH_asp[a][0])
       
        # The derivative of the non-local PAW potential has two contributions
        # 1) Sum over projectors
        c_ni = np.dot(dP_ni, dH_ii)
        c_ani = pt.dict(shape=n, zero=True)
        c_ani[a] = c_ni
        # k+q !!
        pt.add(y_nG, c_ani, q=kplusq)

        # 2) Sum over derivatives of the projectors
        dc_ni = np.dot(P_ni, dH_ii)
        dc_ani = pt.dict(shape=n, zero=True)
        # Take care of sign of derivative in the coefficients
        dc_ani[a] = -1 * dc_ni
        # k+q !!
        pt.add_derivative(a, v, y_nG, dc_ani, q=kplusq)
