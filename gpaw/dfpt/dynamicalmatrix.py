"""This module provides a class for assembling the dynamical matrix."""

__all__ = ["DynamicalMatrix"]

from math import sqrt

import numpy as np

import ase.units as units
from gpaw.utilities import unpack, unpack2

class DynamicalMatrix:
    """Class for assembling the dynamical matrix from first-order responses.

    The second order derivative of the total energy with respect to atomic
    displacements (for periodic systems collective atomic displacemnts
    characterized by a q-vector) can be obtained from an expression involving
    the first-order derivatives of the density and the wave-functions.
    
    Each of the various contributions to the second order derivative of the
    total energy are implemented in separate functions.
    
    """
    
    def __init__(self, atoms, ibzq_qc=None, dtype=float):
        """Inititialize class with a list of atoms."""

        # Store useful objects
        self.atoms = atoms
        self.dtype = dtype
        self.calc = atoms.get_calculator()
        self.masses = atoms.get_masses()
        self.N = atoms.get_number_of_atoms()

        if ibzq_qc = None:
            ibzq_qc = [(0, 0, 0)]
            assert dtype == float
            
        # Matrix of force constants -- dict of dicts in atomic indices
        self.C_aavv = dict([(atom.index,
                               dict([(atom_.index, np.zeros((3,3), dtype=dtype))
                                     for atom_ in atoms])) for atom in atoms])
        
        # Dynamical matrix -- 3Nx3N ndarray (vs q)
        self.D_u = []
        self.D = None
        
    def assemble(self, acoustic=False):
        """Assemble dynamical matrix from the force constants attribute ``C``.

        The elements of the dynamical matrix are given by::
        
            D_ij = 1/(M_i + M_j) * C_ij ,

        where i and j are collective atomic and cartesian indices.

        Parameters
        ----------
        acoustic: bool
            When True, the diagonal of the matrix of force constants is
            corrected to obey the acoustic sum-rule.
            
        """

        # Dynamical matrix - need a dtype here
        D = np.zeros((3*self.N, 3*self.N))

        for atom in self.atoms:

            a = atom.index
            m_a = self.masses[a]

            for atom_ in self.atoms:

                a_ = atom_.index
                m_a_ = self.masses[a_]

                # Mass prefactor
                c = (m_a * m_a_)**(-.5)

                #if a != a_:
                    
                D[3*a : 3*a + 3, 3*a_ : 3*a_ + 3] += c * self.C_aavv[a][a_]
                
                    # Acoustic sum-rule
                    # D[3*a : 3*a + 3, 3*a : 3*a + 3] -= 1/m_a * self.C_aavv[a][a_]
                
        # Symmetrize
        self.D_ = D.copy()
        D *= 0.5
        self.D = D + D.T
        
        return self.D, self.D_

    def update_row(self, a, v, nt1_G, psit1_nG, vghat1_g, dP_aniv):
        """Update row of force constant matrix.

        Parameters
        ----------
        a: int
            Atomic index.
        v: int
            Cartesian index.
        nt1_G: ndarray
            First-order density variation.

        """

        self.density_response_local(a, v, nt1_G, vghat1_g)
        self.wfs_variations_nonlocal(a, v, psit1_nG, dP_aniv)
        
    def ground_state_local(self):
        """Contributions involving ground-state quantities only.

        These terms contains second-order derivaties of the localized functions
        ghat and vbar. They are therefore diagonal in the atomic indices.

        """

        # Localized functions from the local part of the PAW potential
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = self.calc.density.Q_aL
        
        # Integral of Hartree potential times the second derivative of ghat
        vH_g = self.calc.hamiltonian.vHt_g
        d2ghat_aLvv = dict([ (atom.index, np.zeros((3,3)))
                             for atom in self.atoms ])
        ghat.second_derivative(vH_g, d2ghat_aLvv)
 
        # Integral of electron density times the second derivative of vbar
        nt_g = self.calc.density.nt_g
        d2vbar_avv = dict([(atom.index, np.zeros((3,3)))
                           for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        for atom in self.atoms:

            a = atom.index

            # NOTICE: HGH has only one ghat pr atoms -> generalize when
            # implementing PAW            
            self.C_aavv[a][a] += d2ghat_aLvv[a] * Q_aL[a]
            self.C_aavv[a][a] += d2vbar_avv[a]

    def ground_state_nonlocal(self, dP_aniv):
        """Ground state contributions from the non-local potential."""

        # K-point
        kpt = self.calc.wfs.kpt_u[0]
        # Number of occupied bands
        nbands = self.calc.wfs.nvalence/2
        
        # Projector functions
        pt = self.calc.wfs.pt
        # Projector coefficients
        dH_asp = self.calc.hamiltonian.dH_asp
        # Overlap between wave-functions and projectors (NOTE: here n > nbands)
        P_ani = kpt.P_ani
        # Calculate d2P_anivv coefficients
        # d2P_anivv = self.calculate_d2P_anivv()
        d2P_anivv = dict([(atom.index,
                           np.zeros((nbands, pt.get_function_count(atom.index),
                                     3, 3))) for atom in self.atoms])
        # Temp dict, second_derivative method only takes a_G array -- no extra dims
        d2P_avv = dict([(atom.index, np.zeros((3, 3)))
                        for atom in self.atoms])
        psit_nG = self.calc.wfs.kpt_u[0].psit_nG[:nbands]
     
        for n in range(nbands):

            pt.second_derivative(psit_nG[n], d2P_avv)
            # Insert in other dict
            for atom in self.atoms:
                a = atom.index
                d2P_anivv[a][n, 0] = d2P_avv[a]
        
        for atom in self.atoms:

            a = atom.index

            dH_ii = unpack(dH_asp[a][0])
            P_ni = P_ani[a][:nbands]
            dP_niv = -1 * dP_aniv[a]
            d2P_nivv = d2P_anivv[a]
            
            # Term with second-order derivative of projector
            dHP_ni = np.dot(P_ni, dH_ii)
            d2PdHP_nvv = (d2P_nivv * dHP_ni[:, :, np.newaxis, np.newaxis]).sum(1)
            d2PdHP_nvv *= kpt.f_n[:nbands, np.newaxis, np.newaxis]
            A_vv = d2PdHP_nvv.sum(0)

            # Term with first-order derivative of the projectors
            dHdP_inv = np.dot(dH_ii, dP_niv)
            dHdP_niv = np.swapaxes(dHdP_inv, 0, 1)
            dHdP_niv *= kpt.f_n[:nbands, np.newaxis, np.newaxis]

            B_vv = (dP_niv[:, :, np.newaxis, :] * 
                    dHdP_niv[:, :, :, np.newaxis]).sum(0).sum(0)
            
            # B_vv = np.zeros((3,3))
            # for v1 in range(3):
            #     for v2 in range(3):
            #         B_vv[v1,v2] = (dP_niv[...,v2] * dHdP_niv[...,v1]).sum()
            
            self.C_aavv[a][a] += 2 * (A_vv + B_vv)

    def density_response_local(self, a, v, nt1_G, vghat1_g):
        """Contributions involving the first-order density response."""

        # Localized functions from the local part of the PAW potential
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = self.calc.density.Q_aL

        # Integral of density/ghat variation times variation in ghat potential
        dghat_aLv = ghat.dict(derivative=True)
        # Integral of density variation times the variation in vbar potential
        dvbar_av = vbar.dict(derivative=True)
        
        # Transfer density variation to the fine grid
        density = self.calc.density
        nt1_g = density.finegd.zeros()  
        density.interpolator.apply(nt1_G, nt1_g)
        # Calculate corresponding variation in the Hartree potential
        poisson = self.calc.hamiltonian.poisson
        v1_g = density.finegd.zeros()
        poisson.solve(v1_g, nt1_g)
        # Add variation in the compensation charge potential
        v1_g += vghat1_g

        ghat.derivative(v1_g, dghat_aLv)
        vbar.derivative(nt1_g, dvbar_av)        

        # Add to force constant matrix attribute
        for atom_ in self.atoms:
            
            a_ = atom_.index
            # Minus sign below - see doc string to the lfc method
            # derivative
            self.C_aavv[a][a_][v] -= np.dot(Q_aL[a_], dghat_aLv[a_]) 
            self.C_aavv[a][a_][v] -= dvbar_av[a_][0]

    def wfs_variations_nonlocal(self, a, v, psit1_nG, dP_aniv):
        """Contributions from the non-local part of the PAW potential.

        These contributions involve overlaps between wave-functions and their
        variations and the projectors in their variations.

        In contrast to ghat and vbar, the projector functions are defined on
        the coarse grid.
        
        Generalize to sum over k-points when time comes.
        
        """

        # K-point
        kpt = self.calc.wfs.kpt_u[0]
        # Number of occupied bands
        nbands = self.calc.wfs.nvalence/2
        assert len(psit1_nG) == nbands
        
        # Projector functions
        pt = self.calc.wfs.pt
        # Projector coefficients
        dH_asp = self.calc.hamiltonian.dH_asp
        # Overlap between wave-functions and projectors (NOTE: here n > nbands)
        P_ani = kpt.P_ani
        # Overlap between wave-function variations and projectors
        Pdpsi_ani = pt.dict(shape=nbands, zero=True)
        pt.integrate(psit1_nG, Pdpsi_ani)
        # Overlap between wave-function variations and derivative of projectors
        dPdpsi_aniv = pt.dict(shape=nbands, derivative=True)
        pt.derivative(psit1_nG, dPdpsi_aniv)

        for atom_ in self.atoms:

            a_ = atom_.index

            dH_ii = unpack(dH_asp[a_][0])
            P_ni = P_ani[a_][:nbands]
            dP_niv = -1 * dP_aniv[a_]
            Pdpsi_ni = Pdpsi_ani[a_]
            dPdpsi_niv = -1 * dPdpsi_aniv[a_]
            
            # Term with dPdpsi and P coefficients
            dHP_ni = np.dot(P_ni, dH_ii)
            dPdpsidHP_nv = (dPdpsi_niv * dHP_ni[:, :, np.newaxis]).sum(1)
            dPdpsidHP_nv *= kpt.f_n[:nbands, np.newaxis]
            A_v = dPdpsidHP_nv.sum(0)

            # Term with dP and Pdpsi coefficients
            dHPdpsi_ni = np.dot(Pdpsi_ni, dH_ii)
            dPdHPdpsi_nv = (dP_niv * dHPdpsi_ni[:, :, np.newaxis]).sum(1)
            dPdHPdpsi_nv *= kpt.f_n[:nbands, np.newaxis]
            B_v = dPdHPdpsi_nv.sum(0)

            self.C_aavv[a][a_][v] += 2 * (A_v + B_v)




##     def calculate_d2P_anivv(self, kpt=None, eps=1e-5/units.Bohr):
##         """Coefficients for the second derivative of total energy.

##         Parameters
##         ----------
##         kpt: k-point
##             K-point of the Bloch state on which the non-local potential acts
##             upon

##         The calculated coefficients are the following::
        
##                       /         a*           
##           d2P_aniv =  | dG d2Phi  (G) Psi (G) ,
##                       /         iv       n

##         where::
        
##                          2
##                a        d       a
##           d2Phi  (G) =  ---  Phi (G) .
##                iv         2     i
##                         dR
##                           a
        
##         """

##         atoms = self.calc.get_atoms()
##         # Projectors on the atoms
##         pt = self.calc.wfs.pt
##         nbands = self.calc.wfs.nvalence/2
##         # Wave functions
##         psit_nG = self.calc.wfs.kpt_u[0].psit_nG[:nbands]
##         # Integration dict
##         d2P_anivv = dict([(
##             atom.index, np.zeros((nbands, pt.get_function_count(atom.index),
##                                   3, 3))) for atom in atoms ])

##         gd = self.calc.density.gd
        
##         # Temp solution - finite difference of diagonal derivatives only
##         for atom in atoms:
##             a = atom.index
##             c_ai = pt.dict(zero=True)
##             # For now, only one projector pr atom
##             c_ai[a][0]= 1.
            
##             for v in [0,1,2]:

##                 d2P_G = gd.zeros()
##                 # Atomic displacements in scaled coordinates
##                 eps_s = eps/gd.cell_cv[v,v]
##                 spos_ac = self.calc.atoms.get_scaled_positions()
##                 #
##                 c_ai[a] *= -2
##                 pt.add(d2P_G, c_ai)
##                 # -
##                 c_ai[a] *= -.5
##                 spos_ac[a, v] -= eps_s
##                 pt.set_positions(spos_ac)
##                 pt.add(d2P_G, c_ai)
##                 # +
##                 spos_ac[a, v] += 2 * eps_s
##                 pt.set_positions(spos_ac)
##                 pt.add(d2P_G, c_ai)
##                 # Return to initial positions
##                 spos_ac[a, v] -= eps_s
##                 pt.set_positions(spos_ac)
                
##                 # Convert change to a derivative
##                 d = eps**2
##                 d2P_G /= d

##                 int_n = gd.integrate(d2P_G * psit_nG)

##                 d2P_anivv[a][:,0,v,v] = int_n


##         return d2P_anivv
