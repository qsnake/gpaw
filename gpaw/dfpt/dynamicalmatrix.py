"""This module provides a class for assembling the dynamical matrix."""

__all__ = ["DynamicalMatrix"]

from math import sqrt

import numpy as np

import ase.units as units

from gpaw.utilities import unpack, unpack2


class DynamicalMatrix:
    """This class is used to assemble the dynamical matrix.

    Each of the various contributions to the second derivative of the total
    energy are implemented in separate functions.
    
    """
    
    def __init__(self, atoms):
        """Inititialize class with a list of atoms."""

        # Store useful objects
        self.atoms = atoms
        self.calc = atoms.get_calculator()
        self.masses = atoms.get_masses()
        self.N = atoms.get_number_of_atoms()
        
        # Matrix of force constants -- dict of dicts in atomic indices
        self.C = dict([(atom.index,
                 dict([(atom_.index,np.zeros((3,3)))
                       for atom_ in atoms])) for atom in atoms])
        # Dynamical matrix -- 3Nx3N ndarray
        self.D = None

    def assemble(self):
        """Assemble dynamical matrix from the force constant attribute ``C``.

        D_ij = 1/(M_i + M_j) * C_ij

        """

        # Dynamical matrix
        D = np.zeros((3 * N_atoms, 3 * N_atoms))

        for atom in self.atoms:

            a = atom.index
            m_a = self.masses[a]

            for atom_ in self.atoms:

                a_ = atom_.index
                m_a_ = self.masses[a_]
                
                # Mass prefactor
                c = (m_a * m_a_)**(-.5)
                D[3*a:3*a+3, 3*a_: 3*a_+3] += c * self.C[a][a_]
                
        # Symmetrize the dynamical matrix
        D *= 0.5
        self.D = D + D.T

        return self.D
        
    def contrib_1(self):
        """Contributions from local ground-state potentials.

        Only diagonal contributions.

        """

        # Localized functions from the local part of the PAW potential
        ghat = self.calc.density.ghat
        vbar = self.calc.hamiltonian.vbar
        # Compensation charge coefficients
        Q_aL = self.calc.density.Q_aL
        
        # Integrate Hartree potential with second derivative of ghat
        vH_g = self.calc.hamiltonian.vHt_g
        d2ghat_aLvv = dict([ (atom.index, np.zeros((3,3)))
                             for atom in self.atoms ])
        ghat.second_derivative(vH_g, d2ghat_aLvv)
 
        # Integrate electron density with second derivative of vbar
        nt_g = self.calc.density.nt_g
        d2vbar_avv = dict([(atom.index, np.zeros((3,3)))
                           for atom in self.atoms ])
        vbar.second_derivative(nt_g, d2vbar_avv)

        for atom in self.atoms:

            a = atom.index

            # NOTICE: HGH has only one ghat pr atoms -> generalize when
            # implementing PAW            
            C[3*a:3*a+3, 3*a:3*a+3] += d2ghat_aLvv[a] * Q_aL[a]
            C[3*a:3*a+3, 3*a:3*a+3] += d2vbar_avv[a]
        
    def contrib_2(self):
        """Contributions from density derivative and local potentials."""

        dghat_aaLvv = dict([(atom.index, dict([(atom_.index,
                          np.zeros((ghat.get_function_count(atom.index), 3, 3)))
                          for atom_ in self.atoms ])) for atom in self.atoms] )
        # Integrate first-order density variation with vbar derivative
        dvbar_aavv = dict( [ (atom.index, dict( [ (atom_.index, np.zeros((3,3)))
                           for atom_ in self.atoms ] )) for atom in self.atoms])        
