"""This module provides an interface class for phonon calculations."""

__all__ = ["PhononCalculator"]

from math import sqrt

import pickle
import numpy as np

import ase.units as units

from gpaw.utilities import unpack, unpack2

from gpaw.dfpt.linearresponse import LinearResponse
from gpaw.dfpt.phononperturbation import PhononPerturbation
from gpaw.dfpt.dynamicalmatrix import DynamicalMatrix


class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    
    def __init__(self, atoms):
        """Inititialize class with a list of atoms."""
 
        # Store useful objects
        self.atoms = atoms
        self.calc = atoms.get_calculator()
        # Make sure localized functions are initialized
        self.calc.set_positions()
        # Note that this under some circumstances (e.g. when called twice)
        # allocates a new array for the P_ani coefficients !!

        self.atoms_a = [atom.index for atom in atoms]
        
        # Phonon perturbation
        self.perturbation = PhononPerturbation(self.calc)
        # Linear response calculator
        self.response = LinearResponse(self.calc, self.perturbation)

        # Dynamical matrix object
        self.D_matrix = DynamicalMatrix(atoms)
        self.D = None

    def set_atoms(self, atoms_a):
        """Set indices of atoms to include in the calculation."""

        self.atoms_a = atoms_a
        
    def __call__(self, tolerance_sc = 1e-5,
                 tolerance_sternheimer = 1e-5, use_dfpt=True,
                 save=False, load=False, filebase=None):
        """Run calculation for atomic displacements and update matrix."""

        dP_aniv = self.perturbation.dP_aniv
        
        # Calculate linear response wrt displacements of all atoms
        # for atom in self.atoms:
        for a in self.atoms_a:
            
            for v in [0,1,2]:

                self.perturbation.set_perturbation(a, v)

                if use_dfpt:

                    if load:
                        assert filebase is not None
                        file_av = "a_%.1i_v_%.1i.pckl" % (a,v)
                        fname = "_".join([filebase, file_av])
                        nt1_G, psit1_unG = pickle.load(open(fname))
                        self.perturbation.calculate_derivative()
                    else:
                        nt1_G, psit1_unG = \
                               self.response(tolerance_sc = tolerance_sc,
                                             tolerance_sternheimer = tolerance_sternheimer)
                        if save:
                            assert filebase is not None
                            file_av = "a_%.1i_v_%.1i.pckl" % (a,v)
                            fname = "_".join([filebase, file_av])
                            pickle.dump([nt1_G, psit1_unG], open(fname,'w'))
                else: # load from file
                    assert filebase is not None

                    file_av = "a_%.1i_v_%.1i.pckl" % (a,v)
                    fname = "_".join([filebase, file_av])
                    dn, dpsi_n = pickle.load(open(fname))
                    
                    # Project out components from occupied orbitals
                    self.response.initialize()
                    s = self.response.sternheimer_operator
                    for n in range(len(dpsi_n)):
                        s.project(dpsi_n[n])
                    nt1_G = dn.copy()
                    psit1_unG = [dpsi_n.copy()]
                    # Call this function to calculate derivatives of Vghat to be
                    # used below (only when dn is loaded from file!!)
                    self.perturbation.calculate_derivative()

                vghat1_g = self.perturbation.vghat1_g
                
                self.D_matrix.update_row(a, v, nt1_G, psit1_unG[0],
                                         vghat1_g, dP_aniv)
                
        self.D_matrix.ground_state_local()
        self.D_matrix.ground_state_nonlocal(dP_aniv)

        return self.D_matrix.assemble()
        
    def frequencies(self):
        """Calculate phonon frequencies from the dynamical matrix."""

        if self.D is None:
            print "Calculating dynamical matrix with default parameters."
            self.dynamical_matrix()

        # In Ha/(Bohr^2 * amu)
        D = self.D
        freq2, modes = np.linalg.eigh(D)
        freq = np.sqrt(freq2)
        # Convert to eV
        freq *= sqrt(units.Hartree) / (units.Bohr * 1e-10) / \
                sqrt(units._amu * units._e) * units._hbar

        print "Calculated frequencies (meV):"
        for f in freq:
            print f*1000

        return freq

    def forces(self):
        """Check for the forces."""

        N_atoms = self.atoms.get_number_of_atoms()
        nbands = self.calc.wfs.nvalence/2        
        Q_aL = self.calc.density.Q_aL
        # Forces
        F = np.zeros((N_atoms, 3))
        F_1 = np.zeros((N_atoms, 3))
        F_2 = np.zeros((N_atoms, 3))
        F_3 = np.zeros((N_atoms, 3))
        
        # Check the force
        ghat = self.calc.density.ghat
        vH_g = self.calc.hamiltonian.vHt_g
        F_ghat_aLv = ghat.dict(derivative=True)
        ghat.derivative(vH_g,F_ghat_aLv)
        
        vbar = self.calc.hamiltonian.vbar
        nt_g = self.calc.density.nt_g
        F_vbar_av = vbar.dict(derivative=True)
        vbar.derivative(nt_g,F_vbar_av)

        pt = self.calc.wfs.pt
        psit_nG = self.calc.wfs.kpt_u[0].psit_nG[:nbands]        
        P_ani = pt.dict(nbands)
        pt.integrate(psit_nG, P_ani)        
        dP_aniv = pt.dict(nbands, derivative=True)
        pt.derivative(psit_nG, dP_aniv)
        dH_asp = self.calc.hamiltonian.dH_asp

        # P_ani = self.calc.wfs.kpt_u[0].P_ani
        kpt = self.calc.wfs.kpt_u[0]
        
        for atom in self.atoms:

            a = atom.index
            # ghat and vbar contributions
            F_1[a] += F_ghat_aLv[a][0] * Q_aL[a][0]
            F_2[a] += F_vbar_av[a][0]
            # Contribution from projectors
            dH_ii = unpack(dH_asp[a][0])
            P_ni = P_ani[a][:nbands]
            dP_niv = dP_aniv[a]
            dHP_ni = np.dot(P_ni, dH_ii)
            # Factor 2 for spin
            dHP_ni *= kpt.f_n[:nbands, np.newaxis]

            F_3[a] += 2 * (dP_niv * dHP_ni[:, :, np.newaxis]).sum(0).sum(0)
            #print F_3[a]
            F[a] += F_1[a] + F_2[a] + F_3[a]
            
        # Convert to eV/Ang            
        F *= units.Hartree/units.Bohr
        F_1 *= units.Hartree/units.Bohr
        F_2 *= units.Hartree/units.Bohr
        F_3 *= units.Hartree/units.Bohr
        
        return F, F_1, F_2, F_3
