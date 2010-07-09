"""This module provides an interface class for phonon calculations."""

__all__ = ["PhononCalculator"]

from math import sqrt

import pickle
import numpy as np

import ase.units as units

# Temp modules
from gpaw.utilities import unpack, unpack2

from gpaw.dfpt.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.dfpt.kpointdescriptor import KPointDescriptor
from gpaw.dfpt.responsecalculator import ResponseCalculator
from gpaw.dfpt.phononperturbation import PhononPerturbation
from gpaw.dfpt.wavefunctions import WaveFunctions
from gpaw.dfpt.dynamicalmatrix import DynamicalMatrix


class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    
    def __init__(self, calc, gamma=True, **kwargs):
        """Inititialize class with a list of atoms.

        The atoms object must contain a converged ground-state calculation.

        For now the q-points are taken from the ground-state calculation.

        """
 
        # Store useful objects
        self.atoms = calc.get_atoms()
        self.calc = calc
        
        # Make sure localized functions are initialized
        calc.set_positions()
        # Note that this under some circumstances (e.g. when called twice)
        # allocates a new array for the P_ani coefficients !!

        # Boundary conditions
        pbc_c = calc.atoms.get_pbc()
        
        if np.all(pbc_c == False):
            self.gamma = True
            self.dtype = float
            self.ibzq_qc = None # np.array((0, 0, 0), dtype=float)
            # Multigrid Poisson solver
            poisson = PoissonSolver()
        else:
            if gamma:
                self.gamma = True
                self.dtype = float
                self.ibzq_qc = None #np.array(((0, 0, 0),), dtype=float)
            else:
                self.gamma = False
                self.dtype = complex
                # Get k-points -- only temp, I need q-vectors; maybe the same ???
                self.ibzq_qc = calc.get_ibz_k_points()
                phase_qcd = [kpt.phase_cd for kpt in calc.wfs.kpt_u]
                
            # FFT Poisson solver
            poisson = FFTPoissonSolver(dtype=self.dtype)
            
        # Include all atoms and cartesian coordinates by default
        self.atoms_a = dict([ (atom.index, [0, 1, 2]) for atom in self.atoms])
        
        # Phonon perturbation
        self.perturbation = PhononPerturbation(calc, self.gamma,
                                               ibzq_qc=self.ibzq_qc)
                                               #, poisson_solver=poisson)

        # Ground-state k-points
        ibzk_qc = calc.get_ibz_k_points()
        bzk_qc = calc.get_ibz_k_points()
        kd = KPointDescriptor(bzk_qc, ibzk_qc)

        # Number of occupied bands
        nvalence = calc.wfs.nvalence
        nbands = nvalence/2 + nvalence%2
        assert nbands <= calc.wfs.nbands

        # WaveFunctions object
        wfs = WaveFunctions(nbands, calc.wfs.kpt_u, calc.wfs.setups,
                            calc.wfs.gamma, kd, calc.density.gd)
        
        # Linear response calculator
        self.response_calc = ResponseCalculator(calc, wfs, self.perturbation, kd)

        # Dynamical matrix object
        self.D_matrix = DynamicalMatrix(self.atoms, ibzq_qc=self.ibzq_qc,
                                        dtype=self.dtype)

        # Initialize flag
        self.initialized = False
        
    def initialize(self):
        """Initialize response calculator and perturbation."""

        # Get scaled atomic positions
        spos_ac = self.atoms.get_scaled_positions()

        self.perturbation.initialize(spos_ac)
        self.response_calc.initialize(spos_ac)

        self.initialized = True
        
    def set_atoms(self, atoms_a):
        """Set indices of atoms to include in the calculation."""

        if isinstance(atoms_a, dict):
            self.atoms_a.update(atoms_a)
        else:
            # List of atoms indices
            for a in atoms_a:
                self.atoms_a = [0, 1, 2]

    def __call__(self):
        """Run calculation for atomic displacements and update matrix."""

        if not self.initialized:
            self.initialize()
       
        if self.gamma:
            i_q = [0]
        else:
            i_q = range(len(self.ibzq_qc))
                        
        # Calculate linear response wrt displacements of specified atoms
        for q in i_q:

            if not self.gamma:
                self.perturbation.set_q(q)

            for a in self.atoms_a:
    
                for v in self.atoms_a[a]:
    
                    components = ['x','y','z']
                    symbols = self.atoms.get_chemical_symbols()
                    print "q-vector index: %i" % q
                    print "Atom index: %i" % a
                    print "Atomic symbol: %s" % symbols[a]
                    print "Component: %s" % components[v]
                    
                    self.perturbation.set_perturbation(a, v)
        
                    output = self.response_calc()
    
                    if False: #save:
                        assert filebase is not None
                        file_av = "a_%.1i_v_%.1i.pckl" % (a, v)
                        fname = "_".join([filebase, file_av])
                        f = open(fname, 'w')
                        pickle.dump([nt1_G, psit1_unG], f)
                        f.close()
                                
                    self.D_matrix.update_row(self.perturbation,
                                             self.response_calc)
                    
        self.D_matrix.density_ground_state(self.calc)
        self.D_matrix.wfs_ground_state(self.calc, self.response_calc)

    def get_dynamical_matrix(self):
        """Assemble and return the dynamical matrix as an ndarray."""
        
        return self.D_matrix.assemble()
        
    def frequencies(self):
        """Calculate phonon frequencies from the dynamical matrix."""

        # In Ha/(Bohr^2 * amu)
        D, D_ = self.D_matrix.assemble()
        
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
