"""This module provides an interface class for phonon calculations."""

__all__ = ["PhononCalculator"]

import time
import os
import pickle

import numpy as np

from gpaw import GPAW
from gpaw.mpi import serial_comm, rank
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.dfpt.poisson import PoissonSolver, FFTPoissonSolver
from gpaw.dfpt.responsecalculator import ResponseCalculator
from gpaw.dfpt.phononperturbation import PhononPerturbation
from gpaw.dfpt.wavefunctions import WaveFunctions
from gpaw.dfpt.dynamicalmatrix import DynamicalMatrix
from gpaw.dfpt.electronphononcoupling import ElectronPhononCoupling

class PhononCalculator:
    """This class defines the interface for phonon calculations."""
    
    def __init__(self, calc, gamma=True, symmetry=False, e_ph=False,
                 communicator=serial_comm):
        """Inititialize class with a list of atoms.

        The atoms object must contain a converged ground-state calculation.

        The set of q-vectors in which the dynamical matrix will be calculated
        is determined from the ``symmetry`` kwarg. For now, only time-reversal
        symmetry is used to generate the irrecducible BZ.

        Add a little note on parallelization strategy here.

        Parameters
        ----------
        calc: str or Calculator
            Calculator containing a ground-state calculation.
        gamma: bool
            Gamma-point calculation with respect to the q-vector of the
            dynamical matrix. When ``False``, the Monkhorst-Pack grid from the
            ground-state calculation is used.
        symmetry: bool
            Use symmetries to reduce the q-vectors of the dynamcial matrix
            (None, False or True). The different options are equivalent to the
            options in a ground-state calculation.
        e_ph: bool
            Save the derivative of the effective potential.
        communicator: Communicator
            Communicator for parallelization over k-points and real-space
            domain.
            
        """

        # XXX
        assert symmetry in [None, False], "Spatial symmetries not allowed yet"

        self.symmetry = symmetry

        if isinstance(calc, str):
            self.calc = GPAW(calc, communicator=serial_comm, txt=None)
        else:
            self.calc = calc

        # Make sure localized functions are initialized
        self.calc.set_positions()
        # Note that this under some circumstances (e.g. when called twice)
        # allocates a new array for the P_ani coefficients !!

        # Store useful objects
        self.atoms = self.calc.get_atoms()
        # Get rid of ``calc`` attribute
        self.atoms.calc = None
 
        # Boundary conditions
        pbc_c = self.calc.atoms.get_pbc()

        if np.all(pbc_c == False):
            self.gamma = True
            self.dtype = float
            kpts = None
            # Multigrid Poisson solver
            poisson_solver = PoissonSolver()
        else:
            if gamma:
                self.gamma = True
                self.dtype = float
                kpts = None
            else:
                self.gamma = False
                self.dtype = complex
                # Get k-points from ground-state calculation
                kpts = self.calc.input_parameters.kpts
                
            # FFT Poisson solver
            poisson_solver = FFTPoissonSolver(dtype=self.dtype)

        # K-point descriptor for the q-vectors of the dynamical matrix
        # Note, no explicit parallelization here.
        self.kd = KPointDescriptor(kpts, 1)
        self.kd.set_symmetry(self.atoms, self.calc.wfs.setups, symmetry)
        self.kd.set_communicator(serial_comm)

        # Number of occupied bands
        nvalence = self.calc.wfs.nvalence
        nbands = nvalence/2 + nvalence%2
        assert nbands <= self.calc.wfs.nbands

        # Extract other useful objects
        # Ground-state k-point descriptor - used for the k-points in the
        # ResponseCalculator
        # XXX replace communicators when ready to parallelize
        kd_gs = self.calc.wfs.kd
        gd = self.calc.density.gd
        kpt_u = self.calc.wfs.kpt_u
        setups = self.calc.wfs.setups
        dtype_gs = self.calc.wfs.dtype
        
        # WaveFunctions
        wfs = WaveFunctions(nbands, kpt_u, setups, kd_gs, gd, dtype=dtype_gs)

        # Linear response calculator
        self.response_calc = ResponseCalculator(self.calc, wfs, dtype=self.dtype)
        
        # Phonon perturbation
        self.perturbation = PhononPerturbation(self.calc, self.kd,
                                               poisson_solver,
                                               dtype=self.dtype)

        # Dynamical matrix
        self.D_matrix = DynamicalMatrix(self.atoms, self.kd, dtype=self.dtype)

        # Electron-phonon couplings
        if e_ph:
            self.e_ph = ElectronPhononCoupling(self.atoms, gd, self.kd,
                                               dtype=self.dtype)
        else:
            self.e_ph = None
                                               
        # Initialization flag
        self.initialized = False

        # Parallel communicator for parallelization over kpts and domain
        self.comm = communicator

    def __getstate__(self): 
        """Method used to pickle.

        Bound method attributes cannot be pickled and must therefore be deleted
        before an instance is dumped to file.

        """

        # Get state of object and take care of troublesome attributes
        state = dict(self.__dict__)
        state['kd'].__dict__['comm'] = serial_comm
        state.pop('calc')
        state.pop('perturbation')
        state.pop('response_calc')
        
        return state
           
    def initialize(self):
        """Initialize response calculator and perturbation."""

        # Get scaled atomic positions
        spos_ac = self.atoms.get_scaled_positions()

        self.perturbation.initialize(spos_ac)
        self.response_calc.initialize(spos_ac)

        self.initialized = True

    def collect(self):
        """Collect calculated quantities."""

        self.D_matrix.collect()
        if self.e_ph is not None:
            self.e_ph.collect()
            
    def set_atoms(self, atoms_a):
        """Set indices of atoms to include in the calculation."""

        assert isinstance(atoms_a, list)

        self.D_matrix.set_indices(atoms_a)

    def get_dynamical_matrix(self):
        """Return reference to ``DynamicalMatrix`` object."""
        
        return self.D_matrix
    
    def run(self, qpts_q=None, clean=False, name=None, path=None):
        """Run calculation for atomic displacements and update matrix.

        Parameters
        ----------
        qpts: List
            List of q-points indices for which the dynamical matrix will be
            calculated (only temporary).

        """

        if not self.initialized:
            self.initialize()

        if self.gamma:
            qpts_q = [0]
        elif qpts_q is None:
            qpts_q = range(self.kd.nibzkpts)
        else:
            assert isinstance(qpts_q, list)

        # Update name and path in the dynamical matrix
        self.D_matrix.set_name_and_path(name=name, path=path)
        # Get string template for filenames
        filename_str = self.D_matrix.get_filename_string()

        # Delay the ranks belonging to the same k-point/domain decomposition
        # equally
        time.sleep(rank // self.comm.size)

        # XXX Make a single ground_state_contributions member function
        # Ground-state contributions to the force constants
        self.D_matrix.density_ground_state(self.calc)
        # self.D_matrix.wfs_ground_state(self.calc, self.response_calc)
        
        # Calculate linear response wrt q-vectors and displacements of atoms
        for q in qpts_q:
            
            if not self.gamma:
                self.perturbation.set_q(q)

            # First-order contributions to the force constants
            for a in self.D_matrix.indices:
    
                for v in [0, 1, 2]:

                    # Check if the calculation has already been done
                    filename = filename_str % (q, a, v)
                    # Wait for all sub-ranks to enter
                    self.comm.barrier()
                    
                    if os.path.isfile(filename):
                        continue

                    if self.comm.rank == 0:
                        fd = open(filename, 'w')
                        fd.close()
                    # Wait for all sub-ranks here
                    self.comm.barrier()
                    
                    components = ['x', 'y', 'z']
                    symbols = self.atoms.get_chemical_symbols()
                    print "q-vector index: %i" % q
                    print "Atom index: %i" % a
                    print "Atomic symbol: %s" % symbols[a]
                    print "Component: %s" % components[v]

                    # Set atom and cartesian component of perturbation
                    self.perturbation.set_av(a, v)
                    # Calculate linear response
                    self.response_calc(self.perturbation)

                    # Calculate row of the matrix of force constants
                    self.D_matrix.update_row(self.perturbation,
                                             self.response_calc)

                    # Write force constants to file
                    if self.comm.rank == 0:
                        self.D_matrix.write(q, a, v)

                    # Store effective potential derivative
                    if self.e_ph is not None:
                        v1_eff_G = self.perturbation.v1_G + \
                                   self.response_calc.vHXC1_G
                        self.e_ph.v1_eff_qavG.append(v1_eff_G)

                    # Wait for the file-writing rank here
                    self.comm.barrier()

        # Remove the files
        if clean:
            self.D_matrix.clean()
