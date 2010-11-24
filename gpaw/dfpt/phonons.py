"""Module for calculating phonons of periodic systems."""

import sys
import pickle
from math import sin, pi, sqrt
from os import remove
from os.path import isfile

import numpy as np
import numpy.linalg as la
import numpy.fft as fft

import ase.units as units
from ase.io.trajectory import PickleTrajectory
from ase.parallel import rank, barrier

class Phonons:
    """Class for calculating phonon modes using finite difference.

    The matrix of force constants is calculated from the finite difference
    approximation to the first-order derivative of the atomic forces as::
    
                            2           nbj   nbj
                nbj        d V         F-  - F+
               C     = ------------ ~  ----------  ,
                mai     dR   dR         2 * delta
                          mai  nbj       

    where F+/F- denotes the force in direction j on atom nb when atom ma is
    displaced in direction +i/-i. The force constants are related by various
    symmetry relations. From the definition of the force constants it must
    be symmetric in the three indices mai::

                nbj    mai         bj        ai
               C    = C      ->   C  (R ) = C  (-R )  .
                mai    nbj         ai  n     bj   n

    As the force constants can only depend on the difference between the m and
    n indices, this symmetry is more conveniently expressed as shown on the
    right hand-side.

    The acoustic sum-rule::

                            _
                aj         \    bj    
               C  (R ) = -  >  C  (R )
                ai  0      /_   ai  m
                         (m, b)
                           !=
                         (0, a)
                        
    Ordering of the unit cells illustrated here for a 1-dimensional system:
    
    ::
    
               m = 0        m = 1        m = -1        m = -2
           -----------------------------------------------------
           |            |            |            |            |
           |        * b |        *   |        *   |        *   |
           |            |            |            |            |
           |   * a      |   *        |   *        |   *        |
           |            |            |            |            |
           -----------------------------------------------------
       
    Example:

    >>> from ase.structure import bulk
    >>> from ase.phonons import Phonons
    >>> from gpaw import GPAW, FermiDirac
    >>> atoms = bulk('Si2', 'diamond', a=5.4)
    >>> calc = GPAW(kpts=(5, 5, 5),
                    h=0.2,
                    occupations=FermiDirac(0.))
    >>> ph = Phonons(atoms, calc, supercell=(5, 5, 5))
    >>> ph.run()
    >>> ph.read(method='frederiksen', acoustic=True)

    """

    def __init__(self, atoms, calc, supercell=(1, 1, 1), name='phonon',
                 delta=0.01):
        """Init with an instance of class ``Atoms`` and a calculator.

        Parameters
        ----------
        atoms: Atoms object
            The atoms to work on.
        calc: Calculator
            Calculator for the supercell calculation.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Name to use for files.
        delta: float
            Magnitude of displacements.

        """
        
        self.atoms = atoms
        # Atoms in the supercell -- repeated in the lattice vector directions
        # beginning with the last
        self.atoms_lmn = atoms * supercell
        self.atoms_lmn.set_calculator(calc)
        # Vibrate all atoms in small unit cell by default
        self.indices = range(len(atoms))
        self.name = name
        self.delta = delta
        self.N_c = supercell

        # Attributes for force constants and dynamical matrix in real-space
        self.C_m = None  # in units of eV / Ang**2 
        self.D_m = None  # in units of eV / Ang**2 / amu
        
    def set_atoms(self, atoms):
        """Set the atoms to vibrate.

        Parameters
        ----------
        atoms: list
            Can be either a list of strings, ints or ...
            
        """
        
        assert isinstance(atoms, list)
        assert len(atoms) <= len(self.atoms)
        
        if isinstance(atoms[0], str):
            assert np.all([isinstance(atom, str) for atom in atoms])
            sym_a = self.atoms.get_chemical_symbols()
            # List for atomic indices
            indices = []
            for type in atoms:
                indices.extend([a for a, atom in enumerate(sym_a)
                                if atom == type])
        else:
            assert np.all([isinstance(atom, int) for atom in atoms])
            indices = atoms

        self.indices = indices
        
    def run(self):
        """Run the total energy calculations for the required displacements.

        This will calculate the forces for 6 displacements per atom, +-x, +-y,
        and +-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .pckl), which must be deleted before restarting the
        job. Otherwise the forces will not be calculated for that displacement.

        """

        # Calculate forces in equilibrium structure
        filename = self.name + '.eq.pckl'
        
        if not isfile(filename):
            # Wait for all ranks to enter
            barrier()
            # Create file
            if rank == 0:
                fd = open(filename, 'w')
                fd.close()
            # Calculate forces
            forces = self.atoms_lmn.get_forces()
            # Write forces to file
            if rank == 0:
                fd = open(filename, 'w')
                pickle.dump(forces, fd)
                sys.stdout.write('Writing %s\n' % filename)
                fd.close()
            sys.stdout.flush()

        # Positions of atoms to be displaced in the small unit cell
        pos = self.atoms.positions.copy()
        
        # Loop over all displacements and calculate forces
        for a in self.indices:
            for i in range(3):
                for sign in [-1, 1]:
                    # Filename for atomic displacement
                    filename = ('%s.%d%s%s.pckl' %
                                (self.name, a, 'xyz'[i], ' +-'[sign]))
                    # Skip if already being processed
                    if isfile(filename):
                        continue
                    # Wait for ranks
                    barrier()
                    if rank == 0:
                        fd = open(filename, 'w')
                        fd.close()
                    # Update atomic positions and calculate forces
                    self.atoms_lmn.positions[a, i] = \
                        pos[a, i] + sign * self.delta
                    forces = self.atoms_lmn.get_forces()
                    # Write forces to file                        
                    if rank == 0:
                        fd = open(filename, 'w')
                        pickle.dump(forces, fd)
                        sys.stdout.write('Writing %s\n' % filename)
                        fd.close()
                    sys.stdout.flush()
                    # Return to initial positions
                    self.atoms_lmn.positions[a, i] = pos[a, i]
                        
        # self.atoms.set_positions(pos)

    def clean(self):
        """Delete generated pickle files."""
        
        if isfile(self.name + '.eq.pckl'):
            remove(self.name + '.eq.pckl')
        
        for a in self.indices:
            for i in 'xyz':
                for sign in '-+':
                    name = '%s.%d%s%s.pckl' % (self.name, a, i, sign)
                    if isfile(name):
                        remove(name)
        
    def read(self, method='Frederiksen', acoustic=True):
        """Read pickle files and calculate matrix of force constants.

        Parameters
        ----------
        method: str
            Specify method for evaluating the atomic forces.
        acoustic: bool
            Restore the acoustic sum-rule on the force constants.
            
        """

        method = method.lower()
        assert method in ['standard', 'frederiksen']
        
        # Number of atoms
        N = len(self.indices)
        # Number of unit cells
        M = np.prod(self.N_c)
        # Matrix of force constants as a function of unit cell index in units
        # of eV/Ang**2 
        C_m = np.empty((3*N, 3*N*M), dtype=float)

        # Loop over all atomic displacements and calculate force constants
        for i, a in enumerate(self.indices):
            for j, v in enumerate('xyz'):
                # Atomic forces for a displacement of atom a in direction v
                name = '%s.%d%s' % (self.name, a, v)
                fminus_av = pickle.load(open(name + '-.pckl'))
                fplus_av = pickle.load(open(name + '+.pckl'))
                
                if method == 'frederiksen':
                    fminus_av[a] -= fminus_av.sum(0)
                    fplus_av[a] -= fplus_av.sum(0)

                # Finite difference derivative
                C_av = fminus_av - fplus_av
                C_av /= 2 * self.delta

                # Slice out included atoms
                C_mav = C_av.reshape((M, len(self.atoms), 3))[:, self.indices]
                index = 3*i + j                
                C_m[index] = C_mav.ravel()

        # Reshape force constant to (l, m, n) cell indices
        C_lmn = C_m.transpose().copy().reshape(self.N_c + (3*N, 3*N))
        # Shift reference cell to center
        C_lmn = fft.fftshift(C_lmn, axes=(0, 1, 2))
        # Make force constants symmetric in indices -- in case of an even
        # number of unit cells don't include the first
        i, j, k = (np.asarray(self.N_c) + 1) % 2
        C_lmn[i:, j:, k:] *= 0.5
        C_lmn[i:, j:, k:] += \
            C_lmn[i:, j:, k:][::-1, ::-1, ::-1].transpose(0, 1, 2, 4, 3).copy()
        C_lmn = fft.ifftshift(C_lmn, axes=(0, 1, 2))

        # Change to single unit cell index shape
        C_m = C_lmn.reshape((M, 3*N, 3*N))

        # Restore acoustic sum-rule
        if acoustic:
            # Copy force constants
            C_m_temp = C_m.copy()
            # Correct atomic diagonals of R_m = 0 matrix
            for C in C_m_temp:
                for a in range(N):
                    for a_ in range(N):
                        C_m[0, 3*a: 3*a + 3, 3*a: 3*a + 3] -= \
                               C[3*a: 3*a+3, 3*a_: 3*a_+3]
                        
        # Store force constants and dynamical matrix
        self.C_m = C_m
        self.D_m = C_m.copy()
        
        # Add mass prefactor
        m = self.atoms.get_masses()
        self.m_inv = np.repeat(m[self.indices]**-0.5, 3)
        M_inv = self.m_inv[:, np.newaxis] * self.m_inv
        for D in self.D_m:
            D *= M_inv

    def get_force_constant(self):
        """Return matrix of force constants."""

        assert self.C_m is not None
        
        return self.C_m
    
    def band_structure(self, path_kc, modes=False):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space force constants. In case of negative
        eigenvalues (squared frequency), the corresponding negative frequency
        is returned.

        Eigenvalues and modes are in units of eV and Ang/sqrt(amu),
        respectively.

        Parameters
        ----------
        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
            
        """

        assert self.D_m is not None
        
        for k_c in path_kc:
            assert np.all(np.asarray(k_c) <= 1.0), \
                   "Scaled coordinates must be given"

        # Lattice vectors
        R_cm = np.indices(self.N_c).reshape(3, -1)
        N_c = np.array(self.N_c)[:, np.newaxis]
        R_cm += N_c // 2
        R_cm %= N_c
        R_cm -= N_c // 2        

        # Lists for frequencies and modes along path
        omega_kn = []
        u_kn =  []
        
        for q_c in path_kc:

            # Evaluate fourier sum
            phase_m = np.exp(-2.j * pi * np.dot(q_c, R_cm))
            D_q = np.sum(phase_m[:, np.newaxis, np.newaxis] * self.D_m, axis=0)

            if modes:
                omega2_n, u_avn = la.eigh(D_q, UPLO='L')
                # Sort eigenmodes according to eigenvalues (see below) and 
                # multiply with mass prefactor
                u_nav = self.m_inv * u_avn[:, omega2_n.argsort()].T.copy()
                u_kn.append(u_nav.reshape((-1, 3)))
            else:
                omega2_n = la.eigvalsh(D_q, UPLO='L')

            # Sort eigenvalues in increasing order
            omega2_n.sort()
            # Use dtype=complex to handle negative eigenvalues
            omega_n = np.sqrt(omega2_n.astype(complex))

            # Take care of imaginary frequencies
            if not np.all(omega2_n >= 0.):
                indices = np.where(omega2_n < 0)[0]
                print ("WARNING, %i imaginary frequencies at "
                       "q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)"
                       % (len(indices), q_c[0], q_c[1], q_c[2],
                          omega_n[indices][0].imag))
                
                omega_n[indices] = -1 * np.sqrt(np.abs(omega2_n[indices].real))

            omega_kn.append(omega_n.real)

        # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        omega_kn = s * np.asarray(omega_kn)
        
        if modes:
            return omega_kn, np.asarray(u_kn)
        
        return omega_kn

    def write_modes(self, q_c, branches=0, kT=units.kB*300, repeat=(1, 1, 1),
                    nimages=30):
        """Write modes to trajectory file.

        Parameters
        ----------
        q_c: ndarray
            q-vector of modes.
        branches: int or list
            Branch index of modes.
        kT: float
            Temperature in units of eV. Determines the amplitude of the atomic
            displacements in the modes.
        repeat: tuple
            Repeat atoms (l, m, n) times in the directions of the lattice
            vectors. Displacements of atoms in repeated cells carry a Bloch
            phase factor given by the q-vector and the cell lattice vector R_m.
        nimages: int
            Number of images in an oscillation.
            
        """

        if isinstance(branches, int):
            branch_n = [branches]
        else:
            branch_n = list(branches)

        # Calculate modes
        omega_n, u_n = self.band_structure([q_c], modes=True)
        
        # Repeat atoms
        atoms = self.atoms * repeat
        # Here ma refers to a composite unit cell/atom dimension
        pos_mav = atoms.get_positions()
        # Total number of unit cells
        M = np.prod(repeat)

        # Corresponding lattice vectors R_m
        R_cm = np.indices(repeat).reshape(3, -1)
        # Bloch phase
        phase_m = np.exp(2.j * pi * np.dot(q_c, R_cm))
        phase_ma = phase_m.repeat(len(self.atoms))

        for n in branch_n:
            
            omega = omega_n[0, n]
            u_av = u_n[0, n]
            # Mean displacement of a classical oscillator at temperature T
            u_av *= sqrt(kT) / abs(omega)

            mode_av = np.zeros((len(self.atoms), 3), dtype=complex)
            # Insert slice with atomic displacements for the included atoms
            mode_av[self.indices] = u_av
            # Repeat and multiply by Bloch phase factor
            mode_mav = (np.vstack([mode_av]*M) * phase_ma[:, np.newaxis]).real

            traj = PickleTrajectory('%s.mode.%d.traj' % (self.name, n), 'w')
            
            for x in np.linspace(0, 2*pi, nimages, endpoint=False):
                atoms.set_positions(pos_mav + sin(x) * mode_mav)
                traj.write(atoms)
                
            traj.close()
