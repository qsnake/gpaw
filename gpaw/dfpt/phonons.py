# -*- coding: utf-8 -*-

"""Phonons for periodic systems."""

import pickle
from math import sin, pi, sqrt
from os import remove
from os.path import isfile
import sys

import numpy as np

import ase.units as units
from ase.io.trajectory import PickleTrajectory
from ase.parallel import rank, barrier, paropen

class Phonons:
    """Class for calculating phonon modes using finite difference.

    The normal modes are calculated from a finite difference approximation to
    the matrix of force constants.

    Example:

    >>> from ase.structure import bulk
    >>> from ase.optimizers import BFGS
    >>> from ase.phonons import Phonons
    >>> from gpaw import GPAW, FermiDirac
    >>> atoms = bulk('Si2', 'diamond', a=5.4)
    >>> calc = GPAW(kpts=(5, 5, 5),
                    h=0.2,
                    occupations=FermiDirac(0.))
    >>> atoms.set_calculator(calc)
    >>> BFGS(atoms).run(fmax=0.01)
    BFGS:   0  19:16:06        0.042171       2.9357
    BFGS:   1  19:16:07        0.104197       3.9270
    BFGS:   2  19:16:07        0.000963       0.4142
    BFGS:   3  19:16:07        0.000027       0.0698
    BFGS:   4  19:16:07        0.000000       0.0010
    >>> ph = Phonons(atoms, calc, supercell=(5, 5, 5))
    >>> ph.run()

    """

    def __init__(self, atoms, calc, supercell=(1, 1, 1), name='phonon',
                 delta=0.01, nfree=2):
        """Init with an instance of class ``Atoms`` containing a calculator.

        Parameters
        ----------
        atoms: Atoms object
            The atoms to work on.
        supercell: tuple
            Size of supercell given by the number of repetitions of the small
            unit cell in each direction.
        name: str
            Name to use for files.
        delta: float
            Magnitude of displacements.
        nfree: int
            Number of displacements per atom and cartesian coordinate,
            2 and 4 are supported. Default is 2 which will displace 
            each atom +delta and -delta for each cartesian coordinate.

        """
        
        assert nfree in [2, 4]
        self.atoms = atoms
        self.atoms_lmn = atoms * supercell
        self.atoms_lmn.set_calculator(calc)
        # Vibrate all atoms in small unit cell by default
        indices = range(len(atoms))
        self.name = name
        self.delta = delta
        self.nfree = nfree

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

        This will calculate the forces for 6 displacements per atom ±x, ±y, ±z.
        Only those calculations that are not already done will be started. Be
        aware that an interrupted calculation may produce an empty file (ending
        with .pckl), which must be deleted before restarting the job. Otherwise
        the forces will not be calculated for that displacement.

        """

        # Calculate forces in equilibrium structure
        filename = self.name + '.eq.pckl'
        if not isfile(filename):
            barrier()
            forces = self.atoms_lmn.get_forces()
            ## if self.ir:
            ##     dipole = self.calc.get_dipole_moment(self.atoms)
            if rank == 0:
                fd = open(filename, 'w')
                if False: ## self.ir:
                    pickle.dump([forces, dipole], fd)
                    sys.stdout.write(
                        'Writing %s, dipole moment = (%.6f %.6f %.6f)\n' % 
                        (filename, dipole[0], dipole[1], dipole[2]))
                else:
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
                    for ndis in range(1, self.nfree//2+1):
                        # Filename for atomic displacement
                        filename = ('%s.%d%s%s.pckl' %
                                    (self.name, a, 'xyz'[i], ndis*' +-'[sign]))
                        # Skip if already being processed
                        if isfile(filename):
                            continue
                        barrier()
                        # XXX The file should be created immediately ???
                        self.atoms_lmn.positions[a, i] = \
                            (pos[a, i] + ndis * sign * self.delta)
                        forces = self.atoms_lmn.get_forces()
                        ## if self.ir:
                        ##     dipole = self.calc.get_dipole_moment(self.atoms_lmn)
                        if rank == 0:
                            fd = open(filename, 'w')
                            if False: ## self.ir:
                                pickle.dump([forces, dipole], fd)
                                sys.stdout.write(
                                    'Writing %s, ' % filename +
                                    'dipole moment = (%.6f %.6f %.6f)\n' % 
                                    (dipole[0], dipole[1], dipole[2]))
                            else:
                                pickle.dump(forces, fd)
                                sys.stdout.write('Writing %s\n' % filename)
                            fd.close()
                        sys.stdout.flush()
                        self.atoms_lmn.positions[a, i] = pos[a, i]
                        
        # self.atoms.set_positions(pos)

    def clean(self):
        """Delete generated pickle files."""
        
        if isfile(self.name + '.eq.pckl'):
            remove(self.name + '.eq.pckl')
        
        for a in self.indices:
            for i in 'xyz':
                for sign in '-+':
                    for ndis in range(1, self.nfree/2+1):
                        name = '%s.%d%s%s.pckl' % (self.name, a, i, ndis*sign)
                        if isfile(name):
                            remove(name)
        
    def read(self, acoustic=True, direction='central', method='frederiksen'):
        """Read pickle files and calculate matrix of force constants.

        Parameters
        ----------
        acoustic: bool
            Enforce the acoustic sum-rule on the matrix of force constants.
        direction: str
            Type of finite difference approximation to use.
        method: str
            Specify method for evaluating the atomic forces.
            
        """
        
        direction = direction.lower()
        assert method in ['standard', 'frederiksen']
        assert direction in ['central', 'forward', 'backward']

        
        # Number of atoms
        N = len(self.indices)
        # Number of unit cells
        M = np.prod(self.supercell)
        # Matrix of force constants as a function of unit cell index
        C_m = np.empty((M, 3*N, 3*N), dtype=float)

        # Equilibrium forces
        if direction != 'central':
            feq = pickle.load(open(self.name + '.eq.pckl'))
            
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
                if direction == 'central':
                    if self.nfree == 2:
                        C_av = fminus_av - fplus_av
                   
                C_av /= 2 * self.delta
                
                # Slice out included atoms
                C_mav = C_av.reshape((M, len(self.atoms), 3))[:, self.indices]
                index = 3*i + j                
                C_m[:, index] = C_mav.reshape((-1, 3*N))

        # Make force constants symmetric
        ## C_lmn = C_m.reshape(self.supercell + (-1,))
        ## for C in C_m:
        ##     C *= 0.5
        ##     C += C.T.copy()

        # Add mass prefactor
        m = self.atoms.get_masses()
        self.m_inv = np.repeat(m[self.indices]**-0.5, 3)
        for C in C_m:
            C *= self.m_inv[:, np.newaxis] * self.m_inv

        self.D = C_m
        # omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        # self.modes = modes.T.copy()

        # Conversion factor:
        # s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        # self.hnu = s * omega2.astype(complex)**0.5


    def band_structure(self, path_kc, modes=False, acoustic=True):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space matrix. In case of negative eigenvalues
        (squared frequency), the corresponding negative frequency is returned.

        Parameters
        ----------
        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
            
        """

        for k_c in path_kc:
            assert np.all(np.asarray(k_c) <= 1.0), \
                   "Scaled coordinates must be given"

        R_cm = np.indices(self.supercell).reshape(3, -1)
        N_c = np.array(self.supercell)[:, np.newaxis]
        R_cm += N_c // 2
        R_cm %= N_c
        R_cm -= N_c // 2        

        # Lists for frequencies and modes along path
        omega_kn = []
        u_kn =  []
        
        for q_c in path_kc:

            # Evaluate fourier transform 
            phase_m = np.exp(-2.j * pi * np.dot(q_c, R_cm))
            # Dynamical matrix in unit of Ha / Bohr**2 / amu
            D_q = np.sum(phase_m[:, np.newaxis, np.newaxis] * DR_m, axis=0)

            if modes:
                omega2_n, u_avn = la.eigh(D_q, UPLO='L')
                # Sort eigenmodes according to eigenvalues (see below) 
                u_nav = u_avn[:, omega2_n.argsort()].T.copy()
                # Multiply with mass prefactor
                u_kn.append(u_nav * self.m_inv)
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

        if modes:
            return np.asarray(omega_kn), np.asarray(u_kn)
        
        return np.asarray(omega_kn)
