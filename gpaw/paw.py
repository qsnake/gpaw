# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import sys
import weakref

import numpy as npy
from ase.atoms import Atoms
from ase.data import atomic_numbers, chemical_symbols
from ase.units import Bohr, Hartree
from ase.dft import monkhorst_pack

import gpaw.io
import gpaw.mpi as mpi
import gpaw.occupations as occupations
from gpaw import parsize, parsize_bands, dry_run
from gpaw import ConvergenceError
from gpaw.density import Density
from gpaw.eigensolvers import get_eigensolver
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.poisson import PoissonSolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import Hamiltonian
from gpaw.overlap import Overlap
from gpaw.lcao.eigensolver import LCAO
from gpaw.kpoint import KPoint
from gpaw.localized_functions import LocFuncBroadcaster
from gpaw.utilities.timing import Timer
from gpaw.xc_functional import XCFunctional
from gpaw.mpi import run, MASTER
from gpaw.brillouin import reduce_kpoints
import _gpaw


import os
import sys
import tempfile
import time

from gpaw.utilities import check_unit_cell
from gpaw.utilities.memory import maxrss
import gpaw.utilities.timing as timing
import gpaw.io
import gpaw.mpi as mpi
from gpaw.nucleus import Nucleus
from gpaw.rotation import rotation
from gpaw.domain import Domain
from gpaw.xc_functional import XCFunctional
from gpaw.utilities import gcd
from gpaw.utilities.memory import estimate_memory
from gpaw.setup import create_setup
from gpaw.pawextra import PAWExtra
from gpaw.output import Output


class PAW(PAWExtra, Output):
    """This is the main calculation object for doing a PAW calculation.

    The ``Paw`` object is the central object for a calculation.  It is
    a container for **k**-points (there may only be one **k**-point).
    The attribute ``kpt_u`` is a list of ``KPoint`` objects (the
    **k**-point object stores the actual wave functions, occupation
    numbers and eigenvalues).  Each **k**-point object can be either
    spin up, spin down or no spin (spin-saturated calculation).
    Example: For a spin-polarized calculation on an isolated molecule,
    the **k**-point list will have length two (assuming the
    calculation is not parallelized over **k**-points/spin).

    These are the most important attributes of a ``Paw`` object:

    =============== =====================================================
    Name            Description
    =============== =====================================================
    ``domain``      Domain object.
    ``setups``      List of setup objects.
    ``symmetry``    Symmetry object.
    ``timer``       Timer object.
    ``nuclei``      List of ``Nucleus`` objects.
    ``gd``          Grid descriptor for coarse grids.
    ``finegd``      Grid descriptor for fine grids.
    ``kpt_u``       List of **k**-point objects.
    ``occupation``  Occupation-number object.
    ``nkpts``       Number of irreducible **k**-points.
    ``nmyu``        Number of irreducible spin/**k**-points pairs on
                    *this* CPU.
    ``nvalence``    Number of valence electrons.
    ``nbands``      Number of bands.
    ``nspins``      Number of spins.
    ``dtype``       Data type of wave functions (``float`` or
                    ``complex``).
    ``bzk_kc``      Scaled **k**-points used for sampling the whole
                    Brillouin zone - values scaled to [-0.5, 0.5).
    ``ibzk_kc``     Scaled **k**-points in the irreducible part of the
                    Brillouin zone.
    ``weight_k``    Weights of the **k**-points in the irreducible part
                    of the Brillouin zone (summing up to 1).
    ``myibzk_kc``   Scaled **k**-points in the irreducible part of the
                    Brillouin zone for this CPU.
    ``kpt_comm``    MPI-communicator for parallelization over
                    **k**-points.
    =============== =====================================================

    Energy contributions and forces:

    =========== ==========================================
                Description
    =========== ==========================================
    ``Ekin``    Kinetic energy.
    ``Epot``    Potential energy.
    ``Etot``    Total energy.
    ``Exc``     Exchange-Correlation energy.
    ``Eext``    Energy of external potential
    ``Eref``    Reference energy for all-electron atoms.
    ``S``       Entropy.
    ``Ebar``    Should be close to zero!
    ``F_ac``    Forces.
    =========== ==========================================

    The attribute ``usesymm`` has the same meaning as the
    corresponding ``Calculator`` keyword (see the Manual_).  Internal
    units are Hartree and Angstrom and ``Ha`` and ``a0`` are the
    conversion factors to external `ASE units`_.  ``error`` is the
    error in the Kohn-Sham wave functions - should be zero (or small)
    for a converged calculation.

    Booleans describing the current state:

    ============= ======================================
    Boolean       Description
    ============= ======================================
    ``forces_ok`` Have the forces bee calculated yet?
    ``converged`` Do we have a self-consistent solution?
    ============= ======================================

    Number of iterations for:

    ============ ===============================
                 Description
    ============ ===============================
    ``nfermi``   finding the Fermi-level
    ``niter``    solving the Kohn-Sham equations
    ``npoisson`` Solving the Poisson equation
    ============ ===============================

    Only attribute not mentioned now is ``nspins`` (number of spins) and
    those used for parallelization:

    ================== ===================================================
    ``my_nuclei``      List of nuclei that have their
                       center in this domain.
    ``pt_nuclei``      List of nuclei with projector functions
                       overlapping this domain.
    ``ghat_nuclei``    List of nuclei with compensation charges
                       overlapping this domain.
    ``locfuncbcaster`` ``LocFuncBroadcaster`` object for parallelizing
                       evaluation of localized functions (used when
                       parallelizing over **k**-points).
    ================== ===================================================

    .. _Manual: https://wiki.fysik.dtu.dk/gridcode/Manual
    .. _ASE units: https://wiki.fysik.dtu.dk/ase/Units

    Parameters:

    =============== ===================================================
    ``nvalence``    Number of valence electrons.
    ``nbands``      Number of bands.
    ``nspins``      Number of spins.
    ``random``      Initialize wave functions with random numbers
    ``dtype``       Data type of wave functions (``float`` or
                    ``complex``).
    ``kT``          Temperature for Fermi-distribution.
    ``bzk_kc``      Scaled **k**-points used for sampling the whole
                    Brillouin zone - values scaled to [-0.5, 0.5).
    ``ibzk_kc``     Scaled **k**-points in the irreducible part of the
                    Brillouin zone.
    ``myspins``     List of spin-indices for this CPU.
    ``weight_k``    Weights of the **k**-points in the irreducible part
                    of the Brillouin zone (summing up to 1).
    ``myibzk_kc``   Scaled **k**-points in the irreducible part of the
                    Brillouin zone for this CPU.
    ``world``       MPI-communicator for any parallelized operations
    ``kpt_comm``    MPI-communicator for parallelization over
                    **k**-points.
    =============== ===================================================
    """

    def __init__(self, filename=None, **kwargs):
        """ASE-calculator interface.

        The following parameters can be used: `nbands`, `xc`, `kpts`,
        `spinpol`, `gpts`, `h`, `charge`, `usesymm`, `width`, `mixer`,
        `hund`, `lmax`, `fixdensity`, `convergence`, `txt`,
        `parsize`, `softgauss` and `stencils`.

        If you don't specify any parameters, you will get:

        Defaults: neutrally charged, LDA, gamma-point calculation, a
        reasonable grid-spacing, zero Kelvin electronic temperature,
        and the number of bands will be equal to the number of atomic
        orbitals present in the setups. Only occupied bands are used
        in the convergence decision. The calculation will be
        spin-polarized if and only if one or more of the atoms have
        non-zero magnetic moments. Text output will be written to
        standard output.

        For a non-gamma point calculation, the electronic temperature
        will be 0.1 eV (energies are extrapolated to zero Kelvin) and
        all symmetries will be used to reduce the number of
        **k**-points."""

        self.timer = Timer()

        self.input_parameters = {
            'h':             None,  # Angstrom
            'xc':            'LDA',
            'gpts':          None,
            'kpts':          None,
            'lmax':          2,
            'charge':        0,
            'fixmom':        False,
            'nbands':        None,
            'setups':        'paw',
            'basis':         {},
            'width':         None,  # eV
            'spinpol':       None,
            'usesymm':       True,
            'stencils':      (2, 3),
            'convergence':   {'energy': 0.001,  # eV
                              'density': 1.0e-3,
                              'eigenstates': 1.0e-9,
                              'bands': 'occupied'},
            'fixdensity':    0,
            'mixer':         None,
            'txt':           '-',
            'hund':          False,
            'random':        False,
            'maxiter':       120,
            'parsize':       None,
            'parsize_bands': None,
            'external':      None,  # eV
            'verbose':       0,
            'eigensolver':   'rmm-diis',
            'poissonsolver': None,
            'communicator' : None,
            'idiotproof'   : True,
            }

        self.initialized = False
        self.converged = False
        self.error = {}
        self.wave_functions_initialized = False
        self.wave_functions_orthonormalized = False
        self.callback_functions = []
        self.niter = 0
        self.F_ac = None

        self.eigensolver = None
        self.density = None
        self.kpt_u = None
        self.nbands = None
        self.nmybands = None
        self.atoms = None

        if filename is not None:
            reader = self.read_parameters(filename)

        self.set(**kwargs)

        Output.__init__(self)

        if filename is not None:
            self.initialize()
            gpaw.io.read(self, reader)
            self.plot_atoms(self.atoms)

    def set(self, **kwargs):
        p = self.input_parameters
        
        if (kwargs.get('h') is not None) and (kwargs.get('gpts') is not None):
            raise TypeError("""You can't use both "gpts" and "h"!""")
        if 'h' in kwargs:
            p['gpts'] = None
        if 'gpts' in kwargs:
            p['h'] = None

        # Special treatment for convergence criteria dictionary:
        if kwargs.get('convergence') is not None:
            cc = p['convergence']
            cc.update(kwargs['convergence'])
            kwargs['convergence'] = cc

        p.update(kwargs)

        self.initialized = False
        self.reuse_old_density = True
        
        for key in kwargs:
            if key in ['fixmom', 'mixer', 'convergence', 'fixdensity',
                       'verbose', 'txt', 'hund', 'random', 'maxiter',
                       'eigensolver', 'poissonsolver', 'idiotproof']:
                continue

            # More drastic changes:
            self.converged = False
            self.error = {}
            self.wave_functions_orthonormalized = False
            if key in ['lmax', 'width', 'stencils', 'external']:
                pass
            elif key in ['charge', 'xc']:
                self.reuse_old_density = False
            elif key in ['kpts', 'nbands']:
                self.kpt_u = None
            elif key in ['h', 'gpts', 'setups', 'basis', 'spinpol',
                         'usesymm', 'parsize', 'parsize_bands',
                         'communicator']:
                self.reuse_old_density = False
                self.kpt_u = None
            
    def calculate(self, atoms):
        """Update PAW calculaton if needed."""

        if (self.atoms is not None and
            ((len(atoms) != len(self.atoms) or
              (atoms.get_atomic_numbers() !=
               self.atoms.get_atomic_numbers()).any() or
              (atoms.get_cell() != self.atoms.get_cell()).any() or
              (atoms.get_pbc() != self.atoms.get_pbc()).any()))):
            # Drastic changes:
            self.kpt_u = None
            self.reuse_old_density = False
            self.initialize(atoms)
            self.print_parameters()
            self.find_ground_state(atoms)
            return

        if not self.initialized:
            self.initialize(atoms)
            self.print_parameters()
            self.find_ground_state(atoms)
            return

        if not self.converged:
            self.find_ground_state(atoms)
            return

        if (atoms.get_positions() != self.atoms.get_positions()).any():
            # The positions have changed.  Wave functions are no
            # longer orthonormal!
            self.wave_functions_orthonormalized = False
            self.find_ground_state(atoms)
        
    def get_atoms(self):
        atoms = self.atoms.copy()
        atoms.set_calculator(self)
        return atoms

    def find_ground_state(self, atoms, write=True):
        """Start iterating towards the ground state."""
        
        self.set_positions(atoms)
        self.initialize_kinetic()
        if not self.eigensolver.initialized:
            # We know that we have enough memory to initialize the
            # eigensolver here:
            self.eigensolver.initialize(self)
        if not self.wave_functions_initialized:
            self.initialize_wave_functions()
        if not self.wave_functions_orthonormalized:
            self.orthonormalize_wave_functions()
        if not self.density.initialized:
            self.density.update(self.kpt_u, self.symmetry)
        if self.xcfunc.is_gllb():
            if not self.xcfunc.xc.initialized:
                self.xcfunc.initialize_gllb(self)
        self.hamiltonian.update(self.density)

        # Self-consistency loop:
        while not self.converged:
            if self.niter > self.maxiter:
                raise ConvergenceError('Did not converge!')
            self.step()
            self.add_up_energies()
            self.check_convergence()
            self.print_iteration()
            self.niter += 1
            if write:
                self.call()

        if write:
            self.call(final=True)
            self.print_converged()

        # Don't fix the density in the next step:
        self.fixdensity = -1
        
    def step(self):
        if self.niter > self.fixdensity:
            self.density.update(self.kpt_u, self.symmetry)
            self.update_kinetic()
            self.hamiltonian.update(self.density)

        self.eigensolver.iterate(self.hamiltonian, self.kpt_u)

        # Make corrections due to non-local xc:
        xcfunc = self.hamiltonian.xc.xcfunc
        self.Enlxc = xcfunc.get_non_local_energy()
        self.Enlkin = xcfunc.get_non_local_kinetic_corrections()

        # Calculate occupation numbers:
        self.occupation.calculate(self.kpt_u)

    def add_up_energies(self):
        H = self.hamiltonian
        self.Ekin = H.Ekin + self.occupation.Eband + self.Enlkin
        self.Epot = H.Epot
        self.Eext = H.Eext
        self.Ebar = H.Ebar
        self.Exc = H.Exc + self.Enlxc
        self.S = self.occupation.S
        self.Etot = self.Ekin + self.Epot + self.Ebar + self.Exc - self.S

        if len(self.old_energies) == 3:
            self.old_energies.pop(0)
        self.old_energies.append(self.Etot)
        
    def set_positions(self, atoms=None):
        """Update the positions of the atoms.

        Localized functions centered on atoms that have moved will
        have to be computed again.  Neighbor list is updated and the
        array holding all the pseudo core densities is updated."""

        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()
            
        pos_ac = atoms.get_positions() / Bohr

        movement = False
        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = self.domain.scale_position(pos_c)
            if npy.sometrue(spos_c != nucleus.spos_c) or not nucleus.ready:
                movement = True
                nucleus.set_position(spos_c, self.domain, self.my_nuclei,
                                     self.nspins, self.nmyu, self.nmybands)
                nucleus.move(spos_c, self.gd, self.finegd,
                             self.ibzk_kc, self.locfuncbcaster,
                             self.pt_nuclei, self.ghat_nuclei)

        if movement:
            self.niter = 0
            self.converged = False
            self.F_ac = None
            self.old_energies = []
            
            self.locfuncbcaster.broadcast()

            for nucleus in self.nuclei:
                nucleus.normalize_shape_function_and_pseudo_core_density()

            if self.symmetry:
                self.symmetry.check(pos_ac)

            self.hamiltonian.pairpot.update(pos_ac, self.nuclei, self.domain,
                                            self.text)

            self.density.move()

            # Output the updated position of the atoms
            self.print_positions(pos_ac)

    def initialize_wave_functions(self):
        # do at least the first 3 iterations with fixed density
        self.fixdensity = max(2, self.fixdensity)

        if self.eigensolver.lcao:
            for nucleus in self.nuclei:
                nucleus.initialize_atomic_orbitals(self.gd, self.ibzk_kc,
                                                   self.locfuncbcaster)
            self.locfuncbcaster.broadcast()
            
            if not self.density.initialized:
                self.density.initialize()

            for kpt in self.kpt_u:
                kpt.allocate(self.nbands)

            self.wave_functions_initialized = True
            return
        
        if self.kpt_u[0].psit_nG is None:
            # Initialize wave functions from atomic orbitals:
            original_eigensolver = self.eigensolver
            original_nbands = self.nbands
            original_maxiter = self.maxiter

            self.text('Atomic orbitals used for initialization:', self.nao)
            if self.nbands > self.nao:
                self.text('Random orbitals used for initialization:',
                          self.nbands - self.nao)
            
            self.maxiter = 0
            self.nbands = min(self.nbands, self.nao)
            self.eigensolver = get_eigensolver('lcao')

            for nucleus in self.my_nuclei:
                nucleus.reallocate(self.nbands)

            self.density.lcao = True

            try:
                self.find_ground_state(self.atoms, write=False)
            except ConvergenceError:
                pass

            self.maxiter = original_maxiter
            self.nbands = original_nbands
            for kpt in self.kpt_u:
                kpt.calculate_wave_functions_from_lcao_coefficients(
                    self.nbands)
                # Delete basis-set expansion coefficients:
                kpt.C_nm = None

            for nucleus in self.nuclei:
                del nucleus.P_kmi
                
            for nucleus in self.my_nuclei:
                nucleus.reallocate(self.nbands)

            self.eigensolver = original_eigensolver
            if self.xcfunc.is_gllb():
                self.xcfunc.xc.eigensolver = self.eigensolver
            #self.density.mixer.reset(self.my_nuclei)
            self.density.lcao = False
            self.density.scale()
            self.density.interpolate_pseudo_density()
            self.converged = False
            self.wave_functions_orthonormalized = False
            self.wave_functions_initialized = True
            self.converged = False
            self.F_ac = None
            self.old_energies = []
            self.niter = 0

            # Free allocated space for radial grids:
            for setup in self.setups:
                 del setup.phit_j
            for nucleus in self.nuclei:
                del nucleus.phit_i

            if self.nbands > self.nao:
                for kpt in self.kpt_u:
                    kpt.add_extra_bands(self.nbands, self.nao)
                if self.xcfunc.is_gllb():
                    self.xcfunc.xc.update_band_count()

            # We should only create pt_i now !!!!!!!!!!!!!!!!!
            
            self.orthonormalize_wave_functions()

        elif not isinstance(self.kpt_u[0].psit_nG, npy.ndarray):
            # Calculation started from a restart file.  Copy data
            # from the file to memory:
            if self.world.size > 1:
                i = self.gd.get_slice()
                for kpt in self.kpt_u:
                    refs = kpt.psit_nG
                    kpt.psit_nG = self.gd.empty(self.nmybands, self.dtype)
                    # Read band by band to save memory
                    for n, psit_G in enumerate(kpt.psit_nG):
                        full = refs[n][:]
                        psit_G[:] = full[i]
            else:
                for kpt in self.kpt_u:
                    kpt.psit_nG = kpt.psit_nG[:]

            self.wave_functions_initialized = True

    def orthonormalize_wave_functions(self):
        if self.eigensolver.lcao:
            self.wave_functions_orthonormalized = True
            return
        
        for kpt in self.kpt_u:
            run([nucleus.calculate_projections(kpt)
                for nucleus in self.pt_nuclei])
            self.overlap.orthonormalize(kpt.psit_nG, kpt)
        self.wave_functions_orthonormalized = True

    def calculate_forces(self):
        """Return the atomic forces."""

        if self.F_ac is not None:
            return

        self.F_ac = npy.empty((self.natoms, 3))
        
        nt_g = self.density.nt_g
        vt_sG = self.hamiltonian.vt_sG
        vHt_g = self.hamiltonian.vHt_g

        if self.nspins == 2:
            vt_G = 0.5 * (vt_sG[0] + vt_sG[1])
        else:
            vt_G = vt_sG[0]

        for nucleus in self.my_nuclei:
            nucleus.F_c[:] = 0.0

        # Calculate force-contribution from k-points:
        for kpt in self.kpt_u:
            for nucleus in self.pt_nuclei:
                nucleus.calculate_force_kpoint(kpt)
        for nucleus in self.my_nuclei:
            self.kpt_comm.sum(nucleus.F_c)

        for nucleus in self.nuclei:
            nucleus.calculate_force(vHt_g, nt_g, vt_G)

        # Global master collects forces from nuclei into self.F_ac:
        if self.master:
            for a, nucleus in enumerate(self.nuclei):
                if nucleus.in_this_domain:
                    self.F_ac[a] = nucleus.F_c
                else:
                    self.domain.comm.receive(self.F_ac[a], nucleus.rank, 7)
        else:
            if self.kpt_comm.rank == 0:
                for nucleus in self.my_nuclei:
                    self.domain.comm.send(nucleus.F_c, MASTER, 7)

        # Broadcast the forces to all processors
        self.world.broadcast(self.F_ac, MASTER)

        # Add non-local contributions
        for kpt in self.kpt_u:
            self.F_ac += self.xcfunc.get_non_local_force(kpt)
            
        if self.symmetry is not None:
            # Symmetrize forces:
            F_ac = npy.zeros((self.natoms, 3))
            for map_a, symmetry in zip(self.symmetry.maps,
                                       self.symmetry.symmetries):
                swap, mirror = symmetry
                for a1, a2 in enumerate(map_a):
                    F_ac[a2] += npy.take(self.F_ac[a1] * mirror, swap)
            self.F_ac[:] = F_ac / len(self.symmetry.symmetries)
        
        self.print_forces()

    def attach(self, function, n, *args, **kwargs):
        """Register callback function.

        Call ``function`` every ``n`` iterations using ``args`` and
        ``kwargs`` as arguments."""

        try:
            slf = function.im_self
        except AttributeError:
            pass
        else:
            if slf is self:
                # function is a bound method of self.  Store the name
                # of the method and avoid circular reference:
                function = function.im_func.func_name
                
        self.callback_functions.append((function, n, args, kwargs))

    def call(self, final=False):
        """Call all registered callback functions."""
        for function, n, args, kwargs in self.callback_functions:
            if ((self.niter % n) == 0) != final:
                if isinstance(function, str):
                    function = getattr(self, function)
                function(*args, **kwargs)

    def create_nuclei_and_setups(self, Z_a):
        p = self.input_parameters

        setup_types = p['setups']
        if isinstance(setup_types, str):
            setup_types = {None: setup_types}
        
        # setup_types is a dictionary mapping chemical symbols and/or atom
        # numbers to setup types.
        
        # If present, None will map to the default type:
        default = setup_types.get(None, 'paw')
        
        type_a = [default] * self.natoms
        
        # First symbols ...
        for symbol, type in setup_types.items():
            if isinstance(symbol, str):
                number = atomic_numbers[symbol]
                for a, Z in enumerate(Z_a):
                    if Z == number:
                        type_a[a] = type
        
        # and then atom numbers:
        for a, type in setup_types.items():
            if isinstance(a, int):
                type_a[a] = type
        
        basis_sets = p['basis']
        if isinstance(basis_sets, str):
            basis_sets = {None: basis_sets}
        
        # basis_sets is a dictionary mapping chemical symbols and/or atom
        # numbers to basis sets.
        
        # If present, None will map to the default type:
        default = basis_sets.get(None, None)
        
        basis_a = [default] * self.natoms
        
        # First symbols ...
        for symbol, basis in basis_sets.items():
            if isinstance(symbol, str):
                number = atomic_numbers[symbol]
                for a, Z in enumerate(Z_a):
                    if Z == number:
                        basis_a[a] = basis
        
        # and then atom numbers:
        for a, basis in basis_sets.items():
            if isinstance(a, int):
                basis_a[a] = basis
        
        # Build list of nuclei and construct necessary PAW-setup objects:
        self.nuclei = []
        setups = {}
        for a, (Z, type, basis) in enumerate(zip(Z_a, type_a, basis_a)):
            if (Z, type, basis) in setups:
                setup = setups[(Z, type, basis)]
            else:
                symbol = chemical_symbols[Z]
                setup = create_setup(symbol, self.xcfunc, p['lmax'],
                                     self.nspins, type, basis)
                setup.print_info(self.text)
                setups[(Z, type, basis)] = setup
            self.nuclei.append(Nucleus(setup, a, self.dtype))

        self.setups = setups.values()
        return type_a, basis_a

    def read_parameters(self, filename):
        """Read state from file."""

        r = gpaw.io.open(filename, 'r')
        p = self.input_parameters

        version = r['version']
        
        assert version >= 0.3
    
        p['xc'] = r['XCFunctional']
        p['nbands'] = r.dimension('nbands')
        p['spinpol'] = (r.dimension('nspins') == 2)
        p['kpts'] = r.get('BZKPoints')
        p['usesymm'] = r['UseSymmetry']
        p['gpts'] = ((r.dimension('ngptsx') + 1) // 2 * 2,
                     (r.dimension('ngptsy') + 1) // 2 * 2,
                     (r.dimension('ngptsz') + 1) // 2 * 2)
        p['lmax'] = r['MaximumAngularMomentum']
        p['setups'] = r['SetupTypes']
        p['fixdensity'] = r['FixDensity']
        if version <= 0.4:
            # Old version: XXX
            print('# Warning: Reading old version 0.3/0.4 restart files ' +
                  'will be disabled some day in the future!')
            p['convergence']['eigenstates'] = r['Tolerance']
        else:
            p['convergence'] = {'density': r['DensityConvergenceCriterion'],
                                'energy':
                                r['EnergyConvergenceCriterion'] * Hartree,
                                'eigenstates':
                                r['EigenstatesConvergenceCriterion'],
                                'bands': r['NumberOfBandsToConverge']}
            if version <= 0.6:
                mixer = 'Mixer'
                weight = r['MixMetric']
                if weight == 1.0:
                    metric = None
                else:
                    metric = 'old'
            else:
                mixer = r['MixClass']
                weight = r['MixWeight']
                metric = r['MixMetric']

            if mixer == 'Mixer':
                from gpaw.mixer import Mixer
            elif mixer == 'MixerSum':
                from gpaw.mixer import MixerSum as Mixer
            else:
                Mixer = None

            if Mixer is None:
                p['mixer'] = None
            else:
                p['mixer'] = Mixer(r['MixBeta'], r['MixOld'], metric, weight)
            
        if version == 0.3:
            # Old version: XXX
            print('# Warning: Reading old version 0.3 restart files is ' +
                  'dangerous and will be disabled some day in the future!')
            p['stencils'] = (2, 3)
            p['charge'] = 0.0
            p['fixmom'] = False
            self.converged = True
        else:
            p['stencils'] = (r['KohnShamStencil'],
                             r['InterpolationStencil'])
            p['poissonsolver'] = PoissonSolver(nn=r['PoissonStencil'])
            p['charge'] = r['Charge']
            p['fixmom'] = r['FixMagneticMoment']
            self.converged = r['Converged']

        p['width'] = r['FermiWidth'] * Hartree

        try:
            self.error['density'] = r['DensityError']
            self.error['energy'] = r['EnergyError']
            self.error['eigenstates'] = r['EigenstateError'] 
        except (AttributeError, KeyError):
            pass

        pos_ac = r.get('CartesianPositions')
        Z_a = npy.asarray(r.get('AtomicNumbers'), int)
        cell_cc = r.get('UnitCell')
        pbc_c = r.get('BoundaryConditions')
        tag_a = r.get('Tags')
        magmom_a = r.get('MagneticMoments')

        self.atoms = Atoms(positions=pos_ac * Bohr,
                           numbers=Z_a,
                           tags=tag_a,
                           magmoms=magmom_a,
                           cell=cell_cc * Bohr,
                           pbc=pbc_c)
        return r
    
    def check_convergence(self):
        """Check convergence of eigenstates, energy and density."""
        
        # Get convergence criteria and error dictionaries:
        cc = self.input_parameters['convergence']
        er = self.error

        # Eigenstates:
        if cc['bands'] == 'occupied':
            n = self.nvalence
        else:
            n = 2 * cc['bands']
        if n > 0:
            eigenstates_error = self.eigensolver.error / n
        else:
            eigenstates_error = 0.0
        er['eigenstates'] = eigenstates_error

        # Energy:
        if len(self.old_energies) < 3:
            energy_change = 10000.0
        else:
            energy_change = (max(self.old_energies) -
                             min(self.old_energies)) / self.natoms
        er['energy'] = energy_change

        # Density:
        dNt = self.density.mixer.get_charge_sloshing()
        if dNt is None:
            dNt = 10000.0
        elif self.nvalence == 0:
            dNt = 0.0
        else:
            dNt /= self.nvalence
        er['density'] = dNt

        self.converged = ((er['eigenstates'] < cc['eigenstates']) and
                          (er['energy'] * Hartree < cc['energy']) and
                          (er['density'] < cc['density']))
        return self.converged
    
    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if not hasattr(self, 'txt') or self.txt is None:
            return
        
        if hasattr(self, 'timer'):
            self.timer.write(self.txt)

        mr = maxrss()
        if mr > 0:
            self.text('memory  : %.2f MB' % (mr / 1024**2))

    def distribute_cpus(self, parsize_c, parsize_bands, N_c):
        """Distribute k-points/spins to processors.

        Construct communicators for parallelization over
        k-points/spins and for parallelization using domain
        decomposition."""
        
        size = self.world.size
        rank = self.world.rank

        if parsize_bands is None:
            parsize_bands = 1
        self.nmybands = self.nbands // parsize_bands
        if self.nbands != self.nmybands * parsize_bands:
            raise RuntimeError('Cannot distribute %d bands to %d processors' %
                               (self.nbands, parsize_bands))

        ntot = self.nspins * self.nkpts * parsize_bands
        if parsize_c is None:
            ndomains = size // gcd(ntot, size)
        else:
            ndomains = parsize_c[0] * parsize_c[1] * parsize_c[2]

        r0 = (rank // ndomains) * ndomains
        ranks = range(r0, r0 + ndomains)
        domain_comm = self.world.new_communicator(npy.array(ranks))
        self.domain.set_decomposition(domain_comm, parsize_c, N_c)

        r0 = rank % (ndomains * parsize_bands)
        ranks = range(r0, r0 + size, ndomains * parsize_bands)
        self.kpt_comm = self.world.new_communicator(npy.array(ranks))

        r0 = rank % ndomains + self.kpt_comm.rank * (ndomains * parsize_bands)
        ranks = range(r0, r0 + (ndomains * parsize_bands), ndomains)
        self.band_comm = self.world.new_communicator(npy.array(ranks))

        assert (size == domain_comm.size * self.kpt_comm.size *
                self.band_comm.size)

    def initialize(self, atoms=None):
        """Inexpensive initialization."""

        if atoms is None:
            atoms = self.atoms
            
        p = self.input_parameters
        
        self.world = p.get('communicator')
        if self.world is None:
            self.world = mpi.world
        self.master = (self.world.rank == 0)
        
        self.set_text(p['txt'], p['verbose'])
        self.plot_atoms(atoms)
        
        self.natoms = len(atoms)

        pos_ac = atoms.get_positions() / Bohr
        cell_cc = atoms.get_cell() / Bohr
        pbc_c = atoms.get_pbc()
        Z_a = atoms.get_atomic_numbers()
        try:
            magmom_a = atoms.get_magnetic_moments()
            if magmom_a is None:
                print 'Please update ase!'
                raise KeyError
        except KeyError:
            magmom_a = npy.zeros(self.natoms)
        try:
            tag_a = atoms.get_tags()
            if tag_a is None:
                print 'Please update ase!'
                raise KeyError
        except KeyError:
            tag_a = npy.zeros(self.natoms, int)

        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)
        # Get the diagonal:
        cell_c = npy.diagonal(cell_cc)
        
        # Set the scaled k-points:
        kpts = p['kpts']
        if kpts is None:
            self.bzk_kc = npy.zeros((1, 3))
        elif isinstance(kpts[0], int):
            self.bzk_kc = monkhorst_pack(kpts)
        else:
            self.bzk_kc = npy.array(kpts)
        
        magnetic = bool(npy.sometrue(magmom_a))  # numpy!

        self.spinpol = p['spinpol']
        if self.spinpol is None:
            self.spinpol = magnetic
        elif magnetic and not self.spinpol:
            raise ValueError('Non-zero initial magnetic moment for a ' +
                             'spin-paired calculation!')

        self.nspins = 1 + int(self.spinpol)

        self.fixmom = p['fixmom']
        if p['hund']:
            self.fixmom = True
            assert self.natoms == 1
            if not self.spinpol:
                p['hund'] = False
                self.fixmom = False

        self.xcfunc = XCFunctional(p['xc'], self.nspins)
        self.xcfunc.set_timer(self.timer)
        
        if p['gpts'] is not None and p['h'] is None:
            N_c = npy.array(p['gpts'])
        else:
            if p['h'] is None:
                self.text('Using default value for grid spacing.')
                h = 0.2 / Bohr
            else:
                h = p['h'] / Bohr
            # N_c should be a multiple of 4:
            N_c = npy.array([max(4, int(L / h / 4 + 0.5) * 4) for L in cell_c])
        
        # Create a Domain object:
        self.domain = Domain(cell_c, pbc_c)

        # Is this a gamma-point calculation?
        self.gamma = (len(self.bzk_kc) == 1 and
                      not npy.sometrue(self.bzk_kc[0]))

        if self.gamma:
            self.dtype = float
        else:
            self.dtype = complex

        if self.density is None:
            self.reuse_old_density = False
            
        if self.reuse_old_density:
            nt_sG = self.density.nt_sG
            D_asp = {}
            P_auni = {}
            for nucleus in self.my_nuclei:
                D_asp[nucleus.a] = nucleus.D_sp
                if self.kpt_u is not None:
                    P_auni[nucleus.a] = nucleus.P_uni
                
        type_a, basis_a = self.create_nuclei_and_setups(Z_a)

        self.eigensolver = p['eigensolver']
        if isinstance(self.eigensolver, str):
            self.eigensolver = get_eigensolver(self.eigensolver)

        # Brillouin zone stuff:
        if self.gamma:
            self.symmetry = None
            self.weight_k = [1.0]
            self.ibzk_kc = npy.zeros((1, 3))
            self.nkpts = 1
        else:
            # Reduce the the k-points to those in the irreducible part of
            # the Brillouin zone:
            self.symmetry, self.weight_k, self.ibzk_kc = reduce_kpoints(
                self.bzk_kc, pos_ac, Z_a, type_a, magmom_a, basis_a,
                self.domain, p['usesymm'])
            self.nkpts = len(self.ibzk_kc)
        
            if p['usesymm'] and self.symmetry is not None:
                # Find rotation matrices for spherical harmonics:
                R_slmm = [[rotation(l, symm) for l in range(3)]
                          for symm in self.symmetry.symmetries]
        
                for setup in self.setups:
                    setup.calculate_rotations(R_slmm)

        self.kT = p['width']
        if self.kT is None:
            if self.gamma:
                self.kT = 0
            else:
                self.kT = 0.1 / Hartree
        else:
            self.kT /= Hartree
            
        self.initialize_occupation(p['charge'], p['nbands'],
                                   self.kT, p['fixmom'], magmom_a)

        if parsize is not None:  # command-line option
            p['parsize'] = parsize

        if parsize_bands is not None:  # command-line option
            p['parsize_bands'] = parsize_bands

        self.distribute_cpus(p['parsize'], p['parsize_bands'], N_c)
        ## print "My bands", self.nmybands

        self.occupation.set_communicator(self.kpt_comm)

        self.stencils = p['stencils']
        self.maxiter = p['maxiter']

        if p['convergence'].get('bands') == 'all':
            p['convergence']['bands'] = self.nbands

        cbands = p['convergence']['bands']
        if isinstance(cbands, int) and cbands < 0:
            p['convergence']['bands'] += self.nbands
            
        if p['fixdensity'] == True:
            self.fixdensity = self.maxiter + 1000000
            # Density won't converge
            p['convergence']['density'] = 1e8
        else:
            self.fixdensity = p['fixdensity']

        self.random_wf = p['random']

        # Construct grid descriptors for coarse grids (wave functions) and
        # fine grids (densities and potentials):
        self.gd = GridDescriptor(self.domain, N_c)
        self.finegd = GridDescriptor(self.domain, 2 * N_c)

        # Total number of k-point/spin combinations:
        nu = self.nkpts * self.nspins

        # Number of k-point/spin combinations on this cpu:
        self.nmyu = nu // self.kpt_comm.size

        if self.kpt_u is None:
            self.wave_functions_initialized = False
            self.wave_functions_orthonormalized = False
            self.kpt_u = []
            for u in range(self.nmyu):
                s, k = divmod(self.kpt_comm.rank * self.nmyu + u, self.nkpts)
                weight = self.weight_k[k] * 2 / self.nspins
                k_c = self.ibzk_kc[k]
                self.kpt_u.append(KPoint(self.nuclei,
                                         self.gd, weight, s, k, u, k_c,
                                         self.dtype))
        else:
            for kpt in self.kpt_u:
                kpt.set_grid_descriptor(self.gd)
                kpt.nuclei = self.nuclei
                
        cc = p['convergence']  # convergence criteria
        er = self.error        # errors

        if len(er) != 0:
            self.converged = ((er['eigenstates'] < cc['eigenstates']) and
                              (er['energy'] * Hartree < cc['energy']) and
                              (er['density'] < cc['density']))
        
        self.locfuncbcaster = LocFuncBroadcaster(self.kpt_comm)

        self.my_nuclei = []
        self.pt_nuclei = []
        self.ghat_nuclei = []

        self.density = Density(self, magmom_a)#???
        self.hamiltonian = Hamiltonian()
        self.hamiltonian.initialize(self)
        
        self.overlap = Overlap(self)

        self.xcfunc.set_non_local_things(self)

        self.Eref = 0.0
        for nucleus in self.nuclei:
            self.Eref += nucleus.setup.E

        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = self.domain.scale_position(pos_c)
            nucleus.set_position(spos_c, self.domain, self.my_nuclei,
                                 self.nspins, self.nmyu, self.nmybands)
            
        if self.reuse_old_density:
            for a, D_sp in D_asp.items():
                self.nuclei[a].D_sp[:] = D_sp
            for a, P_uni in P_auni.items():
                self.nuclei[a].P_uni[:] = P_uni
            self.density.nt_sG[:] = nt_sG
            #self.density.scale()
            self.density.interpolate_pseudo_density()
            self.density.initialized = True

        self.print_init(pos_ac)

        if dry_run:
            self.print_parameters()
            estimate_memory(self)
            self.txt.flush()
            sys.exit()

        self.initialized = True

    def initialize_occupation(self, charge, nbands, kT, fixmom, magmom_a):
        """Sets number of valence orbitals and initializes occupation."""

        # Sum up the number of valence electrons:
        self.nvalence = 0
        self.nao = 0
        for nucleus in self.nuclei:
            self.nvalence += nucleus.setup.Nv
            self.nao += nucleus.setup.niAO
        self.nvalence -= charge
        
        self.nbands = nbands
        if self.nbands is None:
            self.nbands = self.nao
        elif self.nbands > self.nao and self.eigensolver.lcao:
            raise ValueError('Too many bands for LCAO calculation:' +
                             '%d bands and only %d atomic orbitals!' %
                             (self.nbands, self.nao))
        
        if self.nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                charge)

        # check number of bands ?  XXX
        
        M = magmom_a.sum()

        if self.nbands <= 0:
            self.nbands = int(self.nvalence + M + 0.5) // 2 + (-self.nbands)
        
        if self.nvalence > 2 * self.nbands:
            raise ValueError('Too few bands!')

        # Create object for occupation numbers:
        if kT == 0 or 2 * self.nbands == self.nvalence:
            self.occupation = occupations.ZeroKelvin(self.nvalence,
                                                     self.nspins)
        else:
            self.occupation = occupations.FermiDirac(self.nvalence,
                                                     self.nspins, kT)

        if fixmom:
            self.occupation.fix_moment(M)

        # self.occupation.set_communicator(self.kpt_comm)
        

    def initialize_kinetic(self):
        if not self.hamiltonian.xc.xcfunc.mgga:
            return
        else:
            #pseudo kinetic energy array on 3D grid
            self.density.initialize_kinetic()
            self.hamiltonian.xc.set_kinetic(self.density.taut_sg)

    def update_kinetic(self):
        if not self.hamiltonian.xc.xcfunc.mgga:
            return
        else:
            #pseudo kinetic energy array on 3D grid
            self.density.update_kinetic(self.kpt_u)
            self.hamiltonian.xc.set_kinetic(self.density.taut_sg)           


