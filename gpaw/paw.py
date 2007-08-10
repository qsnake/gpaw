# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import sys
import weakref

import Numeric as num
from ASE import Atom, ListOfAtoms

import gpaw.io
import gpaw.mpi as mpi
import gpaw.occupations as occupations
from gpaw import debug
from gpaw import ConvergenceError
from gpaw.density import Density
from gpaw.eigensolvers import eigensolver
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import Hamiltonian
from gpaw.kpoint import KPoint
from gpaw.localized_functions import LocFuncBroadcaster
from gpaw.utilities.timing import Timer
from gpaw.xc_functional import XCFunctional
from gpaw.mpi import run, new_communicator
from gpaw.brillouin import reduce_kpoints
import _gpaw

MASTER = 0


# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.


"""ASE-calculator interface."""


import os
import sys
import tempfile
import time

import Numeric as num
from ASE.Units import units, Convert
from ASE.Utilities.MonkhorstPack import MonkhorstPack
from ASE.ChemicalElements.symbol import symbols
from ASE.ChemicalElements import numbers
import ASE

from gpaw.utilities import check_unit_cell
from gpaw.utilities.memory import maxrss
from gpaw.version import version
import gpaw.utilities.timing as timing
import gpaw
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
from gpaw import dry_run


MASTER = 0


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
    ``typecode``    Data type of wave functions (``Float`` or
                    ``Complex``).
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
    ``typecode``    Data type of wave functions (``Float`` or
                    ``Complex``).
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
    ``kpt_comm``    MPI-communicator for parallelization over
                    **k**-points.
    =============== ===================================================
    """

    def __init__(self, filename=None, **kwargs):
        """ASE-calculator interface.

        The following parameters can be used: `nbands`, `xc`, `kpts`,
        `spinpol`, `gpts`, `h`, `charge`, `usesymm`, `width`, `mix`,
        `hund`, `lmax`, `fixdensity`, `tolerance`, `txt`,
        `hosts`, `parsize`, `softgauss`, `stencils`, and
        `convergeall`.

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

        self.input_parameters = {
            'h':             None,
            'xc':            'LDA',
            'gpts':          None,
            'kpts':          None,
            'lmax':          2,
            'charge':        0,
            'fixmom':        False,
            'nbands':        None,
            'setups':        'paw',
            'width':         None,
            'spinpol':       None,
            'usesymm':       True,
            'stencils':      (2, 'M', 3),
            'tolerance':     1.0e-9,
            'fixdensity':    False,
            'convergeall':   False,
            'mix':           (0.25, 3, 1.0),
            'txt':           '-',
            'hund':          False,
            'random':        False,
            'maxiter':       120,
            'parsize':       None,
            'external':      None,
            'decompose':     None,
            'verbose':       0,
            'eigensolver':   'rmm-diis',
            'poissonsolver': 'GS'}

        self.converged = False
        self.initialized = False
        self.wave_functions_initialized = False
        self.callback_functions = []
        
        if filename is not None:
            reader = self.read_parameters(filename)

        if 'h' in kwargs and 'gpts' in kwargs:
            raise TypeError("""You can't use both "gpts" and "h"!""")
            
        for name, value in kwargs.items():
            if name in ['parsize',
                        'random', 'hund', 'mix', 'txt', 'maxiter', 'verbose',
                        'decompose', 'eigensolver', 'poissonsolver',
                        'external']:
                self.input_parameters[name] = value
            elif name in ['xc', 'nbands', 'spinpol', 'kpts', 'usesymm',
                          'gpts', 'h', 'width', 'lmax', 'setups', 'stencils',
                          'charge', 'fixmom', 'fixdensity', 'tolerance',
                          'convergeall']:
                self.converged = False
                self.input_parameters[name] = value
            else:
                raise RuntimeError('Unknown keyword: ' + name)

        Output.__init__(self)
        
        if filename is not None:
            self.initialize()
            gpaw.io.read(self, reader)
            self.plot_atoms()

    def set(self, **kwargs):
        self.convert_units(kwargs)
        self.initialized = False
        self.wave_functions_initialized = False
        self.input_parameters.update(kwargs)
                
    def calculate(self):
        """Update PAW calculaton if needed."""

        if not self.initialized:
            self.initialize()
            self.find_ground_state()
            return
        
        atoms = self.atoms
        if self.lastcount == atoms.GetCount():
            # Nothing to do:
            return

        pos_ac, Z_a, cell_cc, pbc_c = self.last_atomic_configuration

        if (atoms.GetAtomicNumbers() != Z_a or
            atoms.GetUnitCell() / self.a0 != cell_cc or
            atoms.GetBoundaryConditions() != pbc_c):
            # Drastic changes:
            self.initialize()
            self.find_ground_state()
            return

        # Something else has changed:
        if atoms.GetCartesianPositions() / self.a0 != pos_ac:
            # It was the positions:
            self.find_ground_state()
        else:
            # It was something that we don't care about - like
            # velocities, masses, ...
            pass
        
    def get_atoms(self):
        atoms = self.atoms
        assert isinstance(atoms, ListOfAtoms)
        self.atoms = weakref.proxy(atoms)
        atoms.calculator = self
        return atoms

    def find_ground_state(self):
        """Start iterating towards the ground state."""

        self.set_positions()

        if not self.wave_functions_initialized:
            self.initialize_wave_functions()

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
            self.call()

        self.call(final=True)
        self.print_converged()

    def step(self):
        if not self.fixdensity and self.niter > 2:
            self.density.update(self.kpt_u, self.symmetry)
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

    def set_positions(self):
        """Update the positions of the atoms.

        Localized functions centered on atoms that have moved will
        have to be computed again.  Neighbor list is updated and the
        array holding all the pseudo core densities is updated."""

        # Save the state of the atoms:
        atoms = self.atoms
        pos_ac = atoms.GetCartesianPositions() / self.a0
        self.lastcount = atoms.GetCount()
        self.last_atomic_configuration = (
            pos_ac,
            atoms.GetAtomicNumbers(),
            atoms.GetUnitCell() / self.a0,
            atoms.GetBoundaryConditions())

        movement = False
        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = self.domain.scale_position(pos_c)
            if num.sometrue(spos_c != nucleus.spos_c) or not nucleus.ready:
                movement = True
                nucleus.set_position(spos_c, self.domain, self.my_nuclei,
                                     self.nspins, self.nmyu, self.nbands)
                nucleus.move(spos_c, self.gd, self.finegd,
                             self.ibzk_kc, self.locfuncbcaster,
                             self.domain,
                             self.pt_nuclei, self.ghat_nuclei)

        if movement:
            self.niter = 0
            self.converged = False
            self.F_ac = None

            self.locfuncbcaster.broadcast()

            for nucleus in self.nuclei:
                nucleus.normalize_shape_function_and_pseudo_core_density()

            if self.symmetry:
                self.symmetry.check(pos_ac)

            self.hamiltonian.pairpot.update(pos_ac, self.nuclei, self.domain)

            self.density.move()

    def initialize_wave_functions_from_atomic_orbitals(self):
        """Initialize wave function from atomic orbitals."""  # surprise!
        
        # count the total number of atomic orbitals (bands):
        nao = 0
        for nucleus in self.nuclei:
            nao += nucleus.get_number_of_atomic_orbitals()

        if self.random_wf:
            nao = 0

        nrandom = max(0, self.nbands - nao)

        if self.nbands == 1:
            string = 'Initializing one band from'
        else:
            string = 'Initializing %d bands from' % self.nbands
        if nao == 1:
            string += ' one atomic orbital'
        elif nao > 0:
            string += ' linear combination of %d atomic orbitals' % nao

        if nrandom > 0 :
            if nao > 0:
                string += ' and'
            string += ' %d random orbitals' % nrandom
        string += '.'

        self.text(string)

        xcfunc = self.hamiltonian.xc.xcfunc

        if xcfunc.hybrid > 0.0:
            # At this point, we can't use orbital dependent
            # functionals, because we don't have the right orbitals
            # yet.  So we use a simple density functional to set up the
            # initial hamiltonian:
            if xcfunc.xcname == 'EXX':
                localxcfunc = XCFunctional('LDAx', self.nspins)
            else:
                assert xcfunc.xcname == 'PBE0'
                localxcfunc = XCFunctional('PBE', self.nspins)
            self.hamiltonian.xc.set_functional(localxcfunc)
            for setup in self.setups:
                setup.xc_correction.xc.set_functional(localxcfunc)

        self.hamiltonian.update(self.density)

        if self.random_wf:
            # Improve the random guess with conjugate gradient
            eig = eigensolver('cg', self, convergeall=True)
            for kpt in self.kpt_u:
                kpt.create_random_orbitals(self.nbands)
                # Calculate projections and orthogonalize wave functions:
                run([nucleus.calculate_projections(kpt)
                     for nucleus in self.pt_nuclei])
                kpt.orthonormalize(self.my_nuclei)
            for nit in range(2):
                eig.iterate(self.hamiltonian, self.kpt_u)
        else:
            for nucleus in self.my_nuclei:
                # XXX already allocated once, but with wrong size!!!
                ni = nucleus.get_number_of_partial_waves()
                nucleus.P_uni = num.empty((self.nmyu, nao, ni), self.typecode)

            # Use the generic eigensolver for subspace diagonalization
            eig = Eigensolver(self, nao)
            for kpt in self.kpt_u:
                kpt.create_atomic_orbitals(nao, self.nuclei)
                # Calculate projections and orthogonalize wave functions:
                run([nucleus.calculate_projections(kpt)
                     for nucleus in self.pt_nuclei])
                kpt.orthonormalize(self.my_nuclei)
                eig.diagonalize(self.hamiltonian, kpt)


        for nucleus in self.my_nuclei:
            nucleus.reallocate(self.nbands)

        for kpt in self.kpt_u:
            kpt.adjust_number_of_bands(self.nbands,
                                       self.pt_nuclei, self.my_nuclei)

        if xcfunc.hybrid > 0:
            # Switch back to the orbital dependent functional:
            self.hamiltonian.xc.set_functional(xcfunc)
            for setup in self.setups:
                setup.xc_correction.xc.set_functional(xcfunc)

        # Calculate occupation numbers:
        self.occupation.calculate(self.kpt_u)

        self.wave_functions_initialized = True


    def initialize_wave_functions(self):
        if not self.wave_functions_initialized:
            # Initialize wave functions from atomic orbitals:
            for nucleus in self.nuclei:
                nucleus.initialize_atomic_orbitals(self.gd, self.ibzk_kc,
                                                   self.locfuncbcaster)
            self.locfuncbcaster.broadcast()

            if not self.density.initialized:
                self.density.initialize()

            self.initialize_wave_functions_from_atomic_orbitals()

            self.converged = False

            # Free allocated space for radial grids:
            for setup in self.setups:
                del setup.phit_j
            for nucleus in self.nuclei:
                try:
                    del nucleus.phit_i
                except AttributeError:
                    pass

        elif not isinstance(self.kpt_u[0].psit_nG, num.ArrayType):
            # Calculation started from a restart file.  Copy data
            # from the file to memory:
            for kpt in self.kpt_u:
                kpt.psit_nG = kpt.psit_nG[:]

        for kpt in self.kpt_u:
            kpt.adjust_number_of_bands(self.nbands, self.pt_nuclei,
                                       self.my_nuclei)


    def calculate_forces(self):
        """Return the atomic forces."""

        if self.F_ac is not None:
            return

        self.F_ac = num.empty((self.natoms, 3), num.Float)
        
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
        if mpi.rank == MASTER:
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
        mpi.world.broadcast(self.F_ac, MASTER)

        if self.symmetry is not None:
            # Symmetrize forces:
            F_ac = num.zeros((self.natoms, 3), num.Float)
            for map_a, symmetry in zip(self.symmetry.maps,
                                       self.symmetry.symmetries):
                swap, mirror = symmetry
                for a1, a2 in enumerate(map_a):
                    F_ac[a2] += num.take(self.F_ac[a1] * mirror, swap)
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
        
        # setup_types is a dictionary mapping chemical symbols and atom
        # numbers to setup types.
        
        # If present, None will map to the default type:
        default = setup_types.get(None, 'paw')
        
        type_a = [default] * self.natoms
        
        # First symbols ...
        for symbol, type in setup_types.items():
            if isinstance(symbol, str):
                number = numbers[symbol]
                for a, Z in enumerate(Z_a):
                    if Z == number:
                        type_a[a] = type
        
        # and then atom numbers:
        for a, type in setup_types.items():
            if isinstance(a, int):
                type_a[a] = type
        
        # Build list of nuclei and construct necessary PAW-setup objects:
        self.nuclei = []
        setups = {}
        for a, (Z, type) in enumerate(zip(Z_a, type_a)):
            if (Z, type) in setups:
                setup = setups[(Z, type)]
            else:
                symbol = symbols[Z]
                setup = create_setup(symbol, self.xcfunc, p['lmax'],
                                     self.nspins, type)
                setup.print_info(self.text)
                setups[(Z, type)] = setup
            self.nuclei.append(Nucleus(setup, a, self.typecode))

        self.setups = setups.values()
        return type_a

    def read_parameters(self, filename):
        """Read state from file."""

        r = gpaw.io.open(filename, 'r')
        p = self.input_parameters

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
        p['stencils'] = (r['KohnShamStencil'],
                         r['PoissonStencil'],
                         r['InterpolationStencil'])
        p['charge'] = r['Charge']
        p['fixmom'] = r['FixMagneticMoment']
        p['fixdensity'] = r['FixDensity']
        p['tolerance'] = r['Tolerance']
        p['convergeall'] = r['ConvergeEmptyStates']
        p['width'] = r['FermiWidth'] 

        pos_ac = r.get('CartesianPositions')
        Z_a = num.asarray(r.get('AtomicNumbers'), num.Int)
        cell_cc = r.get('UnitCell')
        pbc_c = r.get('BoundaryConditions')
        tag_a = r.get('Tags')
        magmom_a = r.get('MagneticMoments')

        self.last_atomic_configuration = (pos_ac, Z_a, cell_cc, pbc_c)
        self.extra_list_of_atoms_stuff = (magmom_a, tag_a)

        self.atoms = ListOfAtoms([Atom(position=pos_c * self.a0, Z=Z,
                                       tag=tag, magmom=magmom)
                                  for pos_c, Z, tag, magmom in
                                  zip(pos_ac, Z_a, tag_a, magmom_a)],
                                 cell=cell_cc * self.a0, periodic=pbc_c)
        self.lastcount = self.atoms.GetCount()

        self.converged = r['Converged']

        return r
    
    def reset(self, restart_file=None):
        """Delete PAW-object."""
        self.stop_paw()
        self.restart_file = restart_file
        self.pos_ac = None
        self.cell_cc = None
        self.periodic_c = None
        self.Z_a = None

    def set_h(self, h):
        self.gpts = None
        self.h = h
        self.reset()
     
    def set_gpts(self, gpts):
        self.h = None
        self.gpts = gpts
        self.reset()
     
    
    def set_convergence_criteria(self, tol):
        """Set convergence criteria.

        Stop iterating when the size of the residuals are below
        ``tol``."""

        self.tolerance = tol

    def check_convergence(self):
        self.error = self.eigensolver.error
        if self.input_parameters['convergeall']:
            self.error /= 2 * self.nbands
        else:
            if self.nvalence == 0:
                self.error = self.tolerance
            else:
                self.error /= self.nvalence
        
        self.converged = (self.error <= self.tolerance)
        return self.converged
    
    def __del__(self):
        """Destructor:  Write timing output before closing."""
        if hasattr(self, 'timer'):
            self.timer.write(self.txt)

        mr = maxrss()
        if mr > 0:
            self.text('memory  : %.2f MB' % (mr / 1024**2))

    def distribute_kpoints_and_spins(self, parsize_c, N_c):
        """Distribute k-points/spins to processors.

        Construct communicators for parallelization over
        k-points/spins and for parallelization using domain
        decomposition."""
        
        ntot = self.nspins * self.nkpts
        size = mpi.size
        rank = mpi.rank

        if parsize_c is None:
            ndomains = size // gcd(ntot, size)
        else:
            ndomains = parsize_c[0] * parsize_c[1] * parsize_c[2]

        r0 = (rank // ndomains) * ndomains
        ranks = range(r0, r0 + ndomains)
        domain_comm = new_communicator(ranks)
        self.domain.set_decomposition(domain_comm, parsize_c, N_c)

        r0 = rank % ndomains
        ranks = range(r0, r0 + size, ndomains)
        self.kpt_comm = new_communicator(ranks)

    def initialize(self):
        """Inexpensive initialization."""
        self.timer = Timer()
        self.timer.start('Init')

        self.kpt_u = None
        
        atoms = self.atoms
        self.natoms = len(atoms)
        magmom_a = atoms.GetMagneticMoments()
        pos_ac = atoms.GetCartesianPositions() / self.a0
        cell_cc = atoms.GetUnitCell() / self.a0
        pbc_c = atoms.GetBoundaryConditions()
        Z_a = atoms.GetAtomicNumbers()
        
        # Check that the cell is orthorhombic:
        check_unit_cell(cell_cc)
        # Get the diagonal:
        cell_c = num.diagonal(cell_cc)
        
        p = self.input_parameters
        
        # Set the scaled k-points:
        kpts = p['kpts']
        if kpts is None:
            self.bzk_kc = num.zeros((1, 3), num.Float)
        elif isinstance(kpts[0], int):
            self.bzk_kc = MonkhorstPack(kpts)
        else:
            self.bzk_kc = num.array(kpts)
        
        magnetic = bool(num.sometrue(magmom_a))  # numpy!

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
            assert self.spinpol and self.natoms == 1

        if self.fixmom:
            assert self.spinpol

        self.xcfunc = XCFunctional(p['xc'], self.nspins)
        
        if p['gpts'] is not None and p['h'] is None:
            N_c = num.array(p['gpts'])
        else:
            if p['h'] is None:
                self.text('Using default value for grid spacing.')
                h = Convert(0.2, 'Ang', 'Bohr')
            else:
                h = p['h']
            # N_c should be a multiplum of 4:
            N_c = num.array([max(4, int(L / h / 4 + 0.5) * 4) for L in cell_c])
        
        # Create a Domain object:
        self.domain = Domain(cell_c, pbc_c)

        # Is this a gamma-point calculation?
        self.gamma = (len(self.bzk_kc) == 1 and
                      not num.sometrue(self.bzk_kc[0]))

        if self.gamma:
            self.typecode = num.Float
        else:
            self.typecode = num.Complex
            
        type_a = self.create_nuclei_and_setups(Z_a)

        # Brillouin zone stuff:
        if self.gamma:
            self.symmetry = None
            self.weight_k = [1.0]
            self.ibzk_kc = num.zeros((1, 3), num.Float)
            self.nkpts = 1
        else:
            # Reduce the the k-points to those in the irreducible part of
            # the Brillouin zone:
            self.symmetry, self.weight_k, self.ibzk_kc = reduce_kpoints(
                self.bzk_kc, pos_ac, Z_a, type_a, magmom_a, self.domain,
                p['usesymm'])
            self.nkpts = len(self.ibzk_kc)
        
            if p['usesymm'] and self.symmetry is not None:
                # Find rotation matrices for spherical harmonics:
                R_slmm = [[rotation(l, symm) for l in range(3)]
                          for symm in self.symmetry.symmetries]
        
                for setup in self.setups:
                    setup.calculate_rotations(R_slmm)
        
        self.distribute_kpoints_and_spins(p['parsize'], N_c)
        
        if dry_run:
            # Estimate the amount of memory needed:
            estimate_memory(N_c, nbands, nkpts, nspins, typecode, nuclei, h_c,
                            text)
            self.txt.flush()
            sys.exit()

        # Sum up the number of valence electrons:
        self.nvalence = 0
        nao = 0
        for nucleus in self.nuclei:
            self.nvalence += nucleus.setup.Nv
            nao += nucleus.setup.niAO
        self.nvalence -= p['charge']
        
        if self.nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                p['charge'])
        
        self.nbands = p['nbands']
        if self.nbands is None:
            self.nbands = nao
        elif self.nbands <= 0:
            self.nbands = (self.nvalence + 1) // 2 + (-self.nbands)
            
        if self.nvalence > 2 * self.nbands:
            raise ValueError('Too few bands!')

        self.kT = p['width']
        if self.kT is None:
            if self.gamma:
                self.kT = 0
            else:
                self.kT = Convert(0.1, 'eV', 'Hartree')
        
        self.stencils = p['stencils']
        self.maxiter = p['maxiter']
        self.tolerance = p['tolerance']
        self.fixdensity = p['fixdensity']
        self.random_wf = p['random']

        # Construct grid descriptors for coarse grids (wave functions) and
        # fine grids (densities and potentials):
        self.gd = GridDescriptor(self.domain, N_c)
        self.finegd = GridDescriptor(self.domain, 2 * N_c)

        self.F_ac = None

        # Total number of k-point/spin combinations:
        nu = self.nkpts * self.nspins

        # Number of k-point/spin combinations on this cpu:
        self.nmyu = nu // self.kpt_comm.size

        self.kpt_u = []
        for u in range(self.nmyu):
            s, k = divmod(self.kpt_comm.rank * self.nmyu + u, self.nkpts)
            weight = self.weight_k[k] * 2 / self.nspins
            k_c = self.ibzk_kc[k]
            self.kpt_u.append(KPoint(self.gd, weight, s, k, u, k_c,
                                     self.typecode))

        self.locfuncbcaster = LocFuncBroadcaster(self.kpt_comm)

        self.my_nuclei = []
        self.pt_nuclei = []
        self.ghat_nuclei = []

        self.density = Density(self, magmom_a)#???
        self.hamiltonian = Hamiltonian(self)

        # Create object for occupation numbers:
        if self.kT == 0 or 2 * self.nbands == self.nvalence:
            self.occupation = occupations.ZeroKelvin(self.nvalence,
                                                     self.nspins)
        else:
            self.occupation = occupations.FermiDirac(self.nvalence,
                                                     self.nspins, self.kT)

        if p['fixmom']:
            M = sum(magmom_a)
            self.occupation.fix_moment(M)

        self.occupation.set_communicator(self.kpt_comm)

        self.xcfunc.set_non_local_things(self)

        self.Eref = 0.0
        for nucleus in self.nuclei:
            self.Eref += nucleus.setup.E

        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = self.domain.scale_position(pos_c)
            nucleus.set_position(spos_c, self.domain, self.my_nuclei,
                                 self.nspins, self.nmyu, self.nbands)
            
        self.print_init(pos_ac)
        self.eigensolver = eigensolver(p['eigensolver'], self)
        self.initialized = True
        self.timer.stop('Init')
