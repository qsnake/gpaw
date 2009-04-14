# -*- coding: utf-8 -*-
# Copyright (C) 2003-2007  CAMP
# Copyright (C) 2007-2008  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import numpy as np
from ase.units import Bohr, Hartree
from ase.dft import monkhorst_pack

import gpaw.io
import gpaw.mpi as mpi
import gpaw.occupations as occupations
from gpaw import dry_run, KohnShamConvergenceError
from gpaw.density import Density
from gpaw.eigensolvers import get_eigensolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import Hamiltonian
from gpaw.utilities.timing import Timer
from gpaw.xc_functional import XCFunctional
from gpaw.brillouin import reduce_kpoints
from gpaw.wavefunctions import GridWaveFunctions, LCAOWaveFunctions
from gpaw.wavefunctions import EmptyWaveFunctions
from gpaw.utilities.memory import MemNode, memory
from gpaw.utilities import gcd
from gpaw.parameters import InputParameters
from gpaw.setup import Setups
from gpaw.output import PAWTextOutput
from gpaw.scf import SCFLoop
from gpaw.forces import ForceCalculator

class PAW(PAWTextOutput):
    """This is the main calculation object for doing a PAW calculation."""

    timer_class = Timer
    scf_loop_class = SCFLoop

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

        PAWTextOutput.__init__(self)
        self.input_parameters = InputParameters()
        self.timer = self.timer_class()
        self.scf = None
        self.forces = ForceCalculator(self.timer)
        self.wfs = EmptyWaveFunctions()
        self.occupations = None
        self.density = None
        self.hamiltonian = None
        self.atoms = None
        self.gd = None
        self.finegd = None

        self.initialized = False

        if filename is not None:
            reader = gpaw.io.open(filename, 'r')
            self.atoms = gpaw.io.read_atoms(reader)
            self.input_parameters.read(reader)
            self.input_parameters.txt = kwargs.pop('txt', '-')
            self.input_parameters.idiotproof = kwargs.pop('idiotproof', True)
            self.initialize()
            self.read(reader)
            
        self.set(**kwargs)

        if filename is not None and not self.initialized: # TODO last condition is redundant
            self.initialize()
            self.print_cell_and_parameters()
                
        self.observers = []

    def read(self, reader):
        gpaw.io.read(self, reader)

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

        self.initialized = False

        for key in kwargs:
            if key in ['fixmom', 'mixer', 'basis',
                       'verbose', 'txt', 'hund', 'random',
                       'eigensolver', 'poissonsolver', 'idiotproof', 'notify']:
                continue
                
            if key in ['convergence', 'fixdensity', 'maxiter']:
                self.scf = None
                continue
                
            # More drastic changes:
            self.scf = None
            self.wfs.set_orthonormalized(False)
            if key in ['lmax', 'width', 'stencils', 'external', 'xc']:
                self.hamiltonian = None
                self.occupations = None
            elif key in ['charge']:
                self.hamiltonian = None
                self.density = None
            elif key in ['kpts', 'nbands']:
                self.wfs = EmptyWaveFunctions()
                self.occupations = None
            elif key in ['h', 'gpts', 'setups', 'spinpol',
                         'usesymm', 'parsize', 'parsize_bands',
                         'communicator']:
                self.density = None
                self.occupations = None
                self.hamiltonian = None
                self.wfs = EmptyWaveFunctions()
            elif key in ['mode']:
                self.wfs = EmptyWaveFunctions()
            else:
                raise TypeError('Unknown keyword argument:' + key)
         
        p.update(kwargs)

    def calculate(self, atoms=None, converge=False,
                  force_call_to_set_positions=False):
        """Update PAW calculaton if needed."""

        if atoms is None:
            atoms = self.atoms

        if self.atoms is None:
            # First time:
            self.initialize(atoms)
            self.set_positions(atoms)
        elif (len(atoms) != len(self.atoms) or
              (atoms.get_atomic_numbers() !=
               self.atoms.get_atomic_numbers()).any() or
              (atoms.get_initial_magnetic_moments() !=
               self.atoms.get_initial_magnetic_moments()).any() or
              (atoms.get_cell() != self.atoms.get_cell()).any() or
              (atoms.get_pbc() != self.atoms.get_pbc()).any()):
            # Drastic changes:
            self.wfs = EmptyWaveFunctions()
            self.occupations = None
            self.density = None
            self.hamiltonian = None
            self.scf = None
            self.initialize(atoms)
            self.set_positions(atoms)
        elif not self.initialized:
            self.initialize(atoms)
            self.set_positions(atoms)
        elif (atoms.get_positions() != self.atoms.get_positions()).any():
            self.density.reset()
            self.set_positions(atoms)
        elif not self.scf.check_convergence(self.density,
                                            self.wfs.eigensolver):
            self.set_positions(atoms)
        elif force_call_to_set_positions:
            self.set_positions(atoms)
            
        if self.scf.converged:
            return
        else:
            self.print_cell_and_parameters()

        for iter in self.scf.run(self.wfs, self.hamiltonian, self.density,
                                 self.occupations):
            self.call_observers(iter)
            self.print_iteration(iter)
            
        if self.scf.converged:
            self.call_observers(iter, final=True)
            self.print_converged(iter)
        elif converge:
            raise KohnShamConvergenceError('Did not converge!')        

    def initialize_positions(self, atoms=None):
        """Update the positions of the atoms."""
        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()

        spos_ac = atoms.get_scaled_positions() % 1.0

        self.wfs.set_positions(spos_ac)
        self.density.set_positions(spos_ac, self.wfs.rank_a)
        self.hamiltonian.set_positions(spos_ac, self.wfs.rank_a)

        return spos_ac
    
    def set_positions(self, atoms=None):
        """Update the positions of the atoms and initialize wave functions."""
        spos_ac = self.initialize_positions(atoms)
        self.wfs.initialize(self.density, self.hamiltonian, spos_ac)
        self.scf.reset()
        self.forces.reset()
        self.print_positions()

    def initialize(self, atoms=None):
        """Inexpensive initialization."""

        if atoms is None:
            atoms = self.atoms
        else:
            # Save the state of the atoms:
            self.atoms = atoms.copy()
            
        par = self.input_parameters
        
        world = par.communicator
        if world is None:
            world = mpi.world
        self.wfs.world = world
        
        self.set_text(par.txt, par.verbose)

        natoms = len(atoms)

        pos_av = atoms.get_positions() / Bohr
        cell_cv = atoms.get_cell() / Bohr
        pbc_c = atoms.get_pbc()
        Z_a = atoms.get_atomic_numbers()
        magmom_a = atoms.get_initial_magnetic_moments()
        
        # Set the scaled k-points:
        kpts = par.kpts
        if kpts is None:
            bzk_kc = np.zeros((1, 3))
        elif isinstance(kpts[0], int):
            bzk_kc = monkhorst_pack(kpts)
        else:
            bzk_kc = np.array(kpts)
        
        magnetic = magmom_a.any()

        spinpol = par.spinpol
        fixmom = par.fixmom
        if par.hund:
            if natoms != 1:
                raise ValueError('hund=True arg only valid for single atoms!')
            fixmom = True
            spinpol = True

        if spinpol is None:
            spinpol = magnetic
        elif magnetic and not spinpol:
            raise ValueError('Non-zero initial magnetic moment for a '
                             'spin-paired calculation!')

        nspins = 1 + int(spinpol)

        if par.gpts is not None and par.h is None:
            N_c = np.array(par.gpts)
        else:
            if par.h is None:
                self.text('Using default value for grid spacing.')
                h = 0.2 / Bohr
            else:
                h = par.h / Bohr
            # N_c should be a multiple of 4:
            N_c = []
            for axis_v in cell_cv:
                L = (axis_v**2).sum()**0.5
                N_c.append(max(4, int(L / h / 4 + 0.5) * 4))
            N_c = np.array(N_c)
                       
        # Is this a gamma-point calculation?
        gamma = len(bzk_kc) == 1 and not bzk_kc[0].any()

        if hasattr(self, 'time'):
            dtype = complex
        else:
            if gamma:
                dtype = float
            else:
                dtype = complex

        if isinstance(par.xc, (str, dict)):
            xcfunc = XCFunctional(par.xc, nspins)
        else:
            xcfunc = par.xc

        setups = Setups(Z_a, par.setups, par.basis, nspins, par.lmax, xcfunc)

        # Brillouin zone stuff:
        if gamma:
            symmetry = None
            weight_k = np.array([1.0])
            ibzk_kc = np.zeros((1, 3))
        else:
            # Reduce the the k-points to those in the irreducible part of
            # the Brillouin zone:
            symmetry, weight_k, ibzk_kc = reduce_kpoints(atoms, bzk_kc,
                                                         setups, par.usesymm)

        width = par.width
        if width is None:
            if gamma:
                width = 0
            else:
                width = 0.1 / Hartree
        else:
            width /= Hartree
            
        nao = setups.nao
        self.nvalence = nvalence = setups.nvalence - par.charge
        
        nbands = par.nbands
        if nbands is None:
            nbands = nao
        elif nbands > nao and par.mode == 'lcao':
            raise ValueError('Too many bands for LCAO calculation: ' +
                             '%d bands and only %d atomic orbitals!' %
                             (nbands, nao))
        
        if nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                par.charge)

        M = magmom_a.sum()
        if par.hund:
            f_si = setups[0].calculate_initial_occupation_numbers(
                magmom=0, hund=True, charge=par.charge)
            Mh = -np.diff(f_si.sum(1))
            if magnetic and M != Mh:
                raise RuntimeError('You specified a magmom that does not'
                                   'agree with hunds rule!')
            else:
                M = Mh

        if nbands <= 0:
            nbands = int(nvalence + M + 0.5) // 2 + (-nbands)
        
        if nvalence > 2 * nbands:
            raise ValueError('Too few bands!')

        if self.occupations is None:
            # Create object for occupation numbers:
            if width == 0 or 2 * nbands == nvalence:
                self.occupations = occupations.ZeroKelvin(nvalence, nspins)
            else:
                self.occupations = occupations.FermiDirac(nvalence, nspins,
                                                          width)

        self.occupations.magmom = M
        if fixmom:
            self.occupations.fix_moment(M)

        from gpaw import parsize
        if parsize is None:
            parsize = par.parsize

        from gpaw import parsize_bands
        if parsize_bands is None:
            parsize_bands = par.parsize_bands

        if nbands % parsize_bands != 0:
            raise RuntimeError('Cannot distribute %d bands to %d processors' %
                               (nbands, parsize_bands))
        mynbands = nbands // parsize_bands

        cc = par.convergence

        # Number of bands to converge:
        nbands_converge = cc['bands']
        if nbands_converge == 'all':
            nbands_converge = nbands
        elif nbands_converge < 0:
            nbands_converge += nbands

        if par.mode == 'lcao':
            niter_fixdensity = 0
        else:
            niter_fixdensity = 2

        if self.scf is None:
            self.scf = self.scf_loop_class(cc['eigenstates'] * nvalence, 
                                           cc['energy'] / Hartree * natoms,
                                           cc['density'] * nvalence,
                                           par.maxiter, par.fixdensity,
                                           niter_fixdensity)
        
        if not self.wfs:
            domain_comm, kpt_comm, band_comm = self.distribute_cpus(
                world, parsize, parsize_bands, nspins, len(ibzk_kc))

            if self.gd is not None and self.gd.comm.size != domain_comm.size:
                # Domain decomposition has changed, so we need to
                # reinitialize density and hamiltonian:
                if par.fixdensity:
                    raise RuntimeError("I'm confused - please specify parsize."
                                       )
                self.density = None
                self.hamiltonian = None

            # Construct grid descriptor for coarse grids for wave functions:
            self.gd = GridDescriptor(N_c, cell_cv, pbc_c, domain_comm, parsize)

            # do k-point analysis here? XXX

            args = (self.gd, nspins, setups,
                    nbands, mynbands,
                    dtype, world, kpt_comm, band_comm,
                    gamma, bzk_kc, ibzk_kc, weight_k, symmetry, self.timer)
            if par.mode == 'lcao':
                self.wfs = LCAOWaveFunctions(*args)
            else:
                self.wfs = GridWaveFunctions(par.stencils[0], *args)
        else:
            self.wfs.set_setups(setups)

        self.occupations.set_communicator(self.wfs.kpt_comm,
                                          self.wfs.band_comm)
        
        if not self.wfs.eigensolver:
            eigensolver = get_eigensolver(par.eigensolver, par.mode,
                                          par.convergence)
            eigensolver.nbands_converge = nbands_converge
            # XXX Eigensolver class doesn't define an nbands_converge property
            self.wfs.set_eigensolver(eigensolver)

        if self.density is None:
            if par.stencils[1] != 9:
                # Construct grid descriptor for fine grids for densities
                # and potentials:
                self.finegd = self.gd.refine()
            else:
                # Special case (use only coarse grid):
                self.finegd = self.gd
                
            self.density = Density(self.gd, self.finegd, nspins,
                                   par.charge + setups.core_charge)

        self.density.initialize(setups, par.stencils[1], self.timer,
                                magmom_a, par.hund)
        self.density.set_mixer(par.mixer, fixmom, width)

        if self.hamiltonian is None:
            self.hamiltonian = Hamiltonian(self.gd, self.finegd, nspins,
                                           setups, par.stencils[1], self.timer,
                                           xcfunc, par.poissonsolver,
                                           par.external)

        xcfunc.set_non_local_things(self.density, self.hamiltonian, self.wfs,
                                    self.atoms)

        # For gllb releated calculations, the required parameters (wfs, etc.)
        # are obtained using paw object
        if xcfunc.gllb:
            xcfunc.initialize_gllb(self)

        if dry_run:
            self.print_cell_and_parameters()
            self.print_memory_estimate(self.txt, maxdepth=2)
            self.txt.flush()
            raise SystemExit

        self.initialized = True


    def attach(self, function, n, *args, **kwargs):
        """Register observer function.

        Call *function* every *n* iterations using *args* and
        *kwargs* as arguments."""

        try:
            slf = function.im_self
        except AttributeError:
            pass
        else:
            if slf is self:
                # function is a bound method of self.  Store the name
                # of the method and avoid circular reference:
                function = function.im_func.func_name
                
        self.observers.append((function, n, args, kwargs))

    def call_observers(self, iter, final=False):
        """Call all registered callback functions."""
        for function, n, args, kwargs in self.observers:
            if ((iter % n) == 0) != final:
                if isinstance(function, str):
                    function = getattr(self, function)
                function(*args, **kwargs)
                
    def distribute_cpus(self, world,
                        parsize_c, parsize_bands, nspins, nibzkpts):
        """Distribute k-points/spins to processors.

        Construct communicators for parallelization over
        k-points/spins and for parallelization using domain
        decomposition."""
        
        size = world.size
        rank = world.rank

        ntot = nspins * nibzkpts * parsize_bands
        if parsize_c is None:
            ndomains = size // gcd(ntot, size)
        else:
            ndomains = parsize_c[0] * parsize_c[1] * parsize_c[2]

        r0 = (rank // ndomains) * ndomains
        ranks = np.arange(r0, r0 + ndomains)
        domain_comm = world.new_communicator(ranks)

        r0 = rank % (ndomains * parsize_bands)
        ranks = np.arange(r0, r0 + size, ndomains * parsize_bands)
        kpt_comm = world.new_communicator(ranks)

        r0 = rank % ndomains + kpt_comm.rank * (ndomains * parsize_bands)
        ranks = np.arange(r0, r0 + (ndomains * parsize_bands), ndomains)
        band_comm = world.new_communicator(ranks)

        assert size == domain_comm.size * kpt_comm.size * band_comm.size
        assert nspins * nibzkpts % kpt_comm.size == 0
        
        return domain_comm, kpt_comm, band_comm

    def get_reference_energy(self):
        return self.wfs.setups.Eref * Hartree
    
    def write(self, filename, mode='', db=False, private="660", **kwargs):
        """Write state to file.

        use mode='all' to write the wave functions.  db=True means an
        extra db output file is created and stored in a public
        location If more keyword-parameters are provided, they are
        added to the db-output (``*.db``).
        """
        
        self.timer.start('IO')
        gpaw.io.write(self, filename, mode, db=db, private=private, **kwargs)
        self.timer.stop('IO')
        
    def initialize_kinetic(self):
        if not self.hamiltonian.xc.xcfunc.mgga:
            return
        else:
            #pseudo kinetic energy array on 3D grid
            self.density.initialize_kinetic(self.atoms)
            self.density.interpolate_kinetic()
            self.hamiltonian.xc.set_kinetic(self.density.taut_sg)

    def update_kinetic(self):
        if not self.hamiltonian.xc.xcfunc.mgga:
            return
        else:
            #pseudo kinetic energy array on 3D grid
            self.density.update_kinetic(self.wfs)
            self.density.interpolate_kinetic()
            self.hamiltonian.xc.set_kinetic(self.density.taut_sg)           

    def get_myu(self, k, s):
        """Return my u corresponding to a certain kpoint and spin - or None"""
        # very slow, but we are sure that we have it
        for u in range(len(self.wfs.kpt_u)):
            if self.wfs.kpt_u[u].k == k and self.wfs.kpt_u[u].s == s:
                return u
        return None
            
    def get_homo_lumo(self):
        """Return HOMO and LUMO eigenvalues."""
        return self.occupations.get_homo_lumo(self.wfs) * Hartree

    def estimate_memory(self, mem):
        """Estimate memory use of this object."""
        mem_init = memory() # XXX initial overhead includes part of Hamiltonian
        mem.subnode('Initial overhead', mem_init)
        for name, obj in [('Density', self.density),
                          ('Hamiltonian', self.hamiltonian),
                          ('Wavefunctions', self.wfs),
                          ]:
            obj.estimate_memory(mem.subnode(name))

    def print_memory_estimate(self, txt=None, maxdepth=-1):
        """Print estimated memory usage for PAW object and components.

        maxdepth is the maximum nesting level of displayed components.

        The PAW object must be initialize()'d, but needs not have large
        arrays allocated."""
        # NOTE.  This should work with --dry-run=N
        #
        # However, the initial overhead estimate is wrong if this method
        # is called within a real mpirun/gpaw-python context.
        if txt is None:
            txt = self.txt
        print >> txt, 'Memory estimate'
        print >> txt, '---------------'
        mem = MemNode('Calculator', 0)
        try:
            self.estimate_memory(mem)
        except AttributeError, m:
            print >> txt, 'Attribute error:', m
            print >> txt, 'Some object probably lacks estimate_memory() method'
            print >> txt, 'Memory breakdown may be incomplete'
        totalsize = mem.calculate_size()
        mem.write(txt, maxdepth=maxdepth)

    def converge_wave_functions(self):
        """Converge the wave-functions if not present."""
        
        if self.scf.converged:
            # are the wfs ok ?
            error = self.wfs.eigensolver.error
            criterion = (self.input_parameters['convergence']['eigenstates']
                         * self.nvalence)

            if error < criterion:
                # print "nothing to be done"
                return

            # XXX direct access to private property
            self.scf.converged = False

            # is the density ok ?
            error = self.density.mixer.get_charge_sloshing()
            criterion = (self.input_parameters['convergence']['density']
                         * self.nvalence)

            if error < criterion:
                # print "fixing the density"
                self.scf.fix_density()

        # we have nothing
        self.calculate()
