# -*- coding: utf-8 -*-
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import sys

import Numeric as num

import gpaw.io
import gpaw.mpi as mpi
import gpaw.occupations as occupations
from gpaw import output
from gpaw import debug, sigusr1
from gpaw import ConvergenceError
from gpaw.density import Density
from gpaw.eigensolvers import Eigensolver
from gpaw.eigensolvers.rmm_diis import RMM_DIIS
from gpaw.eigensolvers.cg import CG
from gpaw.eigensolvers.davidson import Davidson
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import Hamiltonian
from gpaw.kpoint import KPoint
from gpaw.localized_functions import LocFuncBroadcaster
from gpaw.utilities import DownTheDrain, warning
from gpaw.utilities.timing import Timer
from gpaw.xc_functional import XCFunctional
import _gpaw

MASTER = 0


class Paw:
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
     ``domain``      Domain object.
     ``setups``      List of setup objects.
     ``symmetry``    Symmetry object.
     ``timer``       Timer object.
     ``nuclei``      List of ``Nucleus`` objects.
     ``out``         Output stream for text.
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
     ``weights_k``   Weights of the **k**-points in the irreducible part
                     of the Brillouin zone (summing up to 1).
     ``myibzk_kc``   Scaled **k**-points in the irreducible part of the
                     Brillouin zone for this CPU.
     ``kpt_comm``    MPI-communicator for parallelization over
                     **k**-points.
     =============== =====================================================

    Energy contributions and forces:
     =========== ================================
     ``Ekin``    Kinetic energy.
     ``Epot``    Potential energy.
     ``Etot``    Total energy.
     ``Exc``     Exchange-Correlation energy.
     ``Eext``    Energy of external potential
     ``Eref``    Reference energy for all-electron atoms.
     ``S``       Entropy.
     ``Ebar``    Should be close to zero!
     ``F_ac``    Forces.
     =========== ================================

    The attribute ``usesymm`` has the same meaning as the
    corresponding ``Calculator`` keyword (see the Manual_).  Internal
    units are Hartree and Angstrom and ``Ha`` and ``a0`` are the
    conversion factors to external `ASE units`_.  ``error`` is the
    error in the Kohn-Sham wave functions - should be zero (or small)
    for a converged calculation.

    Booleans describing the current state:
     ============= ======================================
     ``forces_ok`` Have the forces bee calculated yet?
     ``converged`` Do we have a self-consistent solution?
     ============= ======================================

    Number of iterations for:
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
    """
    
    def __init__(self, a0, Ha,
                 setups, nuclei, domain, N_c, symmetry, xcfunc,
                 nvalence, charge, nbands, nspins, random,
                 typecode, bzk_kc, ibzk_kc, weights_k,
                 stencils, usesymm, mix, fixdensity, maxiter,
                 convergeall, eigensolver, relax, pos_ac, timer, kT,
                 tolerance, kpt_comm, restart_file, hund, fixmom, magmom_a,
                 out, vext_g):
        """Create the PAW-object.
        
        Instantiating such an object by hand is *not* recommended!
        Use the ``create_paw_object()`` helper-function instead (it
        will supply many default values).  The helper-function is used
        by the ``Calculator`` object."""
        """Construct wave-function object.

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
         ``weights_k``   Weights of the **k**-points in the irreducible part
                         of the Brillouin zone (summing up to 1).
         ``myibzk_kc``   Scaled **k**-points in the irreducible part of the
                         Brillouin zone for this CPU.
         ``myweights_k`` Weights of the **k**-points on this CPU.
         ``kpt_comm``    MPI-communicator for parallelization over
                         **k**-points.
         =============== ===================================================
        """

        self.timer = timer

        self.timer.start('Init')

        self.nvalence = nvalence
        self.nbands = nbands
        self.typecode = typecode
        self.bzk_kc = bzk_kc
        self.ibzk_kc = ibzk_kc
        self.weights_k = weights_k
        self.kpt_comm = kpt_comm

        self.nkpts = len(ibzk_kc)

        self.a0 = a0  # Bohr and ...
        self.Ha = Ha  # Hartree units are used internally
        self.nuclei = nuclei
        self.domain = domain
        self.symmetry = symmetry
        self.nspins = nspins
        self.usesymm = usesymm
        self.maxiter = maxiter
        self.setups = setups
        self.random_wf = random
        
        self.set_output(out)
        
        # Construct grid descriptors for coarse grids (wave functions) and
        # fine grids (densities and potentials):
        self.gd = GridDescriptor(domain, N_c)
        self.finegd = GridDescriptor(domain, 2 * N_c)

        self.set_forces(num.empty((len(nuclei), 3), num.Float))
        self.forces_ok = False

        # Total number of k-point/spin combinations:
        nu = self.nkpts * nspins
        
        # Number of k-point/spin combinations on this cpu:
        self.nmyu = nu // kpt_comm.size

        self.kpt_u = []
        for u in range(self.nmyu):
            s, k = divmod(kpt_comm.rank * self.nmyu + u, self.nkpts)
            weight = weights_k[k] * 2 / nspins
            k_c = ibzk_kc[k]
            self.kpt_u.append(KPoint(self.gd, weight, s, k, u, k_c, typecode))

        self.locfuncbcaster = LocFuncBroadcaster(kpt_comm)
        
        self.my_nuclei = []
        self.pt_nuclei = []
        self.ghat_nuclei = []

        self.density = Density(self.gd, self.finegd, hund, fixmom, magmom_a,
                               charge, nspins,
                               stencils, mix, timer, fixdensity, kpt_comm,
                               kT,
                               self.my_nuclei, self.ghat_nuclei, self.nuclei,
                               nvalence)
        
        self.hamiltonian = Hamiltonian(self.gd, self.finegd, xcfunc,
                                       nspins,
                                       typecode, stencils, relax,
                                       timer,
                                       self.my_nuclei, self.pt_nuclei,
                                       self.ghat_nuclei,
                                       self.nuclei, setups, vext_g)
        
        # Create object for occupation numbers:
        if kT == 0 or 2 * nbands == nvalence:
            self.occupation = occupations.ZeroKelvin(nvalence, nspins)
        else:
            self.occupation = occupations.FermiDirac(nvalence, nspins, kT)

        xcfunc.set_non_local_things(self)

        if fixmom:
            M = sum(magmom_a)
            self.occupation.fix_moment(M)

        self.occupation.set_communicator(kpt_comm)

        self.Eref = 0.0
        for nucleus in self.nuclei:
            self.Eref += nucleus.setup.E

        output.print_info(self)

        self.eigensolver = (eigensolver, convergeall, tolerance, nvalence)

        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = domain.scale_position(pos_c)
            nucleus.set_position(spos_c, domain, self.my_nuclei,
                                 self.nspins, self.nmyu, self.nbands)

        output.plot_atoms(self)

        self.density.mixer.reset(self.my_nuclei)
            
        self.wave_functions_initialized = False
        self.density_initialized = False
        if restart_file is not None:
            self.wave_functions_initialized = self.initialize_from_file(
                restart_file)
            self.density_initialized = True

        self.timer.stop()

    def find_ground_state(self, pos_ac, cell_c):
        """Start iterating towards the ground state."""

        pos_ac = pos_ac / self.a0
        cell_c = cell_c / self.a0

        self.load_wave_functions(pos_ac)

        assert not self.converged

        self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc = \
                    self.hamiltonian.update(self.density)

        self.niter = 0
        # Self-consistency loop:
        while not self.converged:
            if self.niter > 2:
                self.density.update(self.kpt_u, self.symmetry)
                self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc = \
                           self.hamiltonian.update(self.density)

            self.error, self.converged = self.eigensolver.iterate(
                self.hamiltonian, self.kpt_u)

            # Make corrections due to non-local xc
            self.Exc += self.hamiltonian.xc.xcfunc.get_non_local_energy()
            self.Ekin0 += self.hamiltonian.xc.xcfunc.get_non_local_kinetic_corrections()
                
            # Calculate occupation numbers:
            self.nfermi, self.magmom, self.S, Eband = \
                         self.occupation.calculate(self.kpt_u)

            self.Ekin = self.Ekin0 + Eband
            self.Etot = (self.Ekin + self.Epot + self.Ebar + 
                         self.Eext + self.Exc - self.S)

            output.iteration(self)

            self.niter += 1
            if self.niter > 120:
                raise ConvergenceError('Did not converge!')

            if self.niter > self.maxiter - 1:
                self.converged = True
                
        output.print_converged(self)

    def set_positions(self, pos_ac=None):
        """Update the positions of the atoms.

        Localized functions centered on atoms that have moved will
        have to be computed again.  Neighbor list is updated and the
        array holding all the pseudo core densities is updated."""
        
        self.timer.start('Init pos.')
        
        if pos_ac is None:
            pos_ac = num.array([nucleus.spos_c * self.domain.cell_c
                                for nucleus in self.nuclei])

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
            self.converged = False
            self.forces_ok = False

            self.locfuncbcaster.broadcast()

            for nucleus in self.nuclei:
                nucleus.normalize_shape_function_and_pseudo_core_density()
                
            if self.symmetry:
                self.symmetry.check(pos_ac)
                
            self.hamiltonian.pairpot.update(pos_ac, self.nuclei, self.domain)

            self.density.move()

            print >> self.out, 'Positions:'
            for a, pos_c in enumerate(pos_ac):
                symbol = self.nuclei[a].setup.symbol
                print >> self.out, '%3d %2s %8.4f%8.4f%8.4f' % \
                      ((a, symbol) + tuple(self.a0 * pos_c))

        self.timer.stop()

    def initialize_wave_functions(self):
        """Initialize wave function from atomic orbitals."""
        # count the total number of atomic orbitals (bands):
        nao = 0
        for nucleus in self.nuclei:
            nao += nucleus.get_number_of_atomic_orbitals()

        if self.random_wf:
            nao = 0

        nrandom = max(0, self.nbands - nao)

        print >> self.out, self.nbands, 'band%s.' % 's'[:self.nbands != 1]
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
                
        print >> self.out, string


        xcfunc = self.hamiltonian.xc.xcfunc

        ## XXXX MK
        # At the first iteration we use LDA in KLI also
        if (xcfunc.xcname == 'KLI'):
            localxcfunc = XCFunctional('LDAx')
            self.hamiltonian.xc.set_functional(localxcfunc)

            
        if xcfunc.hybrid > 0:
            # At this point, we can't use orbital dependent
            # functionals, because we don't have the right orbitals
            # yet.  So we use a simple density functional to setup the
            # initial hamiltonian:
            if xcfunc.xcname == 'EXX':
                localxcfunc = XCFunctional('LDAx')
            else:
                assert xcfunc.xcname == 'PBE0'
                localxcfunc = XCFunctional('PBE')
            self.hamiltonian.xc.set_functional(localxcfunc)
            for setup in self.setups:
                setup.xc_correction.xc.set_functional(localxcfunc)
                            
        self.Ekin0, self.Epot, self.Ebar, self.Eext, self.Exc = \
                   self.hamiltonian.update(self.density)

        if self.random_wf:
            for kpt in self.kpt_u:
                kpt.create_random_orbitals(self.nbands)
                # Calculate projections and orthogonalize wave functions:
                for nucleus in self.pt_nuclei:
                    nucleus.calculate_projections(kpt)
                kpt.orthonormalize(self.my_nuclei)
            # Improve the random guess with conjugate gradients
            eig = CG(self.timer,self.kpt_comm,
                     self.gd, self.hamiltonian.kin,
                     self.typecode, self.nbands)
            eig.set_convergence_criteria(True, 1e-2, self.nvalence)
            for nit in range(2):
                eig.iterate(self.hamiltonian, self.kpt_u) 

        else:
            for nucleus in self.my_nuclei:
                # XXX already allocated once, but with wrong size!!!
                ni = nucleus.get_number_of_partial_waves()
                nucleus.P_uni = num.empty((self.nmyu, nao, ni), self.typecode)

            # Use the generic eigensolver for subspace diagonalization
            eig = Eigensolver(self.timer,self.kpt_comm,
                              self.gd, self.hamiltonian.kin,
                              self.typecode, nao)
            for kpt in self.kpt_u:
                kpt.create_atomic_orbitals(nao, self.nuclei)
                # Calculate projections and orthogonalize wave functions:
                for nucleus in self.pt_nuclei:
                    nucleus.calculate_projections(kpt)
                kpt.orthonormalize(self.my_nuclei)
                eig.diagonalize(self.hamiltonian, kpt)


        for nucleus in self.my_nuclei:
            nucleus.reallocate(self.nbands)

        for kpt in self.kpt_u:
            kpt.adjust_number_of_bands(self.nbands, self.pt_nuclei, self.my_nuclei)

        # XXXX MK
        # Switch back to KLI from Lda
        if xcfunc.xcname == 'KLI':
            self.hamiltonian.xc.set_functional(xcfunc)
                
        if xcfunc.hybrid > 0:
            # Switch back to the orbital dependent functional:
            self.hamiltonian.xc.set_functional(xcfunc)
            for setup in self.setups:
                setup.xc_correction.xc.set_functional(xcfunc)


        # Calculate occupation numbers:
        self.nfermi, self.magmom, self.S, Eband = \
                     self.occupation.calculate(self.kpt_u)

    def get_total_energy(self, force_consistent):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        
        if force_consistent:
            # Free energy:
            return self.Ha * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return self.Ha * (self.Etot + 0.5 * self.S)

    def get_cartesian_forces(self):
        """Return the atomic forces."""
        c = self.Ha / self.a0
        
        if self.forces_ok:
            return c * self.F_ac

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
            F_ac = num.zeros((len(self.nuclei), 3), num.Float)
            for map_a, symmetry in zip(self.symmetry.maps,
                                       self.symmetry.symmetries):
                swap, mirror = symmetry
                for a1, a2 in enumerate(map_a):
                    F_ac[a2] += num.take(self.F_ac[a1] * mirror, swap)
            self.F_ac[:] = F_ac / len(self.symmetry.symmetries)

        if mpi.rank == MASTER:
            for a, nucleus in enumerate(self.nuclei):
                print >> self.out, 'forces ', \
                      a, nucleus.setup.symbol, self.F_ac[a] * c

        self.forces_ok = True

        return c * self.F_ac

    def set_forces(self, F_ac):
        """Initialize atomic forces."""
        self.forces_ok = True
        # Forces for all atoms:
        self.F_ac = F_ac
            
    def set_convergence_criteria(self, tol):
        """Set convergence criteria.

        Stop iterating when the size of the residuals are below
        ``tol``."""
        
        if tol < self.eigensolver.tolerance:
            self.converged = False
        self.eigensolver.tolerance = tol
        
    def set_output(self, out):
        """Set the output stream for text output."""
        if mpi.rank != MASTER:                
            if debug:
                out = sys.stderr
            else:
                out = DownTheDrain()
        self.out = out

    def write_state_to_file(self, filename, pos_ac, magmom_a, tag_a, mode,
                            setup_types):
        """Write current state to a file."""
        gpaw.io.write(self, filename, pos_ac / self.a0, magmom_a, tag_a,
                         mode, setup_types)
        
    def initialize_from_file(self, filename):
        """Read state from a file."""
        wf = gpaw.io.read(self, filename)
        return wf

    def warn(self, message):
        """Print a warning-message."""
        print >> self.out, warning(message)
        raise RuntimeError(message)

    def __del__(self):
        """Destructor:  Write timing output before closing."""
        self.timer.write(self.out)

    def get_fermi_level(self):
        """Return the Fermi-level."""
        e = self.occupation.get_fermi_level()
        if e is None:
            e = 100.0
        return e * self.Ha

    def get_ibz_kpoints(self):
        """Return array of k-points in the irreducible part of the BZ."""
        return self.ibzk_kc

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master.""" 
        
        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)

        if not mpi.parallel:
            return self.kpt_u[u].psit_nG[n]

        if self.kpt_comm.rank == kpt_rank:
            psit_G = self.gd.collect(self.kpt_u[u].psit_nG[n])

            if kpt_rank == MASTER:
                if mpi.rank == MASTER:
                    return psit_G

            # Domain master send this to the global master
            if self.domain.comm.rank == MASTER:
                self.kpt_comm.send(psit_G, MASTER, 1398)

        if mpi.rank == MASTER:
            # allocate full wavefunction and receive 
            psit_G = self.gd.empty(typecode=self.typecode, global_array=True)
            self.kpt_comm.receive(psit_G, kpt_rank, 1398)
            return psit_G

    def get_eigenvalues(self, k, s):
        """Return eigenvalue array.
        
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master.""" 
        
        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)

        if kpt_rank == MASTER:
            return self.kpt_u[u].eps_n
        
        if self.kpt_comm.rank == kpt_rank:
            # Domain master send this to the global master
            if self.domain.comm.rank == MASTER:
                self.kpt_comm.send(self.kpy_u[u].eps_n, MASTER, 1301)
        elif mpi.rank == MASTER:
            eps_n = num.zeros(self.nbands, num.Float)
            self.kpt_comm.receive(eps_n, kpt_rank, 1301)
            return eps_n
        

    def get_wannier_integrals(self, i, s, k, k1, G_I):
        """Calculate integrals for maximally localized Wannier functions."""

        assert s <= self.nspins

        kpt_rank, u = divmod(k + self.nkpts * s, self.nmyu)
        kpt_rank1, u1 = divmod(k1 + self.nkpts * s, self.nmyu)

        # XXX not for the kpoint/spin parallel case
        assert self.kpt_comm.size==1

        G = G_I[i]
        return self.gd.wannier_matrix(self.kpt_u[u].psit_nG,
                                      self.kpt_u[u1].psit_nG,
                                      i,
                                      k,k1,G)
    
    def get_xc_difference(self, xcname):
        """Calculate non-selfconsistent XC-energy difference."""
        xc = self.hamiltonian.xc
        oldxcfunc = xc.xcfunc

        if isinstance(xcname, str):
            newxcfunc = XCFunctional(xcname)
        else:
            newxcfunc = xcname

        newxcfunc.set_non_local_things(self, energy_only=True)

        xc.set_functional(newxcfunc)
        for setup in self.setups:
            setup.xc_correction.xc.set_functional(newxcfunc)

        if newxcfunc.hybrid > 0.0 and not self.nuclei[0].ready:
            self.set_positions(num.array([n.spos_c * self.domain.cell_c
                                          for n in self.nuclei]))
            
        vt_g = self.finegd.empty()  # not used for anything!
        nt_sg = self.density.nt_sg
        if self.nspins == 2:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g, nt_sg[1], vt_g)
        else:
            Exc = xc.get_energy_and_potential(nt_sg[0], vt_g)

        for nucleus in self.my_nuclei:
            D_sp = nucleus.D_sp
            H_sp = num.zeros(D_sp.shape, num.Float) # not used for anything!
            xc_correction = nucleus.setup.xc_correction
            Exc += xc_correction.calculate_energy_and_derivatives(D_sp, H_sp)
            
        Exc = self.domain.comm.sum(Exc)

        for kpt in self.kpt_u:
            newxcfunc.apply_non_local(kpt)
        Exc += newxcfunc.get_non_local_energy()
        
        xc.set_functional(oldxcfunc)
        for setup in self.setups:
            setup.xc_correction.xc.set_functional(oldxcfunc)

        return self.Ha * (Exc - self.Exc)

    def get_grid_spacings(self):
        return self.a0 * self.gd.h_c
    
    def get_exact_exchange(self):
        dExc = self.get_xc_difference('EXX') / self.Ha
        Exx = self.Exc + dExc
        for nucleus in self.nuclei:
            Exx += nucleus.setup.xc_correction.Exc0
        return Exx
    
    def get_weights(self):
        return self.weights_k

    def load_wave_functions(self, pos_ac):
        self.set_positions(pos_ac)
            
        if not self.wave_functions_initialized: 
            # Initialize wave functions and perhaps also the density
            # from atomic orbitals:
            for nucleus in self.nuclei:
                nucleus.initialize_atomic_orbitals(self.gd, self.ibzk_kc,
                                                   self.locfuncbcaster)
            self.locfuncbcaster.broadcast()

            if not self.density_initialized:
                self.density.initialize()
                self.density_initialized = True
                
            self.initialize_wave_functions()
            self.wave_functions_initialized = True
                
            self.converged = False

            # Free allocated space for radial grids:
            for setup in self.setups:
                del setup.phit_j
            for nucleus in self.nuclei:
                try:
                    del nucleus.phit_j
                except AttributeError:
                    pass

        if not isinstance(self.kpt_u[0].psit_nG, num.ArrayType):
            assert not mpi.parallel
            # Calculation started from a restart file.  Allocate arrays
            # for wave functions and copy data from the file:
            for kpt in self.kpt_u:
                kpt.psit_nG = kpt.psit_nG[:]

        for kpt in self.kpt_u:
            kpt.adjust_number_of_bands(self.nbands, self.pt_nuclei, self.my_nuclei)

        if isinstance(self.eigensolver, tuple):
            eigensolver, convergeall, tolerance, nvalence = self.eigensolver
            if eigensolver == 'rmm-diis':
                self.eigensolver = RMM_DIIS(self.timer,
                                            self.kpt_comm,
                                            self.gd, self.hamiltonian.kin,
                                            self.typecode, self.nbands)
            elif eigensolver == 'cg':
                self.eigensolver = CG(self.timer, self.kpt_comm, 
                                      self.gd, self.hamiltonian.kin,
                                      self.typecode, self.nbands)
            elif eigensolver == 'dav':
                self.eigensolver = Davidson(self.timer, self.kpt_comm, 
                                      self.gd, self.hamiltonian.kin,
                                      self.typecode, self.nbands)
            else:
                raise NotImplementedError('Eigensolver %s' % eigensolver)

            self.eigensolver.set_convergence_criteria(convergeall, tolerance,
                                                      nvalence)


    def i2(self):
        self.density.initialize2()
