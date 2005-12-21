# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class."""

import sys
from math import pi, sqrt, log
import time

import Numeric as num

from gridpaw import debug, sigusr1
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.pair_potential import PairPotential
from gridpaw.poisson_solver import PoissonSolver
from gridpaw.density_mixer import Mixer, MixerSum
from gridpaw.utilities import DownTheDrain, warning
from gridpaw.utilities.timing import Timer
from gridpaw.transformers import Interpolator, Restrictor
from gridpaw.wf import WaveFunctions
from gridpaw.xc_functional import XCOperator
from gridpaw.localized_functions import LocFuncBroadcaster
import gridpaw.utilities.mpi as mpi
from gridpaw import netcdf
from gridpaw import output


NOT_INITIALIZED = -1
NOTHING = 0
COMPENSATION_CHARGE = 1
PROJECTOR_FUNCTION = 2
EVERYTHING = 3


MASTER = 0


class Paw:
    """This is the main calculation object for doing a PAW calculation.

    The ``Paw`` object is the central object for a calculation.
    
    These are the most important attributes of a ``Paw`` object:
     =============== =====================================================
     ``domain``      Domain object.
     ``setups``      Dictionary mapping chemical symbols to setup objects.
     ``symmetry``    Symmetry object.
     ``timer``       Timer object.
     ``wf``          ``WaveFunctions`` object.
     ``xc``          ``XCOperator`` object.
     ``xcfunc``      ``XCFunctional`` object.
     ``nuclei``      List of ``Nucleus`` objects.
     ``out``         Output stream for text.
     ``pairpot``     ``PairPotential`` object.
     ``poisson``     ``PoissonSolver``.
     ``gd``          Grid descriptor for coarse grids.
     ``finegd``      Grid descriptor for fine grids.
     ``restrict``    Function for restricting the effective potential.
     ``interpolate`` Function for interpolating the electron density.
     ``mixer``       ``DensityMixer`` object.
     =============== =====================================================

    Energy contributions and forces:
     =========== ================================
     ``Ekin``    Kinetic energy.
     ``Epot``    Potential energy.
     ``Etot``    Total energy.
     ``Exc``     Exchange-Correlation energy.
     ``Eref``    Reference energy for all-electron atoms.
     ``S``       Entropy.
     ``Ebar``    Should be close to zero!
     ``F_ac``    Forces.
     =========== ================================


    The attributes ``tolerance``, ``fixdensity``, ``idiotproof`` and
    ``usesymm`` have the same meaning as the corresponding
    ``Calculator`` keywords (see the Manual_).  Internal units are
    Hartree and Angstrom and ``Ha`` and ``a0`` are the conversion
    factors to external `ASE units`_.  ``error`` is the error in the
    Kohn-Sham wave functions - should be zero (or small) for a
    converged calculation.

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

    Soft and smooth pseudo functions on uniform 3D grids:
     ========== =========================================
     ``nt_sG``  Electron density on the coarse grid.
     ``nt_sg``  Electron density on the fine grid.
     ``rhot_g`` Charge density on the coarse grid.
     ``nct_G``  Core electron-density on the coarse grid.
     ``vHt_g``  Hartree potential on the fine grid.
     ``vt_sG``  Effective potential on the coarse grid.
     ``vt_sg``  Effective potential on the fine grid.
     ========== =========================================

    Only attribute not mentioned now is ``nspins`` (number of spins) and
    those used for parallelization:

     ================== =================================================== 
     ``my_nuclei``      List of nuclei that have their
                        center in this domain.
     ``p_nuclei``       List of nuclei with projector functions
                        overlapping this domain.
     ``g_nuclei``       List of nuclei with compensation charges
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
                 nvalence, nbands, nspins, kT,
                 typecode, bzk_kc, ibzk_kc, weights_k,
                 order, usesymm, mix, old, fixdensity, maxiter, idiotproof,
                 # Parallel stuff:
                 myspins,
                 myibzk_kc, myweights_k, kpt_comm,
                 out):
        """
        Create the PAW-object.
        
        Instantiating such an object by hand is *not* recommended!
        Use the ``create_paw_object()`` helper-function instead (it
        will supply many default values).  The helper-function is used
        by the ``Calculator`` object."""

        self.timer = Timer()

        self.a0 = a0  # Bohr and ...
        self.Ha = Ha  # Hartree units are used internally
        self.setups = setups
        self.nuclei = nuclei
        self.domain = domain
        self.symmetry = symmetry
        self.xcfunc = xcfunc
        self.nspins = nspins
        self.usesymm = usesymm
        self.fixdensity = fixdensity
        self.maxiter = maxiter
        self.idiotproof = idiotproof

        self.set_output(out)
        
        # Construct grid descriptors for coarse grids (wave functions) and
        # fine grids (densities and potentials):
        self.gd = GridDescriptor(domain, N_c)
        self.finegd = GridDescriptor(domain, 2 * N_c)
        if not self.gd.is_healthy():
            self.warn(
                'VERY ANISOTROPIC GRIDSPACINGS: ' + str(self.a0 * self.gd.h_c))

        self.set_forces(num.empty((len(nuclei), 3), num.Float))
        self.forces_ok = False

        # Allocate arrays for potentials and densities on coarse and
        # fine grids:
        self.nct_G = self.gd.new_array()
        self.vt_sG = self.gd.new_array(nspins)
        self.nt_sG = self.gd.new_array(nspins)
        self.rhot_g = self.finegd.new_array()
        self.vHt_g = self.finegd.new_array()        
        self.vt_sg = self.finegd.new_array(nspins)
        self.nt_sg = self.finegd.new_array(nspins)

        # Wave functions ...
        self.wf = WaveFunctions(self.gd, nvalence, nbands, nspins,
                                typecode, kT / Ha,
                                bzk_kc, ibzk_kc, weights_k,
                                myspins, myibzk_kc, myweights_k, kpt_comm)

        self.locfuncbcaster = LocFuncBroadcaster(kpt_comm)
        
        # exchange-correlation functional object:
        self.xc = XCOperator(xcfunc, self.finegd, nspins)

        # Interpolation function for the density:
        self.interpolate = Interpolator(self.gd, order, num.Float).apply
        # Restrictor function for the potential:
        self.restrict = Restrictor(self.finegd, order, num.Float).apply

        # Solver for the posisson equation:
        self.poisson = PoissonSolver(self.finegd, out)
   
        # Density mixer:
        if nspins == 2 and kT != 0:
            self.mixer = MixerSum(mix, old, nspins)
        else:
            self.mixer = Mixer(mix, old, nspins)

        # Pair potential for electrostatic interacitons:
        self.pairpot = PairPotential(setups)

        self.my_nuclei = self.nuclei
        self.p_nuclei = self.nuclei
        self.g_nuclei = self.nuclei
        
        self.Eref = 0.0
        for nucleus in self.nuclei:
            self.Eref += nucleus.setup.E

        self.tolerance = 100000000000.0
        
        output.print_info(self)

    def set_positions(self, pos_ac):
        """Update the positions of the atoms.

        Localized functions centered on atoms that have moved will
        have to be computed again.  Neighbor list is updated and the
        array holding all the pseudo core densities is updated."""
        
        movement = False
        distribute_atoms = False
        for nucleus, pos_c in zip(self.nuclei, pos_ac):
            spos_c = self.domain.scale_position(pos_c)
            if num.sometrue(spos_c != nucleus.spos_c):
                movement = True
                nucleus.spos_c = spos_c
                nucleus.make_localized_grids(self.gd, self.finegd,
                                             self.wf.myibzk_kc,
                                             self.locfuncbcaster)
                rank = self.domain.get_rank_for_position(spos_c)
                # Did the atom move to another processor?
                if nucleus.rank != rank:
                    # Yes!
                    distribute_atoms = True
                    nucleus.rank = rank
        
        if movement:
            self.converged = False
            self.forces_ok = False

            self.locfuncbcaster.broadcast()
        
            if distribute_atoms:
                self.distribute_atoms()
            
            self.mixer.reset(self.my_nuclei)
            
            if self.symmetry:
                self.symmetry.check(pos_ac)
                
            neighborlist_update = self.pairpot.update(pos_ac, self.nuclei,
                                                      self.domain)

            if neighborlist_update:
                print >> self.out
                print >> self.out, 'Neighbor list has been updated!'
                print >> self.out


            # Set up smooth core density:
            self.nct_G[:] = 0.0
            for nucleus in self.nuclei:
                nucleus.add_smooth_core_density(self.nct_G)
            if self.nspins == 2:
                self.nct_G *= 0.5

                print >> self.out, 'Positions:'
                for a, pos_c in enumerate(pos_ac):
                    symbol = self.nuclei[a].setup.symbol
                    print >> self.out, '%3d %2s %8.4f%8.4f%8.4f' % \
                  ((a, symbol) + tuple(self.a0 * pos_c))

    def initialize_density_and_wave_functions(self, hund, magmom_a,
                                              density=True,
                                              wave_functions=True):
        """Initialize density and/or wave functions.

        By default both wave functions and densities are initialized
        (from atomic orbitals) - this can be turned off with the
        ``density`` and ``wave_functions`` keywords.  The density will
        be constructed with the specified magnetic mioments and
        obeying Hund's rules if ``hund`` is true."""
        
        output.plot_atoms(self)
        
        for nucleus in self.nuclei:
            nucleus.initialize_atomic_orbitals(self.gd, self.wf.myibzk_kc,
                                               self.locfuncbcaster)
        self.locfuncbcaster.broadcast()

        if density:
            self.nt_sG[:] = self.nct_G
            for magmom, nucleus in zip(magmom_a, self.nuclei):
                nucleus.add_atomic_density(self.nt_sG, magmom, hund)

        if wave_functions:
            self.wf.initialize_from_atomic_orbitals(self.nuclei,
                                                    self.my_nuclei, self.out)

        # Free allocated space for radial grids:
        for setup in self.setups.values():
            setup.delete_atomic_orbitals()
        for nucleus in self.nuclei:
            try:
                del nucleus.phit_j
            except AttributeError:
                continue

        if hund:
            M = int(0.5 + num.sum(magmom_a))
            from gridpaw.occupations import FixMom
            self.wf.occupation = FixMom(self.wf.occupation.ne, self.nspins, M)

        self.converged = False
        
    def set_convergence_criteria(self, tol):
        """Set convergence criteria.

        Stop iterating when the size of the residuals is belov
        ``tol``."""
        
        if tol < self.tolerance:
            self.converged = False
        self.tolerance = tol
        
    def set_output(self, out):
        """Set the output stream for text output."""
        if mpi.rank != MASTER:                
            if debug:
                out = sys.stderr
            else:
                out = DownTheDrain()
        self.out = out

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

    def find_ground_state(self, pos_ac, cell_c, angle):
        """Start iterating towards the ground state."""
        pos_ac = pos_ac / self.a0
        cell_c = cell_c / self.a0
        self.set_positions(pos_ac)

        assert not self.converged

        out = self.out
        
        print >> out, """\
                       log10     total     iterations:
              time     error     energy    fermi  poisson  magmom"""

        self.niter = 0
        while not self.converged:
            self.improve_wave_functions()
            # Output from each iteration:
            t = time.localtime()
            out.write('iter: %4d %3d:%02d:%02d %6.1f %13.7f %4d %7d' %
                      (self.niter,
                       t[3], t[4], t[5],
                       log(self.error) / log(10),
                       self.Ha * (self.Etot + 0.5 * self.S),
                       self.nfermi,
                       self.npoisson))
            if self.nspins == 2:
                print >> out, '%11.4f' % self.magmom
            else:
                print >> out, '       --'
                
            out.flush()
            self.niter += 1
            if self.niter > 1240:
                raise RuntimeError('Did not converge!')

        output.print_converged(self)

    def improve_wave_functions(self):
        """Iterate towards self-consistency."""

        wf = self.wf

        from netcdf import NetCDFWaveFunction
        if isinstance(wf.kpt_u[0].psit_nG, NetCDFWaveFunction):
            assert self.niter == 0
            # Calculation started from a NetCDF restart file.
            # Allocate array for wavefunctions and copy data from the
            # NetCDFWaveFunction class
            u = 0
            for s in range(len(wf.myspins)):
                for k in range(wf.nmykpts):
                    kpt = wf.kpt_u[u]
                    tmp_nG = kpt.psit_nG
                    kpt.psit_nG = kpt.gd.new_array(wf.nbands, wf.typecode)
                    kpt.Htpsit_nG = kpt.gd.new_array(wf.nbands, wf.typecode)

                    # distribute band by band to save memory
                    for n in range(wf.nbands):
                        kpt.gd.distribute(tmp_nG[n],kpt.psit_nG[n])

                    u += 1
                    
            self.calculate_multipole_moments()

                    
                    
        elif self.niter == 0:
            # We don't have any occupation numbers.  The initial
            # electron density comes from overlapping atomic densities
            # or from a restart file.  We scale the density to match
            # the compensation charges.

            self.calculate_multipole_moments()
            Q = 0.0
            for nuclei in self.my_nuclei:
                Q += nuclei.Q_L[0]
            Q = sqrt(4 * pi) * self.domain.comm.sum(Q)

            Nt = self.gd.integrate(self.nt_sG)
            # Nt + Q must be zero:
            x = -Q / Nt
            assert 0.83 < x < 1.17, 'x=%f' % x
            self.nt_sG *= x
        
        wf.calculate_projections_and_orthogonalize(self.p_nuclei,
                                                   self.my_nuclei)

        if self.niter > 0:
            if not self.fixdensity:
                wf.calculate_electron_density(self.nt_sG, self.nct_G,
                                              self.symmetry, self.gd)
                wf.calculate_atomic_density_matrices(self.my_nuclei,
                                                     self.nuclei,
                                                     self.domain.comm,
                                                     self.symmetry)
                self.mixer.mix(self.nt_sG, self.domain.comm)
                
            self.calculate_multipole_moments()

        # Transfer the density to the fine grid:
        for s in range(self.nspins):
            self.interpolate(self.nt_sG[s], self.nt_sg[s])

        self.calculate_potential()

        self.calculate_atomic_hamiltonians()

        wf.diagonalize(self.vt_sG, self.my_nuclei)

        if self.niter == 0:
            for nucleus in self.my_nuclei:
                nucleus.reallocate(wf.nbands)

        # Calculate occupations numbers, and return number of
        # iterations, magnetic moment and entropy:
        
        self.nfermi, self.magmom, self.S = wf.calculate_occupation_numbers()
        
        dsum = self.domain.comm.sum
        self.Ekin = dsum(self.Ekin) + wf.sum_eigenvalues()
        self.Epot = dsum(self.Epot)
        self.Ebar = dsum(self.Ebar)
        self.Exc = dsum(self.Exc)
        self.Etot = self.Ekin + self.Epot + self.Ebar + self.Exc - self.S

        self.error = dsum(wf.calculate_residuals(self.p_nuclei))

        if (self.error > self.tolerance and self.niter < self.maxiter) and not sigusr1[0]:
            self.timer.start('SD')
            wf.rmm_diis(self.p_nuclei, self.vt_sG)
            self.timer.stop('SD')
        else:
            self.converged = True
            if sigusr1[0]:
                print >> self.out, 'SCF-ITERATIONS STOPPED BY USER!'
                sigusr1[0] = False

    def distribute_atoms(self):
        """Distribute atoms on all CPUs."""
        nspins = self.wf.nspins
        nmykpts = self.wf.nmykpts
        nbands = self.wf.nbands

        if self.domain.comm.size == 1:
            # Serial calculation:
            for nucleus in self.nuclei:
                if nucleus.domain_overlap == NOT_INITIALIZED:
                    nucleus.allocate(nspins, nmykpts, nbands)
                nucleus.domain_overlap = EVERYTHING
            return

        # Parallel calculation:
        natoms = len(self.nuclei)
        domovl_a = num.zeros(natoms, num.Int)
        self.my_nuclei = []
        self.p_nuclei = []
        self.g_nuclei = []
        for a, nucleus in enumerate(self.nuclei):
            domain_overlap = NOTHING
            if nucleus.ghat_L is not None:
                domain_overlap = COMPENSATION_CHARGE
                self.g_nuclei.append(nucleus)
                if nucleus.pt_i is not None:
                    domain_overlap = PROJECTOR_FUNCTION
                    self.p_nuclei.append(nucleus)
                    if nucleus.rank == self.domain.comm.rank:
                        domain_overlap = EVERYTHING
                        self.my_nuclei.append(nucleus)
                        if nucleus.domain_overlap < EVERYTHING:
                            nucleus.allocate(nspins, nmykpts, nbands)
                    else:
                        if nucleus.domain_overlap == EVERYTHING:
                            nucleus.deallocate()

            nucleus.domain_overlap = domain_overlap
            domovl_a[a] = domain_overlap

        domovl_ca = num.zeros((self.domain.comm.size, natoms), num.Int)
        self.domain.comm.all_gather(domovl_a, domovl_ca)

        # Make groups:
        for a, nucleus in enumerate(self.nuclei):
            domovl_c = domovl_ca[:, a]

            # Who owns the atom?
            root = num.argmax(domovl_c)

            g_group = [c for c, b in enumerate(domovl_c) if
                       b >= COMPENSATION_CHARGE]
            g_root = g_group.index(root)
            g_comm = self.domain.comm.new_communicator(
                num.array(g_group, num.Int))

            p_group = [c for c, b in enumerate(domovl_c) if
                       b >= PROJECTOR_FUNCTION]
            p_root = p_group.index(root)
            p_comm = self.domain.comm.new_communicator(
                num.array(p_group, num.Int))

            nucleus.set_communicators(self.domain.comm, root,
                                      g_comm, g_root, p_comm, p_root)
            
    def calculate_atomic_hamiltonians(self):
        """Calculate atomic hamiltonians."""
        self.timer.start('atham')
        nt_sg = self.nt_sg
        if self.nspins == 2:
            nt_g = nt_sg[0] + nt_sg[1]
        else:
            nt_g = nt_sg[0]

        for nucleus in self.g_nuclei:
            k, p, b, x = nucleus.calculate_hamiltonian(nt_g, self.vHt_g)
            self.Ekin += k
            self.Epot += p
            self.Ebar += b
            self.Exc += x
        self.timer.stop('atham')

    def calculate_multipole_moments(self):
        """Calculate multipole moments."""
        for nucleus in self.nuclei:
            nucleus.calculate_multipole_moments()
            
    def calculate_potential(self):
        """Calculate effective potential.

        The XC-potential and the Hartree potentials are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""
        
        self.rhot_g[:] = self.nt_sg[0]
        if self.nspins == 2:
            self.rhot_g += self.nt_sg[1]

        vt_g = self.vt_sg[0]
        vt_g[:] = 0.0
        for nucleus in self.g_nuclei:
            nucleus.add_hat_potential(vt_g)

        self.Epot = num.vdot(vt_g, self.rhot_g) * self.finegd.dv 

        for nucleus in self.p_nuclei:
            nucleus.add_localized_potential(vt_g)

        self.Ebar = num.vdot(vt_g, self.rhot_g) * self.finegd.dv 
        self.Ebar -= self.Epot
        
        if self.nspins == 2:
            self.vt_sg[1, :] = vt_g

        self.timer.start('xc')
        if self.nspins == 2:
            self.Exc = self.xc.get_energy_and_potential(
                self.nt_sg[0], self.vt_sg[0], self.nt_sg[1], self.vt_sg[1])
        else:
            self.Exc = self.xc.get_energy_and_potential(
                self.nt_sg[0], self.vt_sg[0])
        self.timer.stop('xc')

        for nucleus in self.g_nuclei:
            nucleus.add_compensation_charge(self.rhot_g)
        
        assert abs(self.finegd.integrate(self.rhot_g)) < 0.2

        self.timer.start('poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_g, self.rhot_g)
        self.timer.stop('poisson')
        
        self.Epot += 0.5 * num.vdot(self.vHt_g, self.rhot_g) * self.finegd.dv
        self.Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, self.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            self.Ekin -= num.vdot(vt_G, nt_G - self.nct_G) * self.gd.dv

    def get_cartesian_forces(self):
        """Return the atomic forces."""
        c = self.Ha / self.a0
        
        if not self.forces_ok:
            if self.nspins == 2:
                nt_g = self.nt_sg[0] + self.nt_sg[1]
                vt_G = 0.5 * (self.vt_sG[0] + self.vt_sG[1])
            else:
                nt_g = self.nt_sg[0]
                vt_G = self.vt_sG[0]


            for nucleus in self.p_nuclei:
                if nucleus.domain_overlap == EVERYTHING:
                    nucleus.F_c[:] = 0.0

            self.wf.calculate_force_contribution(self.p_nuclei, self.my_nuclei)

            for nucleus in self.nuclei:
                nucleus.calculate_force(self.vHt_g, nt_g, vt_G)

            # Master (domain_comm 0) collects forces from nuclei into
            # self.F_ac:
            if mpi.rank == MASTER:
                for a, nucleus in enumerate(self.nuclei):
                    if nucleus.domain_overlap == EVERYTHING:
                        self.F_ac[a] = nucleus.F_c
                    else:
                        self.domain.comm.receive(self.F_ac[a], nucleus.rank)
            else:
                for nucleus in self.my_nuclei:
                    self.domain.comm.send(nucleus.F_c, MASTER)

            if self.symmetry is not None and mpi.rank == MASTER:
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

        if mpi.rank == MASTER:
            return c * self.F_ac

    def set_forces(self, F_ac):
        """Initialize atomic forces."""
        self.forces_ok = True
        if mpi.rank == MASTER:
            # Forces for all atoms:
            self.F_ac = F_ac
            
    def write_netcdf(self, filename):
        """Write current state to a netDF file."""
        netcdf.write_netcdf(self, filename)
        
    def initialize_from_netcdf(self, filename):
        """Read state from a netCDF file."""
        netcdf.read_netcdf(self, filename)
        output.plot_atoms(self)

    def warn(self, message):
        """Print a wrning-message."""
        print >> self.out, warning(message)
        if self.idiotproof:
            raise RuntimeError(warning)

    def __del__(self):
        """Destructor:  Write timing output before closing."""
        for kpt in self.wf.kpt_u:
            self.timer.add(kpt.timer)
        self.timer.write(self.out)

    def get_fermi_level(self):
        """Return the Fermi-level."""
        e = self.wf.occupation.get_fermi_level()
        if e is None:
            e = 100.0
        return e * self.Ha

    def get_ibz_kpoints(self):
        """Return array of k-points in the irreducible part of the BZ."""
        return self.wf.ibzk_kc

    def get_density_array(self):
        """Return pseudo-density array."""
        c = 1.0 / self.a0**3
        if self.nspins == 2:
            return self.nt_sG * c
        else:
            return self.nt_sG[0] * c

    def get_wave_function_array(self, n, k, s):
        """Return pseudo-wave-function array.
        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master.""" 
        from parallel import get_parallel_info_s_k
        
        c = 1.0 / self.a0**1.5

        kpt_rank,u=get_parallel_info_s_k(self.wf,s,k)

        if not mpi.parallel:
            psit_G = self.wf.kpt_u[u].psit_nG[n]
            return psit_G*c

        if self.wf.kpt_comm.rank==kpt_rank:
            psit_G =  self.wf.kpt_u[u].psit_nG[n]
            a_G = self.gd.collect(psit_G)

            # domain master send this to the global master
            if self.domain.comm.rank == 0:
                self.wf.kpt_comm.send(a_G,MASTER)

        if (mpi.rank==MASTER) and (self.wf.kpt_comm.size>1):
            # allocate full wavefunction and receive 
            psit_G =  self.wf.kpt_u[0].psit_nG[0]
            a_G = num.zeros(psit_G.shape[:-3] + tuple(self.gd.N_c), psit_G.typecode())
            self.wf.kpt_comm.receive(a_G,kpt_rank)
        
        if mpi.rank==MASTER:
            return a_G*c
        else: 
            return


    def get_wannier_integrals(self, i):
        """Calculate integrals for maximally localized Wannier functions."""
        assert self.nspins == 1 and self.wf.typecode is num.Float
        return self.gd.wannier_matrix(self.wf.kpt_u[0].psit_nG, i)

    def get_xc_difference(self, xcname):
        """Calculate non-seflconsistent XC-energy difference."""
        assert self.xcfunc.gga, 'Must be a GGA calculation' # XXX
        oldxcname = self.xcfunc.get_xc_name()
        self.xcfunc.set_xc_functional(xcname)
        
        v_g = self.finegd.new_array()  # not used for anything!
        if self.nspins == 2:
            Exc = self.xc.get_energy_and_potential(self.nt_sg[0], v_g, 
                                                   self.nt_sg[1], v_g)
        else:
            Exc = self.xc.get_energy_and_potential(self.nt_sg[0], v_g)

        for nucleus in self.my_nuclei:
            D_sp = nucleus.D_sp
            H_sp = num.zeros(D_sp.shape, num.Float) # not used for anything!
            Exc += nucleus.setup.xc.calculate_energy_and_derivatives(D_sp,
                                                                     H_sp)

        Exc = self.domain.comm.sum(Exc)

        self.xcfunc.set_xc_functional(oldxcname)
        
        return self.Ha * (Exc - self.Exc)
    
    def get_grid_spacings(self):
        return self.a0 * self.gd.h_c
