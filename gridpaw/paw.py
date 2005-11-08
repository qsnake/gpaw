# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class."""

import sys
import os
from math import pi, sqrt, log
import time

import Numeric as num

from gridpaw import debug, sigusr1
from gridpaw.grid_descriptor import GridDescriptor
from gridpaw.pair_potential import PairPotential
from gridpaw.poisson_solver import PoissonSolver
from gridpaw.rotation import rotation
from gridpaw.density_mixer import Mixer, MixerSum
from gridpaw.utilities.complex import cc, real
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
    Instantiating such an object by hand is not recommended.  Use the
    ``create_paw_object()`` helper-function instead (it will supply
    many default values) - this function is used py the ASE-calculator
    interface.

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
     ``Ekin``    Kinetic energy
     ``Epot``    Potential energy
     ``Etot``    Total energy
     ``Etotold`` Total energy from last iteration
     ``Exc``     Exchange-Correlation energy
     ``S``       Entropy
     ``Ebar``    Should be close to zero!
     ``F_ai``    Forces
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
    those used for parallelization_:

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
    .. _ASE unit: https://wiki.fysik.dtu.dk/ase/Units
    """
    
    def __init__(self, a0, Ha,
                 setups, nuclei, domain, N_c, symmetry, xcfunc,
                 nvalence, nbands, nspins, kT,
                 typecode, bzk_kc, ibzk_kc, weights_k,
                 order, usesymm, mix, old, fixdensity, idiotproof,
                 # Parallel stuff:
                 myspins,
                 myibzk_kc, myweights_k, kpt_comm,
                 out):
        
        self.timer = Timer()

        self.a0 = a0  # Bohr and...
        self.Ha = Ha  # Hartree units are used internally
        self.setups = setups
        self.nuclei = nuclei
        self.domain = domain
        self.symmetry = symmetry
        self.xcfunc = xcfunc
        self.nspins = nspins
        self.usesymm = usesymm
        self.fixdensity = fixdensity
        self.idiotproof = idiotproof

        self.set_output(out)
        
        # Construct grid descriptors for coarse grids (wave functions) and
        # fine grids (densities and potentials):
        self.gd = GridDescriptor(domain, N_c)
        self.finegd = GridDescriptor(domain, 2 * N_c)
        if not self.gd.is_healthy():
            self.warn(
                'VERY ANISOTROPIC GRIDSPACINGS: ' + str(self.a0 * self.gd.h_c))

        if mpi.rank == MASTER:
            # Forces for all atoms:
            self.F_ac = num.zeros((len(nuclei), 3), num.Float)
            
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
        self.pairpot = PairPotential(domain, setups)

        output.print_info(self)

    def initialize_density_and_wave_functions(self, hund, magmom_a,
                                              density=True,
                                              wave_functions=True):
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

        self.Etot = 9999.9
        self.converged = False
        
    def __del__(self):
        # Remove cyclic references: ????  XXXX
        for nucleus in self.nuclei:
            del nucleus.neighbors

        for kpt in self.wf.kpt_u:
            self.timer.add(kpt.timer)
        self.timer.write(self.out)

    def set_convergence_criteria(self, tol):
        if hasattr(self, 'tolerance') and tol < self.tolerance:
            self.converged = False
        self.tolerance = tol
        
    def set_output(self, out):
        if mpi.rank != MASTER:                
            if debug:
                out = sys.stderr
            else:
                out = DownTheDrain()
        self.out = out

    def calculate(self, pos_ac, cell_c, angle):
        pos_ac = pos_ac / self.a0
        cell_c = cell_c / self.a0
        self.set_positions(pos_ac)

        assert not self.converged

        out = self.out
        
        print >> out, """\
                       log10     total     iterations:
              time     error     energy    fermi  poisson  magmom"""

        niter = 0
        while not self.converged:
            self.converge(niter)
            # Output from each iteration:
            t = time.localtime()
            out.write('iter: %4d %3d:%02d:%02d %6.1f %13.7f %4d %7d' %
                      (niter,
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
            niter += 1
            if niter > 1240:
                raise RuntimeError('Did not converge!')

        self.niter = niter

        output.print_converged(self)

    def get_potential_energy(self, force_consistent):
        if force_consistent:
            # Free energy:
            return self.Ha * self.Etot
        else:
            # Energy extrapolated to zero Kelvin:
            return self.Ha * (self.Etot + 0.5 * self.S)

    def set_positions(self, pos_ac):
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
                rank = self.domain.rank(spos_c)
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
                
            neighborlist_update = self.pairpot.update(pos_ac, self.nuclei)

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

    def distribute_atoms(self):
        nspins = self.nspins
        nmykpts = self.wf.nmykpts
        nbands = self.wf.nbands

        if self.domain.comm.size == 1:
            # Serial calculation:
            for nucleus in self.nuclei:
                if nucleus.domain_overlap == NOT_INITIALIZED:
                    nucleus.allocate(nspins, nmykpts, nbands)
                nucleus.domain_overlap = EVERYTHING

            self.my_nuclei = self.nuclei
            self.p_nuclei = self.nuclei
            self.g_nuclei = self.nuclei
            return

        # Parallel calculation:
        natoms = len(self.nuclei)
        domain_overlap_a = num.zeros(natoms, num.Int)
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
            domain_overlap_a[a] = domain_overlap

        domain_overlap_ca = num.zeros((self.domain.comm.size, natoms), num.Int)
        self.domain.comm.all_gather(domain_overlap_a, domain_overlap_ca)

        # Make groups:
        for a, nucleus in enumerate(self.nuclei):
            domain_overlap_c = domain_overlap_ca[:, a]

            # Who owns the atom?
            root = num.argmax(domain_overlap_c)

            g_group = [c for c, b in enumerate(domain_overlap_c) if
                       b >= COMPENSATION_CHARGE]
            g_root = g_group.index(root)
            g_comm = self.domain.comm.new_communicator(
                num.array(g_group, num.Int))

            p_group = [c for c, b in enumerate(domain_overlap_c) if
                       b >= PROJECTOR_FUNCTION]
            p_root = p_group.index(root)
            p_comm = self.domain.comm.new_communicator(
                num.array(p_group, num.Int))

            nucleus.set_communicators(self.domain.comm, root,
                                      g_comm, g_root, p_comm, p_root)
            
    def converge(self, niter):
        if niter == 0:
            # We don't have any occupation numbers.  The initial
            # electron density comes from overlapping atomic densities
            # or from a restart file.  We scale the density to match
            # the compensation charges.

            Nt = self.gd.integrate(self.nt_sG.flat)
            self.calculate_multipole_moments()
            Q = 0.0
            for nuclei in self.my_nuclei:
                Q += nuclei.Q_L[0]
            Q = sqrt(4 * pi) * self.domain.comm.sum(Q)

            # Nt + Q must be zero:
            x = -Q / Nt
            assert 0.93 < x < 1.07, 'x=%f' % x
            self.nt_sG *= x

        wf = self.wf
        
        # Put the calculation of P in kpt.Ortho method? XXXXX
        wf.calculate_projections_and_orthogonalize(self.p_nuclei,
                                                   self.my_nuclei)
        if niter > 0:
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

        if niter == 0:
            for nucleus in self.my_nuclei:
                nucleus.reallocate(wf.nbands)

        # Calculate occupations numbers, and return entropy, number of
        # iteration, and magnetic moment:
        self.nfermi, self.magmom, self.S = \
                self.wf.calculate_occupation_numbers()
        
        dsum = self.domain.comm.sum
        self.Ekin = dsum(self.Ekin) + wf.sum_eigenvalues()
        self.Epot = dsum(self.Epot)
        self.Ebar = dsum(self.Ebar)
        self.Exc = dsum(self.Exc)
        self.Etotold = self.Etot
        self.Etot = self.Ekin + self.Epot + self.Ebar + self.Exc - self.S

        self.error = dsum(wf.calculate_residuals(self.p_nuclei))

        dEtot = abs(self.Etot - self.Etotold)
        de = 1e-8
        if self.error > self.tolerance and dEtot > de and not sigusr1[0]:
            self.timer.start('SD')
            wf.rmm_diis(self.p_nuclei, self.vt_sG)
            self.timer.stop('SD')
        else:
            self.converged = True
            if sigusr1[0]:
                print >> self.out, 'SCF-ITERATIONS STOPPED BY USER!'
                sigusr1[0] = False

    def calculate_atomic_hamiltonians(self):
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
        for nucleus in self.nuclei:
            nucleus.calculate_multipole_moments()
            
    def calculate_potential(self):
        self.rhot_g[:] = self.nt_sg[0]
        if self.nspins == 2:
            self.rhot_g += self.nt_sg[1]

        vt_g = self.vt_sg[0]
        vt_g[:] = 0.0
        for nucleus in self.g_nuclei:
            nucleus.add_hat_potential(vt_g)

        self.Epot = num.dot(vt_g.flat, self.rhot_g.flat) * self.finegd.dv 

        for nucleus in self.p_nuclei:
            nucleus.add_localized_potential(vt_g)

        self.Ebar = num.dot(vt_g.flat, self.rhot_g.flat) * self.finegd.dv 
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

        assert self.finegd.integrate(self.rhot_g.flat) < 0.2
## XXX        self.rhot_g -= sum / self.finegd.arraysize / mpi.size

        # npoisson is the number of iterations:
        self.timer.start('poisson')
        self.npoisson = self.poisson.solve(self.vHt_g, self.rhot_g)
        self.timer.stop('poisson')
        
        self.Epot += 0.5 * num.dot(self.vHt_g.flat,
                                   self.rhot_g.flat) * self.finegd.dv
        self.Ekin = 0.0
        for vt_g, vt_G, nt_G in zip(self.vt_sg, self.vt_sG, self.nt_sG):
            vt_g += self.vHt_g
            self.restrict(vt_g, vt_G)
            self.Ekin -= num.dot(vt_G.flat,
                                (nt_G - self.nct_G).flat) * self.gd.dv

    def warn(self, message):
        print >> self.out, warning(message)
        if self.idiotproof:
            raise RuntimeError, warning

    def get_ibz_kpoints(self):
        return self.ibzk_kc
    
    def get_fermi_level(self):
        e = self.occupation.get_fermi_level()
        if e is None:
            e = 100.0
        return e * self.Ha

    def get_density_array(self):
        c = 1.0 / self.a0**3
        if self.nspins == 2:
            return self.nt_sg * c
        else:
            return self.nt_sg[0] * c

    def get_wave_function_array(self, n, k, s):
        u = s + 2 * k
        c = 1.0 / self.a0**1.5
        return self.kpts[u].psit_nG[n] * c

    def get_wannier_integral(self, i):
        assert self.nspins == 1 and self.nmykpts == 1
        return self.gd.wannier_matrix(self.kpts[0].psit_nG, i)

    def get_magnetic_moment(self):
        return self.magmom

    def get_xc_difference(self, xcname):
        assert self.xcfunc.gga, 'Must be a GGA calculation'
        oldxcname = self.xcfunc.get_xc_name()
        self.xcfunc.set_xc_functional(xcname)
##        xcfunc.set_relativistic(True)
        
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
##        xcfunc.set_relativistic(True)
        
        return self.Ha * (Exc - self.Exc)
    
    def get_cartesian_forces(self):
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

    def get_number_of_iterations(self):
        return self.niter

    def get_nucleus_P_uni(self,nucleus):
        """ return to the master the nucleus with
            domain_overlap = EVERYTHING
        """
        if mpi.rank==MASTER:
            if nucleus.domain_overlap == EVERYTHING:
                return nucleus.P_uni
            else:
                P_uni = nucleus.P_uni.Copy()
                self.domain.comm.receive(P_uni, nucleus.rank)
                return P_uni
        else:
            if nucleus.domain_overlap == EVERYTHING:
                self.domain.comm.send(nucleus.P_uni, MASTER)

    def get_reference_energy(self):
        Eref = 0.0
        for nucleus in self.nuclei:
            Eref += nucleus.setup.E
        return self.Ha * Eref
    
    def write_netcdf(self, filename):
        netcdf.write_netcdf(self, filename)
        
    def initialize_from_netcdf(self, filename):
        netcdf.read_netcdf(self, filename)
        output.plot_atoms(self)
