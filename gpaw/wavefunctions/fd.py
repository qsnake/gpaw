import numpy as np

from gpaw.eigensolvers import get_eigensolver
from gpaw.overlap import Overlap
from gpaw.fd_operators import Laplace
from gpaw.lfc import LocalizedFunctionsCollection as LFC
from gpaw.utilities import unpack
from gpaw.io.tar import TarFileReference
from gpaw.lfc import BasisFunctions
from gpaw.utilities.blas import axpy
from gpaw.transformers import Transformer
from gpaw.fd_operators import Gradient
from gpaw.band_descriptor import BandDescriptor
from gpaw import extra_parameters
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class GridWaveFunctions(WaveFunctions):
    def __init__(self, stencil, diagksl, orthoksl, initksl, *args, **kwargs):
        WaveFunctions.__init__(self, *args, **kwargs)
        # Kinetic energy operator:
        self.kin = Laplace(self.gd, -0.5, stencil, self.dtype, allocate=False)
        self.diagksl = diagksl
        self.orthoksl = orthoksl
        self.initksl = initksl
        self.set_orthonormalized(False)
        self.overlap = Overlap(self) # Object needed by memory estimate
        # (it has to be overwritten on each initialize() anyway, because of
        # weird object reuse issues, but we don't care)

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)
        self.pt = LFC(self.gd, [setup.pt_j for setup in setups],
                      self.kpt_comm, dtype=self.dtype, forces=True)
        if not self.gamma:
            self.pt.set_k_points(self.ibzk_qc)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac):
        if not self.kin.is_allocated():
            self.kin.allocate()
        WaveFunctions.set_positions(self, spos_ac)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def initialize(self, density, hamiltonian, spos_ac):
        self.overlap = Overlap(self)
        if self.kpt_u[0].psit_nG is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             cut=True)
            if not self.gamma:
                basis_functions.set_k_points(self.ibzk_qc)
            basis_functions.set_positions(spos_ac)
        elif isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            self.initialize_wave_functions_from_restart_file()

        if self.kpt_u[0].psit_nG is not None:
            density.initialize_from_wavefunctions(self)
        elif density.nt_sG is None:
            density.initialize_from_atomic_densities(basis_functions)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xcfunc.gllb:
                hamiltonian.xcfunc.xc.initialize_from_atomic_orbitals(
                    basis_functions)
        else: # XXX???
            # We didn't even touch density, but some combinations in paw.set()
            # will make it necessary to do this for some reason.
            density.calculate_normalized_charges_and_mix()
        hamiltonian.update(density)
                
        if self.kpt_u[0].psit_nG is None:
            self.initialize_wave_functions_from_basis_functions(
                basis_functions, density, hamiltonian, spos_ac)

    def initialize_wave_functions_from_basis_functions(self,
                                                       basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        if 0:
            self.timer.start('Random wavefunction initialization')
            for kpt in self.kpt_u:
                kpt.psit_nG = self.gd.zeros(self.mynbands, self.dtype)
                if extra_parameters.get('sic'):
                    kpt.W_nn = np.zeros((self.nbands, self.nbands),
                                        dtype=self.dtype)
            self.random_wave_functions(0)
            self.timer.stop('Random wavefunction initialization')
            return

        self.timer.start('LCAO initialization')
        lcaoksl, lcaobd = self.initksl, self.initksl.bd
        lcaowfs = LCAOWaveFunctions(lcaoksl, self.gd, self.nspins,
                                    self.nvalence, self.setups, lcaobd,
                                    self.dtype, self.world, self.kpt_comm,
                                    self.gamma, self.bzk_kc, self.ibzk_kc,
                                    self.weight_k, self.symmetry)
        lcaowfs.basis_functions = basis_functions
        lcaowfs.timer = self.timer
        self.timer.start('Set positions (LCAO WFS)')
        lcaowfs.set_positions(spos_ac)
        self.timer.stop('Set positions (LCAO WFS)')

        eigensolver = get_eigensolver('lcao', 'lcao')
        eigensolver.initialize(self.gd, self.dtype, self.setups.nao, lcaoksl)

        # XXX when density matrix is properly distributed, be sure to
        # update the density here also
        eigensolver.iterate(hamiltonian, lcaowfs)

        # Transfer coefficients ...
        for kpt, lcaokpt in zip(self.kpt_u, lcaowfs.kpt_u):
            kpt.C_nM = lcaokpt.C_nM

        # and get rid of potentially big arrays early:
        del eigensolver, lcaowfs

        self.timer.start('LCAO to grid')
        for kpt in self.kpt_u:
            kpt.psit_nG = self.gd.zeros(self.mynbands, self.dtype)
            if extra_parameters.get('sic'):
                kpt.W_nn = np.zeros((self.nbands, self.nbands),
                                    dtype=self.dtype)
            basis_functions.lcao_to_grid(kpt.C_nM, 
                                         kpt.psit_nG[:lcaobd.mynbands], kpt.q)
            kpt.C_nM = None
        self.timer.stop('LCAO to grid')

        if self.mynbands > lcaobd.mynbands:
            # Add extra states.  If the number of atomic orbitals is
            # less than the desired number of bands, then extra random
            # wave functions are added.
            self.random_wave_functions(lcaobd.mynbands)
        self.timer.stop('LCAO initialization')

    def initialize_wave_functions_from_restart_file(self):
        if not isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            return

        # Calculation started from a restart file.  Copy data
        # from the file to memory:
        for kpt in self.kpt_u:
            file_nG = kpt.psit_nG
            kpt.psit_nG = self.gd.empty(self.mynbands, self.dtype)
            if extra_parameters.get('sic'):
                kpt.W_nn = np.zeros((self.nbands, self.nbands),
                                    dtype=self.dtype)
            # Read band by band to save memory
            for n, psit_G in enumerate(kpt.psit_nG):
                if self.gd.comm.rank == 0:
                    big_psit_G = np.array(file_nG[n][:], self.dtype)
                else:
                    big_psit_G = None
                self.gd.distribute(big_psit_G, psit_G)
        
    def random_wave_functions(self, nao):
        """Generate random wave functions"""

        gd1 = self.gd.coarsen()
        gd2 = gd1.coarsen()

        psit_G1 = gd1.empty(dtype=self.dtype)
        psit_G2 = gd2.empty(dtype=self.dtype)

        interpolate2 = Transformer(gd2, gd1, 1, self.dtype).apply
        interpolate1 = Transformer(gd1, self.gd, 1, self.dtype).apply

        shape = tuple(gd2.n_c)

        scale = np.sqrt(12 / abs(np.linalg.det(gd2.cell_cv)))

        old_state = np.random.get_state()
        np.random.seed(4 + self.world.rank)

        for kpt in self.kpt_u:
            for psit_G in kpt.psit_nG[nao:]:
                if self.dtype == float:
                    psit_G2[:] = (np.random.random(shape) - 0.5) * scale
                else:
                    psit_G2.real = (np.random.random(shape) - 0.5) * scale
                    psit_G2.imag = (np.random.random(shape) - 0.5) * scale
                    
                interpolate2(psit_G2, psit_G1, kpt.phase_cd)
                interpolate1(psit_G1, psit_G, kpt.phase_cd)
        np.random.set_state(old_state)

    #def add_to_density_from_k_point(self, nt_sG, kpt):
    #    self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)

    def add_orbital_density(self, nt_G, kpt, n):
        if self.dtype == float:
            axpy(1.0, kpt.psit_nG[n]**2, nt_G)
        else:
            axpy(1.0, kpt.psit_nG[n].real**2, nt_G)
            axpy(1.0, kpt.psit_nG[n].imag**2, nt_G)

    def add_to_density_from_k_point_with_occupation(self, nt_sG, kpt, f_n):
        # Used in calculation of response part of GLLB-potential
        nt_G = nt_sG[kpt.s]
        if self.dtype == float:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G**2, nt_G)
        else:
            for f, psit_G in zip(f_n, kpt.psit_nG):
                axpy(f, psit_G.real**2, nt_G)
                axpy(f, psit_G.imag**2, nt_G)

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands), dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            for d_n, psi0_G in zip(d_nn, kpt.psit_nG):
                for d, psi_G in zip(d_n, kpt.psit_nG):
                    if abs(d) > 1.e-12:
                        nt_G += (psi0_G.conj() * d * psi_G).real

    def add_to_kinetic_density_from_k_point(self, taut_G, kpt):
        """Add contribution to pseudo kinetic energy density."""

        if isinstance(kpt.psit_nG, TarFileReference):
            raise RuntimeError('Wavefunctions have not been initialized.')

        d_c = [Gradient(self.gd, c, dtype=self.dtype).apply for c in range(3)]
        dpsit_G = self.gd.empty(dtype=self.dtype)
        if self.dtype == float:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for c in range(3):
                    d_c[c](psit_G, dpsit_G)
                    axpy(0.5*f, dpsit_G**2, taut_G) #taut_G += 0.5*f*dpsit_G**2
        else:
            for f, psit_G in zip(kpt.f_n, kpt.psit_nG):
                for c in range(3):
                    d_c[c](psit_G, dpsit_G, kpt.phase_cd)
                    taut_G += 0.5 * f * (dpsit_G.conj() * dpsit_G).real

        # Hack used in delta-scf calculations:
        if hasattr(kpt, 'c_on'):
            assert self.bd.comm.size == 1
            d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands), dtype=complex)
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn += ne * np.outer(c_n.conj(), c_n)
            dwork_G = self.gd.empty(dtype=self.dtype)
            if self.dtype == float:
                for d_n, psit0_G in zip(d_nn, kpt.psit_nG):
                    for c in range(3):
                        d_c[c](psit0_G, dpsit_G)
                        for d, psit_G in zip(d_n, kpt.psit_nG):
                            if abs(d) > 1.e-12:
                                d_c[c](psit_G, dwork_G)
                                axpy(0.5*d, dpsit_G * dwork_G, taut_G) #taut_G += 0.5*f*dpsit_G*dwork_G
            else:
                for d_n, psit0_G in zip(d_nn, kpt.psit_nG):
                    for c in range(3):
                        d_c[c](psit0_G, dpsit_G, kpt.phase_cd)
                        for d, psit_G in zip(d_n, kpt.psit_nG):
                            if abs(d) > 1.e-12:
                                d_c[c](psit_G, dwork_G, kpt.phase_cd)
                                taut_G += 0.5 * (dpsit_G.conj() * d * dwork_G).real

    def orthonormalize(self):
        for kpt in self.kpt_u:
            self.overlap.orthonormalize(self, kpt)
        self.set_orthonormalized(True)

    def initialize2(self, paw):
        khjgkjhgkhjg
        hamiltonian = paw.hamiltonian
        density = paw.density
        eigensolver = paw.eigensolver
        assert not eigensolver.lcao
        self.overlap = paw.overlap
        if not eigensolver.initialized:
            eigensolver.initialize(paw)
        if not self.initialized:
            if self.kpt_u[0].psit_nG is None:
                paw.text('Atomic orbitals used for initialization:', paw.nao)
                if paw.nbands > paw.nao:
                    paw.text('Random orbitals used for initialization:',
                             paw.nbands - paw.nao)
            
                # Now we should find out whether init'ing from file or
                # something else
                self.initialize_wave_functions_from_atomic_orbitals(paw)

            else:
                self.initialize_wave_functions_from_restart_file(paw)

    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_av.fill(0.0)
        F_aniv = self.pt.dict(self.bd.mynbands, derivative=True)
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = hamiltonian.setups[a].dO_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q) #XXX again
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for a, F_niv in F_aniv.items():
                    F_niv = F_niv.conj()
                    dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
                    Q_ni = np.dot(d_nn, kpt.P_ani[a])
                    F_vii = np.dot(np.dot(F_niv.transpose(), Q_ni), dH_ii)
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    dO_ii = hamiltonian.setups[a].dO_ii
                    F_vii -= np.dot(np.dot(F_niv.transpose(), Q_ni), dO_ii)
                    F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

        self.bd.comm.sum(F_av, 0)

        if self.bd.comm.rank == 0:
            self.kpt_comm.sum(F_av, 0)

    def _get_wave_function_array(self, u, n):
        psit_nG = self.kpt_u[u].psit_nG
        if psit_nG is None:
            raise RuntimeError('This calculator has no wave functions!')
        return psit_nG[n][:] # dereference possible tar-file content

    def write_wave_functions(self, writer):
        try:
            from gpaw.io.hdf5 import Writer as HDF5Writer
        except ImportError:
            hdf5 = False
        else:
            hdf5 = isinstance(writer, HDF5Writer)
            
        if self.world.rank == 0 or hdf5:
            writer.add('PseudoWaveFunctions',
                       ('nspins', 'nibzkpts', 'nbands',
                        'ngptsx', 'ngptsy', 'ngptsz'),
                       dtype=self.dtype)

        if hdf5:
            for kpt in self.kpt_u:
                indices = [kpt.s, kpt.k]
                indices.append(self.bd.get_slice())
                indices += self.gd.get_slice()
                writer.fill(kpt.psit_nG, parallel=True, *indices)
        else:
            for s in range(self.nspins):
                for k in range(self.nibzkpts):
                    for n in range(self.nbands):
                        psit_G = self.get_wave_function_array(n, k, s)
                        if self.world.rank == 0:
                            writer.fill(psit_G, s, k, n)

    def estimate_memory(self, mem):
        gridbytes = self.gd.bytecount(self.dtype)
        mem.subnode('Arrays psit_nG', 
                    len(self.kpt_u) * self.mynbands * gridbytes)
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'), self.gd,
                                         self.dtype, self.mynbands,
                                         self.nbands)
        self.pt.estimate_memory(mem.subnode('Projectors'))
        self.overlap.estimate_memory(mem.subnode('Overlap op'), self.dtype)
        self.kin.estimate_memory(mem.subnode('Kinetic operator'))
