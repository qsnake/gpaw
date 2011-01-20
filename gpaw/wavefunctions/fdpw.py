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


class FDPWWaveFunctions(WaveFunctions):
    """Base class for finite-difference and planewave classes."""
    def __init__(self, diagksl, orthoksl, initksl, *args, **kwargs):
        WaveFunctions.__init__(self, *args, **kwargs)

        self.diagksl = diagksl
        self.orthoksl = orthoksl
        self.initksl = initksl

        self.set_orthonormalized(False)

        self.overlap = Overlap(orthoksl, self.timer)

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)
        if not self.gamma:
            self.pt.set_k_points(self.kd.ibzk_qc)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac):
        WaveFunctions.set_positions(self, spos_ac)
        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def initialize(self, density, hamiltonian, spos_ac):
        if self.kpt_u[0].psit_nG is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             cut=True)
            if not self.gamma:
                basis_functions.set_k_points(self.kd.ibzk_qc)
            basis_functions.set_positions(spos_ac)
        elif isinstance(self.kpt_u[0].psit_nG, TarFileReference):
            self.initialize_wave_functions_from_restart_file()

        if self.kpt_u[0].psit_nG is not None:
            density.initialize_from_wavefunctions(self)
        elif density.nt_sG is None:
            density.initialize_from_atomic_densities(basis_functions)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xc.type == 'GLLB':
                hamiltonian.xc.initialize_from_atomic_orbitals(
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
        lcaowfs = LCAOWaveFunctions(lcaoksl, self.gd, self.nvalence,
                                    self.setups, lcaobd, self.dtype,
                                    self.world, self.kd)
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
        """Generate random wave functions."""

        gpts = self.gd.N_c[0]*self.gd.N_c[1]*self.gd.N_c[2]
        
        if self.nbands < gpts/64:
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
        
        elif gpts/64 <= self.nbands < gpts/8:
            gd1 = self.gd.coarsen()

            psit_G1 = gd1.empty(dtype=self.dtype)

            interpolate1 = Transformer(gd1, self.gd, 1, self.dtype).apply

            shape = tuple(gd1.n_c)
            scale = np.sqrt(12 / abs(np.linalg.det(gd1.cell_cv)))

            old_state = np.random.get_state()

            np.random.seed(4 + self.world.rank)

            for kpt in self.kpt_u:
                for psit_G in kpt.psit_nG[nao:]:
                    if self.dtype == float:
                        psit_G1[:] = (np.random.random(shape) - 0.5) * scale
                    else:
                        psit_G1.real = (np.random.random(shape) - 0.5) * scale
                        psit_G1.imag = (np.random.random(shape) - 0.5) * scale

                    interpolate1(psit_G1, psit_G, kpt.phase_cd)
            np.random.set_state(old_state)
               
        else:
            shape = tuple(self.gd.n_c)
            scale = np.sqrt(12 / abs(np.linalg.det(self.gd.cell_cv)))

            old_state = np.random.get_state()

            np.random.seed(4 + self.world.rank)

            for kpt in self.kpt_u:
                for psit_G in kpt.psit_nG[nao:]:
                    if self.dtype == float:
                        psit_G[:] = (np.random.random(shape) - 0.5) * scale
                    else:
                        psit_G.real = (np.random.random(shape) - 0.5) * scale
                        psit_G.imag = (np.random.random(shape) - 0.5) * scale

            np.random.set_state(old_state)        

    def orthonormalize(self):
        for kpt in self.kpt_u:
            self.overlap.orthonormalize(self, kpt)
        self.set_orthonormalized(True)

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
        self.matrixoperator.estimate_memory(mem.subnode('Overlap op'),
                                            self.dtype)
