"""Module defining and eigensolver base-class."""

from math import ceil

import numpy as np

from gpaw.fd_operators import Laplace
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.tools import apply_subspace_mask
from gpaw.utilities import unpack
from gpaw import debug, extra_parameters


class Eigensolver:
    def __init__(self, keep_htpsit=True, blocksize=1):
        self.keep_htpsit = keep_htpsit
        self.initialized = False
        self.Htpsit_nG = None
        self.error = np.inf
        self.blocksize = blocksize
        
    def initialize(self, wfs):
        self.timer = wfs.timer
        self.world = wfs.world
        self.kpt_comm = wfs.kpt_comm
        self.band_comm = wfs.band_comm
        self.dtype = wfs.dtype
        self.bd = wfs.bd
        self.gd = wfs.wd
        self.ksl = wfs.diagksl
        self.nbands = wfs.nbands
        self.mynbands = wfs.mynbands
        self.operator = wfs.matrixoperator

        if self.mynbands != self.nbands or self.operator.nblocks != 1:
            self.keep_htpsit = False

        # Preconditioner for the electronic gradients:
        self.preconditioner = wfs.make_preconditioner(self.blocksize)

        if self.keep_htpsit:
            # Soft part of the Hamiltonian times psit:
            self.Htpsit_nG = self.gd.zeros(self.nbands, self.dtype)

        for kpt in wfs.kpt_u:
            if kpt.eps_n is None:
                kpt.eps_n = np.empty(self.mynbands)
        
        self.initialized = True

    def iterate(self, hamiltonian, wfs):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement *iterate_one_k_point* method for a single iteration of
        a single kpoint.
        """

        if not self.initialized:
            self.initialize(wfs)

        if not self.preconditioner.allocated:
            self.preconditioner.allocate()

        if not wfs.orthonormalized:
            wfs.orthonormalize()
            
        error = 0.0
        for kpt in wfs.kpt_u:
            error += self.iterate_one_k_point(hamiltonian, wfs, kpt)

        wfs.orthonormalize()

        self.error = self.band_comm.sum(self.kpt_comm.sum(error))

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Implemented in subclasses."""
        raise NotImplementedError

    def calculate_residuals(self, kpt, wfs, hamiltonian, psit_xG, P_axi, eps_x,
                            R_xG, n_x=None, calculate_change=False):
        """Calculate residual.

        From R=Ht*psit calculate R=H*psit-eps*S*psit."""
        
        for R_G, eps, psit_G in zip(R_xG, eps_x, psit_xG):
            axpy(-eps, psit_G, R_G)

        c_axi = {}
        for a, P_xi in P_axi.items():
            dH_ii = unpack(hamiltonian.dH_asp[a][kpt.s])
            dO_ii = hamiltonian.setups[a].dO_ii
            c_xi = (np.dot(P_xi, dH_ii) -
                    np.dot(P_xi * eps_x[:, np.newaxis], dO_ii))
            c_axi[a] = c_xi
        hamiltonian.xc.add_correction(kpt, psit_xG, R_xG, P_axi, c_axi, n_x,
                                      calculate_change)
        wfs.pt.add(R_xG, c_axi, kpt.q)
        
    def subspace_diagonalize(self, hamiltonian, wfs, kpt, rotate=True):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        *Htpsit_nG* is a work array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        First, the Hamiltonian (defined by *kin*, *vt_sG*, and
        *my_nuclei*) is applied to the wave functions, then the
        *H_nn* matrix is calculated and diagonalized, and finally,
        the wave functions are rotated.  Also the projections
        *P_uni* (an attribute of the nuclei) are rotated.

        It is assumed that the wave functions *psit_n* are orthonormal
        and that the integrals of projector functions and wave functions
        *P_uni* are already calculated.
        """

        if self.band_comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        self.timer.start('Subspace diag')

        psit_nG = kpt.psit_nG
        P_ani = kpt.P_ani

        if self.keep_htpsit:
            Htpsit_xG = self.Htpsit_nG
        else:
            Htpsit_xG = self.operator.suggest_temporary_buffer(psit_nG.dtype)

        def H(psit_xG):
            wfs.apply_pseudo_hamiltonian(kpt, hamiltonian, psit_xG, Htpsit_xG)
            hamiltonian.xc.apply_orbital_dependent_hamiltonian(
                kpt, psit_xG, Htpsit_xG, hamiltonian.dH_asp)
            return Htpsit_xG

        def dH(a, P_ni):
            return np.dot(P_ni, unpack(hamiltonian.dH_asp[a][kpt.s]))

        self.timer.start('calc_matrix')
        H_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
                                                       H, dH)
        hamiltonian.xc.correct_hamiltonian_matrix(kpt, H_nn)
        self.timer.stop('calc_matrix')

        diagonalization_string = repr(self.ksl)
        wfs.timer.start(diagonalization_string)
        self.ksl.diagonalize(H_nn, kpt.eps_n)
        # H_nn now contains the result of the diagonalization.
        wfs.timer.stop(diagonalization_string)

        if not rotate:
            self.timer.stop('Subspace diag')
            return

        self.timer.start('rotate_psi')
        kpt.psit_nG = self.operator.matrix_multiply(H_nn, psit_nG, P_ani)
        if self.keep_htpsit:
            self.Htpsit_nG = self.operator.matrix_multiply(H_nn, Htpsit_xG)

        # Rotate orbital dependent XC stuff:
        hamiltonian.xc.rotate(kpt, H_nn)

        self.timer.stop('rotate_psi')

        self.timer.stop('Subspace diag')

    def estimate_memory(self, mem, gd, dtype, mynbands, nbands):
        gridmem = gd.bytecount(dtype)

        keep_htpsit = self.keep_htpsit and (mynbands == nbands)

        if keep_htpsit:
            mem.subnode('Htpsit', nbands * gridmem)
        else:
            mem.subnode('No Htpsit', 0)

        # mem.subnode('U_nn', nbands*nbands*mem.floatsize)
        mem.subnode('eps_n', nbands*mem.floatsize)
        mem.subnode('Preconditioner', 4 * gridmem)
        mem.subnode('Work', gridmem)

