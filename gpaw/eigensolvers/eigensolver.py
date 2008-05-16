"""Module defining and eigensolver base-class."""

from math import ceil

import numpy as npy

from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.tools import apply_subspace_mask
from gpaw.utilities import unpack


def blocked_matrix_multiply(a_nG, U_nn, work_nG):
    nbands = len(a_nG)
    b_ng = a_nG.reshape((nbands, -1))
    w_ng = work_nG.reshape((nbands, -1))
    ngpts = b_ng.shape[1]
    blocksize = w_ng.shape[1]
    g1 = 0
    while g1 < ngpts:
        g2 = g1 + blocksize
        if g2 > ngpts:
            g2 = ngpts
        gemm(1.0, b_ng[:, g1:g2], U_nn, 0.0, w_ng[:, :g2 - g1])
        b_ng[:, g1:g2] = w_ng[:, :g2 - g1]
    
class Eigensolver:
    def __init__(self, keep_htpsit=True, nblocks=1):
        self.keep_htpsit = keep_htpsit
        self.nblocks = nblocks
        self.initialized = False
        self.lcao = False
        self.Htpsit_nG = None

    def initialize(self, paw, nbands=None):
        self.timer = paw.timer
        self.kpt_comm = paw.kpt_comm
        self.dtype = paw.dtype
        self.gd = paw.gd
        self.comm = paw.gd.comm
        if nbands is None:
            self.nbands = paw.nbands
        else:
            self.nbands = nbands
            
        self.nbands_converge = paw.input_parameters['convergence']['bands']
        self.set_tolerance(paw.input_parameters['convergence']['eigenstates'])

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(self.gd, paw.hamiltonian.kin,
                                             self.dtype)

        if self.keep_htpsit:
            # Soft part of the Hamiltonian times psit:
            self.Htpsit_nG = self.gd.empty(self.nbands, self.dtype)

        # Work array for e.g. subspace rotations:
        self.blocksize = int(ceil(1.0 * self.nbands / self.nblocks))
        paw.big_work_arrays['work_nG'] = self.gd.empty(self.blocksize,
                                                       self.dtype)
        self.big_work_arrays = paw.big_work_arrays

        # Hamiltonian matrix
        self.H_nn = npy.empty((self.nbands, self.nbands), self.dtype)
        self.initialized = True

    def set_tolerance(self, tolerance):
        """Sets the tolerance for the eigensolver"""

        self.tolerance = tolerance

    def iterate(self, hamiltonian, kpt_u):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement *iterate_one_k_point* method for a single iteration of
        a single kpoint.
        """

        error = 0.0
        for kpt in kpt_u:
            error += self.iterate_one_k_point(hamiltonian, kpt)
            
        self.error = self.kpt_comm.sum(error)

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Implemented in subclasses."""
        return 0.0

    def calculate_hamiltonian_matrix(self, hamiltonian, kpt):
        """Set up the Hamiltonian in the subspace of kpt.psit_nG

        *Htpsit_nG* is a work array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        The Hamiltonian (defined by *kin*, *vt_sG*, and
        *my_nuclei*) is applied to the wave functions, then the
        *H_nn* matrix is calculated.
        
        It is assumed that the wave functions *psit_n* are orthonormal
        and that the integrals of projector functions and wave functions
        *P_uni* are already calculated
        """        
        
        psit_nG = kpt.psit_nG
        H_nn = self.H_nn
        H_nn[:] = 0.0  # r2k can fail without this!

        if self.keep_htpsit:
            Htpsit_nG = self.Htpsit_nG
            hamiltonian.apply(psit_nG, Htpsit_nG, kpt,
                              local_part_only=True,
                              calculate_projections=False)
        
            hamiltonian.xc.xcfunc.apply_non_local(kpt, Htpsit_nG, H_nn)
        
            r2k(0.5 * self.gd.dv, psit_nG, Htpsit_nG, 1.0, H_nn)

        else:
            Htpsit_nG = self.work_nG
            n1 = 0
            while n1 < self.nbands:
                n2 = n1 + self.blocksize
                if n2 > self.nbands:
                    n2 = self.nbands
                hamiltonian.apply(psit_nG[n1:n2], Htpsit_nG[:n2 - n1], kpt,
                                  local_part_only=True)
        
                r2k(0.5 * self.gd.dv, psit_nG[n1:], Htpsit_nG, 0.0,
                    H_nn[n1:, n1:n2])
                n1 = n2
                
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            dH_ii = unpack(nucleus.H_sp[kpt.s])
            H_nn += npy.dot(P_ni, npy.inner(dH_ii, P_ni.conj()))

        self.comm.sum(H_nn, kpt.root)

        # Uncouple occupied and unoccupied subspaces:
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            apply_subspace_mask(H_nn, kpt.f_n)

    def subspace_diagonalize(self, hamiltonian, kpt):
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
        *P_uni* are already calculated
        """        
           
        self.timer.start('Subspace diag.')

        self.calculate_hamiltonian_matrix(hamiltonian, kpt)
        H_nn = self.H_nn

        eps_n = kpt.eps_n

        self.timer.start('dsyev/zheev')
        if self.comm.rank == kpt.root:
            info = diagonalize(H_nn, eps_n)
            if info != 0:
                raise RuntimeError('Failed to diagonalize: info=%d' % info)
        self.timer.stop('dsyev/zheev')

        U_nn = H_nn
        del H_nn
        
        self.comm.broadcast(U_nn, kpt.root)
        self.comm.broadcast(eps_n, kpt.root)

        work_nG = self.big_work_arrays['work_nG']
        psit_nG = kpt.psit_nG
        
        # Rotate psit_nG:
        if self.nblocks == 1:
            gemm(1.0, psit_nG, U_nn, 0.0, work_nG)
            kpt.psit_nG = work_nG
            work_nG = psit_nG
            self.big_work_arrays['work_nG'] = work_nG
        else:
            blocked_matrix_multiply(psit_nG, U_nn, work_nG)

        if self.keep_htpsit:
            # Rotate Htpsit_nG:
            Htpsit_nG = self.Htpsit_nG
            gemm(1.0, Htpsit_nG, U_nn, 0.0, work_nG)
            self.Htpsit_nG = work_nG
            work_nG = Htpsit_nG
            self.big_work_arrays['work_nG'] = work_nG

        # Rotate P_uni:
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), U_nn, 0.0, P_ni)

        # Rotate EXX related stuff
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            hamiltonian.xc.xcfunc.exx.rotate(kpt.u, U_nn)

        self.timer.stop('Subspace diag.')

