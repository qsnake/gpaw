"""Module defining  ``Eigensolver`` classes."""

import numpy as npy

from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities.tools import apply_subspace_mask
from gpaw.utilities import unpack


class Eigensolver:
    def __init__(self):
        self.initialized = False
        self.lcao = False

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

        # Soft part of the Hamiltonian times psit
        self.Htpsit_nG = self.gd.empty(self.nbands, self.dtype)

        # Work array for e.g. subspace rotations
        self.work = self.gd.empty(self.nbands, self.dtype)

        # Hamiltonian matrix
        self.H_nn = npy.empty((self.nbands, self.nbands), self.dtype)
        self.initialized = True

    def set_tolerance(self, tolerance):
        """Sets the tolerance for the eigensolver"""

        self.tolerance = tolerance

    def iterate(self, hamiltonian, kpt_u):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement ``iterate_one_k_point`` method for a single iteration of
        a single kpoint.
        """

        error = 0.0
        for kpt in kpt_u:
            error += self.iterate_one_k_point(hamiltonian, kpt)
            
        self.error = self.kpt_comm.sum(error)

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Implemented in subclasses."""
        return 0.0
    
    def diagonalize(self, hamiltonian, kpt, rotate=True):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        ``Htpsit_nG`` is working array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        First, the Hamiltonian (defined by ``kin``, ``vt_sG``, and
        ``my_nuclei``) is applied to the wave functions, then the
        ``H_nn`` matrix is calculated and diagonalized, and finally,
        the wave functions are rotated.  Also the projections
        ``P_uni`` (an attribute of the nuclei) are rotated.
        
        It is assumed that the wave functions ``psit_n`` are orthonormal
        and that the integrals of projector functions and wave functions
        ``P_uni`` are already calculated
        """        
           
        self.timer.start('Subspace diag.')

        if self.nbands != kpt.nbands:
            raise RuntimeError('Bands: %d != %d' % (self.nbands, kpt.nbands))
        
        Htpsit_nG = self.Htpsit_nG
        psit_nG = kpt.psit_nG
        eps_n = kpt.eps_n
        H_nn = self.H_nn

        hamiltonian.kin.apply(psit_nG, Htpsit_nG, kpt.phase_cd)
        
        Htpsit_nG += psit_nG * hamiltonian.vt_sG[kpt.s]

        H_nn[:] = 0.0  # r2k fails without this!
        
        self.timer.start('Non-local xc')
        hamiltonian.xc.xcfunc.apply_non_local(kpt, Htpsit_nG, H_nn)
        self.timer.stop('Non-local xc')
        
        r2k(0.5 * self.gd.dv, psit_nG, Htpsit_nG, 1.0, H_nn)
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            dH_ii = unpack(nucleus.H_sp[kpt.s])
            H_nn += npy.dot(P_ni, npy.inner(dH_ii, P_ni.conj()))

        self.comm.sum(H_nn, kpt.root)

        # Uncouple occupied and unoccupied subspaces
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            apply_subspace_mask(H_nn, kpt.f_n)

        if not rotate:
            self.comm.broadcast(H_nn, kpt.root)
            self.timer.stop('Subspace diag.')
            return
        
        self.timer.start('dsyev/zheev')
        if self.comm.rank == kpt.root:
            info = diagonalize(H_nn, eps_n)
            if info != 0:
                raise RuntimeError, 'Very Bad!!'
        self.timer.stop('dsyev/zheev')

        self.timer.start('bcast H')
        self.comm.broadcast(H_nn, kpt.root)
        self.timer.stop('bcast H')
        self.timer.start('bcast eps')
        self.comm.broadcast(eps_n, kpt.root)
        self.timer.stop('bcast eps')

        # Rotate psit_nG:
        gemm(1.0, psit_nG, H_nn, 0.0, self.work)
        
        # Rotate Htpsit_nG:
        gemm(1.0, Htpsit_nG, H_nn, 0.0, psit_nG)

        #Switch the references
        kpt.psit_nG, self.Htpsit_nG, self.work = self.work, psit_nG, Htpsit_nG
        
        # Rotate P_uni:
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            gemm(1.0, P_ni.copy(), H_nn, 0.0, P_ni)

        # Rotate EXX related stuff
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            hamiltonian.xc.xcfunc.exx.rotate(kpt.u, H_nn)

        self.timer.stop('Subspace diag.')

