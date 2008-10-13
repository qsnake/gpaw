"""Module defining and eigensolver base-class."""

from math import ceil

import numpy as npy

from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.tools import apply_subspace_mask
from gpaw.utilities import unpack
from gpaw.mpi import run


def blocked_matrix_multiply(A_nG, U_nn, work_nG):
    """Inplace matrix multipication.

    Perform an inplace multiplication of *A_nG* and *U_nn* using
    *work_nG* as work-space::

             __
            \
      A   <- )  A   U
       nG   /__  mG  nm
             m
             
    """
    
    nbands = len(A_nG)
    b_ng = A_nG.reshape((nbands, -1))
    ngpts = b_ng.shape[1]
    nbands0 = work_nG.shape[0]
    ngpts0 = ngpts * nbands0 // (2 * nbands)
    w = work_nG.reshape((-1,))[:2 * nbands * ngpts0]
    w1_nq, w2_nq = w.reshape(2, nbands, ngpts0)
    g1 = 0
    while g1 < ngpts:
        g2 = g1 + ngpts0
        print g1,g2
        if g2 <= ngpts:
            w1_nq[:] = b_ng[:, g1:g2]
            gemm(1.0, w1_nq, U_nn, 0.0, w2_nq)
            b_ng[:, g1:g2] = w2_nq
            g1 = g2
        else:
            print '*'
            w = work_nG.reshape((-1,))[:2 * nbands * (ngpts - g1)]
            w1_nq, w2_nq = w.reshape(2, nbands, ngpts - g1)
            w1_nq[:] = b_ng[:, g1:]
            gemm(1.0, w1_nq, U_nn, 0.0, w2_nq)
            b_ng[:, g1:] = w2_nq
            break


class Eigensolver:
    def __init__(self, keep_htpsit=True, nblocks=1):
        self.keep_htpsit = keep_htpsit
        self.nblocks = nblocks
        self.initialized = False
        self.lcao = False
        self.Htpsit_nG = None
        self.work_In = None
        self.H_pnn = None

    def initialize(self, paw):
        self.timer = paw.timer
        self.kpt_comm = paw.kpt_comm
        self.band_comm = paw.band_comm
        self.dtype = paw.dtype
        self.gd = paw.gd
        self.comm = paw.gd.comm
        self.nbands = paw.nbands
        self.nmybands = paw.nmybands
        
        if self.nmybands != self.nbands:
            self.keep_htpsit = False

        self.eps_n = npy.empty(self.nbands)

        self.nbands_converge = paw.input_parameters['convergence']['bands']
        self.set_tolerance(paw.input_parameters['convergence']['eigenstates'])

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(self.gd, paw.hamiltonian.kin,
                                             self.dtype)

        if self.keep_htpsit:
            # Soft part of the Hamiltonian times psit:
            self.Htpsit_nG = self.gd.empty(self.nbands, self.dtype)

        # Work array for e.g. subspace rotations:
        self.blocksize = int(ceil(1.0 * self.nmybands / self.nblocks))
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
            
        self.error = self.band_comm.sum(self.kpt_comm.sum(error))

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
        
        if self.band_comm.size > 1:
            return self.calculate_hamiltonian_matrix2(hamiltonian, kpt)

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
            Htpsit_nG = self.big_work_arrays['work_nG']
            n1 = 0
            while n1 < self.nbands:
                n2 = n1 + self.blocksize
                if n2 > self.nbands:
                    n2 = self.nbands
                hamiltonian.apply(psit_nG[n1:n2], Htpsit_nG[:n2 - n1], kpt,
                                  local_part_only=True)
        
                gemm(self.gd.dv, Htpsit_nG, psit_nG[n1:], 0.0,
                     H_nn[n1:, n1:n2], 'c')
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
        *P_uni* are already calculated.
        """        
           
        self.timer.start('Subspace diag.')

        self.calculate_hamiltonian_matrix(hamiltonian, kpt)
        H_nn = self.H_nn
        band_comm = self.band_comm

        self.timer.start('dsyev/zheev')
        if self.comm.rank == kpt.root:
            if band_comm.rank == 0:
                info = diagonalize(H_nn, self.eps_n)
                if info != 0:
                    raise RuntimeError('Failed to diagonalize: info=%d' % info)
                
            band_comm.scatter(self.eps_n, kpt.eps_n, 0)
            band_comm.broadcast(H_nn, 0)

        self.timer.stop('dsyev/zheev')
        U_nn = H_nn
        del H_nn
        
        self.comm.broadcast(U_nn, kpt.root)
        self.comm.broadcast(kpt.eps_n, kpt.root)

        work_nG = self.big_work_arrays['work_nG']
        psit_nG = kpt.psit_nG
        
        # Rotate psit_nG:
        if self.nblocks == 1:
            self.matrix_multiplication(kpt, U_nn)
        else:
            blocked_matrix_multiply(psit_nG, U_nn, work_nG)

        if self.keep_htpsit:
            # Rotate Htpsit_nG:
            Htpsit_nG = self.Htpsit_nG
            work_nG = self.big_work_arrays['work_nG']
            gemm(1.0, Htpsit_nG, U_nn, 0.0, work_nG)
            self.Htpsit_nG = work_nG
            work_nG = Htpsit_nG
            self.big_work_arrays['work_nG'] = work_nG

        if self.band_comm.size == 1:
            for nucleus in hamiltonian.my_nuclei:
                P_ni = nucleus.P_uni[kpt.u]
                gemm(1.0, P_ni.copy(), U_nn, 0.0, P_ni)
        else:
            run([nucleus.calculate_projections(kpt)
                 for nucleus in hamiltonian.pt_nuclei])

        # Rotate EXX related stuff
        if hamiltonian.xc.xcfunc.hybrid > 0.0:
            hamiltonian.xc.xcfunc.exx.rotate(kpt.u, U_nn)

        self.timer.stop('Subspace diag.')

    def calculate_hamiltonian_matrix2(self, hamiltonian, kpt):
        band_comm = self.band_comm
        size = band_comm.size
        assert size % 2 == 1
        np = size // 2 + 1
        rank = band_comm.rank
        psit_nG = kpt.psit_nG
        nmybands = len(psit_nG)

        nI = 0
        for nucleus in hamiltonian.my_nuclei:
            nI += nucleus.get_number_of_partial_waves()

        if self.work_In is None or len(self.work_In) != nI:
            self.work_In = npy.empty((nI, nmybands), psit_nG.dtype)
            self.work2_In = npy.empty((nI, nmybands), psit_nG.dtype)
        work_In = self.work_In
        work2_In = self.work2_In

        work_nG = self.big_work_arrays['work_nG']
        work2_nG = self.big_work_arrays['work2_nG']

        if self.H_pnn is None:
            self.H_pnn = npy.zeros((np, nmybands, nmybands), psit_nG.dtype)
        H_pnn = self.H_pnn

        I1 = 0
        for nucleus in hamiltonian.my_nuclei:
            ni = nucleus.get_number_of_partial_waves()
            I2 = I1 + ni
            P_ni = nucleus.P_uni[kpt.u]
            dH_ii = unpack(nucleus.H_sp[kpt.s])
            work_In[I1:I2] = npy.inner(dH_ii, P_ni).conj()
            I1 = I2

        hamiltonian.apply(psit_nG, work_nG, kpt, local_part_only=True, calculate_projections=False)

        for p in range(np):
            sreq = band_comm.send(work_nG, (rank - 1) % size, 11, False)
            sreq2 = band_comm.send(work_In, (rank - 1) % size, 31, False)
            rreq = band_comm.receive(work2_nG, (rank + 1) % size, 11, False)
            rreq2 = band_comm.receive(work2_In, (rank + 1) % size, 31, False)
            gemm(self.gd.dv, psit_nG, work_nG, 0.0, H_pnn[p], 'c')

            I1 = 0
            for nucleus in hamiltonian.my_nuclei:
                ni = nucleus.get_number_of_partial_waves()
                I2 = I1 + ni
                P_ni = nucleus.P_uni[kpt.u]
                H_pnn[p] += npy.dot(P_ni, work_In[I1:I2]).T
                I1 = I2
            
            band_comm.wait(sreq)
            band_comm.wait(sreq2)
            band_comm.wait(rreq)
            band_comm.wait(rreq2)
            
            work_nG, work2_nG = work2_nG, work_nG
            work_In, work2_In = work2_In, work_In

        self.comm.sum(H_pnn, kpt.root)

        H_nn = self.H_nn
        H_bnbn = H_nn.reshape((size, nmybands, size, nmybands))
        if self.comm.rank == kpt.root:
            if rank == 0:
                H_bnbn[:np, :, 0] = H_pnn
                for p1 in range(1, size):
                    band_comm.receive(H_pnn, p1, 13)
                    for p2 in range(np):
                        if p1 + p2 < size:
                            H_bnbn[p1 + p2, :, p1] = H_pnn[p2]
                        else:
                            H_bnbn[p1, :, p1 + p2 - size] = H_pnn[p2].T
            else:
                band_comm.send(H_pnn, 0, 13)

    def matrix_multiplication(self, kpt, C_nn):
        band_comm = self.band_comm
        size = band_comm.size
        psit_nG =  kpt.psit_nG
        work_nG = self.big_work_arrays['work_nG']
        nmybands = len(psit_nG)
        if size == 1:
            gemm(1.0, psit_nG, C_nn, 0.0, work_nG)
            kpt.psit_nG = work_nG
            if work_nG is self.big_work_arrays.get('work_nG'):
                self.big_work_arrays['work_nG'] = psit_nG

            return

        # Parallelize over bands:
        C_bnbn = C_nn.reshape((size, nmybands, size, nmybands))
        work2_nG = self.big_work_arrays['work2_nG']
        
        rank = band_comm.rank

        beta = 0.0
        for p in range(size - 1):
            sreq = band_comm.send(psit_nG, (rank - 1) % size, 61, False)
            rreq = band_comm.receive(work_nG, (rank + 1) % size, 61, False)
            gemm(1.0, psit_nG, C_bnbn[rank, :, (rank + p) % size],
                 beta, work2_nG)
            beta = 1.0
            band_comm.wait(rreq)
            band_comm.wait(sreq)
            psit_nG, work_nG = work_nG, psit_nG

        gemm(1.0, psit_nG, C_bnbn[rank, :, rank - 1], 1.0, work2_nG)

        kpt.psit_nG = work2_nG
        self.big_work_arrays['work2_nG'] = psit_nG
