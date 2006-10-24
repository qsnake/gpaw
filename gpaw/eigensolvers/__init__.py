"""Module defining  ``Eigensolver`` classes."""

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!

import LinearAlgebra as linalg
from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, rk, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities import unpack

class Eigensolver:
    def __init__(self, exx, timer, convergeall, nn, gd, typecode):
        self.exx = exx
        self.timer = timer
        self.convergeall = convergeall
        self.typecode = typecode
        self.gd = gd
        self.comm = gd.comm

        # Kinetic energy operator:
        self.kin = Laplace(gd, -0.5, nn, typecode)

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(gd, self.kin, typecode)


    def iterate(self, wf, vt_sG, my_nuclei, pt_nuclei, niter=1, tolerance=1e-10):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement ``iterate_once`` method for a single iteration of
        a single kpoint.
        """

        for it in range(niter):
            self.error = 0.0
            for kpt in wf.kpt_u:
                self.iterate_once(kpt, vt_sG, my_nuclei, pt_nuclei)
            nfermi, magmom, S = wf.calculate_occupation_numbers()

            if wf.nvalence == 0:
                self.error = tolerance
            else:
                self.error = wf.kpt_comm.sum(self.error)
                self.error = self.comm.sum(self.error) / wf.nvalence
                
            if (self.error < tolerance):
                break

        return self.error, nfermi, magmom, S


    def diagonalize(self, vt_sG, my_nuclei, kpt, work):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        ``work`` is working array of same size as psit_nG which contains
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

        H_nn = num.zeros((kpt.nbands, kpt.nbands), self.typecode)

        psit_nG = kpt.psit_nG
        eps_n = kpt.eps_n

        self.kin.apply(psit_nG, work, kpt.phase_cd)
        work += kpt.psit_nG * vt_sG[kpt.s]
        if self.exx is not None:
            self.exx.adjust_hamiltonian(psit_nG, work, kpt.nbands,
                                   kpt.f_n, kpt.u, kpt.s)
        r2k(0.5 * self.gd.dv, psit_nG, work, 0.0, H_nn)
        # XXX Do EXX here XXX
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            H_nn += num.dot(P_ni, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                               cc(num.transpose(P_ni))))
            if self.exx is not None:
                self.exx.adjust_hamitonian_matrix(H_nn, P_ni, nucleus, kpt.s)

        self.comm.sum(H_nn, kpt.root)

        if self.comm.rank == kpt.root:
            info = diagonalize(H_nn, eps_n)
            if info != 0:
                raise RuntimeError, 'Very Bad!!'

        self.comm.broadcast(H_nn, kpt.root)
        self.comm.broadcast(eps_n, kpt.root)

        # Rotate psit_nG:
        # We should block this so that we can use a smaller temp !!!!!
        temp = num.array(psit_nG)
        gemm(1.0, temp, H_nn, 0.0, psit_nG)
        
        # Rotate Htpsit_nG:
        temp[:] = work
        gemm(1.0, temp, H_nn, 0.0, work)
        
        # Rotate P_ani:
        for nucleus in my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            temp_ni = P_ni.copy()
            gemm(1.0, temp_ni, H_nn, 0.0, P_ni)

        self.timer.stop('Subspace diag.')
