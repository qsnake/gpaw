"""Module defining  ``Eigensolver`` classes."""

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!

import LinearAlgebra as linalg
from gpaw.operators import Laplace
from gpaw.preconditioner import Preconditioner
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import axpy, r2k, gemm
from gpaw.utilities.complex import cc, real
from gpaw.utilities import unpack

class Eigensolver:
    def __init__(self, timer, kpt_comm, gd, kin, typecode):
        self.timer = timer
        self.kpt_comm = kpt_comm
        self.typecode = typecode
        self.gd = gd
        self.comm = gd.comm

        # Preconditioner for the electronic gradients:
        self.preconditioner = Preconditioner(gd, kin, typecode)

    def set_convergence_criteria(self, convergeall, tolerance, nvalence):
        self.convergeall = convergeall
        self.tolerance = tolerance
        self.nvalence = nvalence
        
    def iterate(self, hamiltonian, kpt_u):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement ``iterate_one_k_point`` method for a single iteration of
        a single kpoint.
        """

        if self.nvalence == 0:
            return self.tolerance, True
        
        error = 0.0
        for kpt in kpt_u:
            error += self.iterate_one_k_point(hamiltonian, kpt)
            
        if self.convergeall:
            error = self.comm.sum(self.kpt_comm.sum(error)) / kpt_u[0].nbands
        else:
            error = self.comm.sum(self.kpt_comm.sum(error)) / self.nvalence
        
        return error, error <= self.tolerance

    def iterate_one_k_point(self, hamiltonian, kpt):
        """Implemented in subclasses."""
        return 0.0
    
    def diagonalize(self, hamiltonian, kpt, Htpsit_nG):
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

        H_nn = num.empty((kpt.nbands, kpt.nbands), self.typecode)

        psit_nG = kpt.psit_nG
        eps_n = kpt.eps_n

        hamiltonian.kin.apply(psit_nG, Htpsit_nG, kpt.phase_cd)
            
        Htpsit_nG += psit_nG * hamiltonian.vt_sG[kpt.s]

        H_nn[:] = 0.0  # r2k fails without this!
        hamiltonian.xc.xcfunc.calculate_non_local_energy(kpt, Htpsit_nG, H_nn)

        r2k(0.5 * self.gd.dv, psit_nG, Htpsit_nG, 1.0, H_nn)

        # XXX Do EXX here XXX
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            H_nn += num.dot(P_ni, num.dot(unpack(nucleus.H_sp[kpt.s]),
                                          cc(num.transpose(P_ni))))
            if 0:#self.exx is not None:
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
        temp[:] = Htpsit_nG
        gemm(1.0, temp, H_nn, 0.0, Htpsit_nG)
        
        # Rotate P_ani:
        for nucleus in hamiltonian.my_nuclei:
            P_ni = nucleus.P_uni[kpt.u]
            temp_ni = P_ni.copy()
            gemm(1.0, temp_ni, H_nn, 0.0, P_ni)

        self.timer.stop()
