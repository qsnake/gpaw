"""Module defining  ``Eigensolver`` classes."""

import numpy as np

from gpaw.utilities.blas import axpy, gemm
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.utilities import unpack, pack
from gpaw.utilities.tools import tri2full
from gpaw.mpi import run


class NUIMin(Eigensolver):
    """Solver for minimization of non-unitary invariant (NUI) functionals

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``kpt.P_ani`` are already calculated

    Solution steps are:

    * Calculation of residuals 
    * Improvement of wave functions:  psi' = psi + PR       
    * Orthonormalization

    *** THIS CLASS IS UNDER DEVELOPMENT ***
    """

    def __init__(self, keep_htpsit=True):
        Eigensolver.__init__(self, keep_htpsit)

    def initialize(self, wfs):
        Eigensolver.initialize(self, wfs)
        self.overlap = wfs.overlap

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        """Do a single NUI minimization iteration for the kpoint"""

        R_nG = self.Htpsit_nG
        setups = hamiltonian.setups

        # 
        # define a function which applies a orbital-density dendent
        # hamiltonian on pseudo-wavefunctions
        def H(psit_nG):
            #
            # the unitary invariant part
            wfs.kin.apply(psit_nG, R_nG, kpt.phase_cd)
            hamiltonian.apply_local_potential(psit_nG, R_nG, kpt.s)
            #
            # the unitary variant part (orbital dependency)
            hamiltonian.xc.add_non_local_terms(psit_nG, R_nG, kpt.s)
            return R_nG
        #
        # shortcuts to the wavefunctions and projections
        psit_nG = kpt.psit_nG
        P_ani   = kpt.P_ani
        #
        # prepare PAW corrections to the matrix-elements
        dH_aii = dict([(a, unpack(dH_sp[kpt.s]))
                       for a, dH_sp in hamiltonian.dH_asp.items()])
        #
        # prepare to setup the hamiltonian matrixelements
        H_nn = np.zeros((wfs.nbands, wfs.nbands))
        #
        # contributions from pseudo-WF overlaps (including
        # orbital dependent components)
        gemm(wfs.gd.dv, psit_nG, H(psit_nG), 0.0, H_nn, 'c')
        #
        # PAW corrections to the unitary invariant part of the
        # hamiltonian
        for a, P_ni in P_ani.items():
            H_nn += np.dot(P_ni, np.dot(dH_aii[a], P_ni.T))
        #
        #
        # PAW corrections to the self-coulomb
        c_ani = {}
        for a, P_ni in P_ani.items():

            # contributions from unintary invariant
            # part of H and overlap matrix
            dH_ii = dH_aii[a]
            dS_ii = hamiltonian.setups[a].O_ii
            c_ni = (  np.dot(P_ni, dH_ii)
                    - np.dot(np.dot(H_nn, P_ni), dS_ii))
            c_ani[a] = c_ni

            # orbital dependent contributions
            # to the PAW corrections to the residual
            for n in range(wfs.nbands):
                P_i = P_ni[n]
                D_ii = np.outer(P_i, P_i)
                D_p = pack(D_ii)
                dU_ii = unpack(np.dot(setups[a].M_pp, D_p))
                H_nn[:,n] -= 2*np.dot(P_ni,np.dot(dU_ii,P_i))
                c_ni       = np.dot(P_ni,dU_ii)
                c_ani[a][n] -= 2*c_ni[n]
            
        #H_nn[:,n] += np.dot(P_ni,np.dot(dU_aii[n,:],P_i.T))
        #H_nn[:,n] += np.dot(P_ni,np.dot(dU_aii,P_i.T))
        
        #H_nn = (H_nn + H_nn.T)/2
        #H_nn = self.operator.calculate_matrix_elements(psit_nG, P_ani,
        #                                               H, dH_aii)
        D_nn=(H_nn-H_nn.T)/2
        D_nn=D_nn**2
        print 'lod=',(D_nn.sum())**(0.5)
        H_nn=(H_nn+H_nn.T)/2

        #
        # diagonal elements (s.p. expectation values) of hamiltonian
        # matrix are used as "eigenvalues"
        # (rather than explicit diagonalization which would affect the
        #  orbitals)
        kpt.eps_n[:] = H_nn.diagonal()
        #
        # compute the residuum
        # ------------------------------------------------------------
        # residuum of the pseudo wavefunctions
        gemm(-1.0, psit_nG, H_nn, 1.0, R_nG)
        #
        # PAW corrections to the residuum
        c_ani = {}
        for a, P_ni in P_ani.items():
            dH_ii = dH_aii[a]
            dS_ii = hamiltonian.setups[a].O_ii
            c_ni = (  np.dot(P_ni, dH_ii)
                    - np.dot(np.dot(H_nn, P_ni), dS_ii))
            c_ani[a] = c_ni

        for n in range(wfs.nbands):
            D_aii={}
            for a, P_ni in P_ani.items():
                P_i       = P_ni[n]
                D_aii[a]  = np.outer(P_i, P_i)
                D_p       = pack(D_aii[a])
                dU_aii    = unpack(np.dot(setups[a].M_pp, D_p))
                d_ni      = np.dot(P_ni,dU_aii)
                c_ani[a][n] -= 2*d_ni[n]
                
        wfs.pt.add(R_nG, c_ani, kpt.q)
        #
        # accumulate total errors, precondition and correct
        # ------------------------------------------------------------
        error = 0.0
        for n in range(self.mynbands):
            R_G = R_nG[n]
            #
            # accumulate errors
            error += np.vdot(R_G, R_G).real
            #
            # precondition the residual
            pR_G = self.preconditioner(R_G, kpt.phase_cd)[0]
            #
            # correct
            psit_nG[n] += pR_G
        #
        # gather error from all nodes
        error = self.gd.comm.sum(error)
        
        return error
    
