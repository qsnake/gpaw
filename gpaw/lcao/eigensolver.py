import numpy as np
from gpaw.utilities import unpack
from gpaw.utilities.lapack import general_diagonalize
from gpaw.utilities.blas import gemm
from gpaw.utilities import scalapack
from gpaw import sl_diagonalize, extra_parameters
import gpaw.mpi as mpi


class BaseDiagonalizer:
    def __init__(self, gd, bd):
        self.gd = gd
        self.bd = bd

    def diagonalize(self, H_MM, C_nM, eps_n, S_MM):
        eps_M = np.empty(C_nM.shape[-1])
        info = self._diagonalize(H_MM, S_MM.copy(), eps_M)
        if info != 0:
            raise RuntimeError('Failed to diagonalize: %d' % info)
        
        nbands = self.bd.nbands
        if self.bd.rank == 0:
            self.gd.comm.broadcast(H_MM[:nbands], 0)
            self.gd.comm.broadcast(eps_M[:nbands], 0)
        self.bd.distribute(H_MM[:nbands], C_nM)
        self.bd.distribute(eps_M[:nbands], eps_n)
    
    def _diagonalize(self, H_MM, S_MM, eps_M):
        raise NotImplementedError

    def estimate_memory(self, mem, dtype):
        nao = self.setups.nao
        itemsize = mem.itemsize[dtype]
        mem.subnode('eps [M]', self.nao * mem.floatsize)
        mem.subnode('H [MM]', self.nao * self.nao * itemsize)
    

class LapackDiagonalizer(BaseDiagonalizer):
    """Serial diagonalizer."""
    def _diagonalize(self, H_MM, S_MM, eps_M):
        # Only one processor really does any work.
        if self.gd.comm.rank == 0 and self.bd.comm.rank == 0:
            return general_diagonalize(H_MM, eps_M, S_MM)
        else:
            return 0


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self, diagonalizer=None):
        self.diagonalizer = diagonalizer
        # ??? why should we be able to set
        # this diagonalizer in both constructor and initialize?
        self.has_initialized = False # XXX

    def initialize(self, gd, dtype, nao, diagonalizer=None):
        self.gd = gd
        self.nao = nao
        if diagonalizer is not None:
            self.diagonalizer = diagonalizer
        assert self.diagonalizer is not None
        self.has_initialized = True # XXX

    def error(self):
        return 0.0
    error = property(error)

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt, root=-1):
        # XXX document parallel stuff, particularly root parameter
        assert self.has_initialized
        vt_G = hamiltonian.vt_sG[kpt.s]
        H_MM = np.empty((wfs.ksl.mynao, wfs.ksl.nao), wfs.dtype)

        wfs.timer.start('Potential matrix')
        wfs.basis_functions.calculate_potential_matrix(vt_G, H_MM, kpt.q)
        wfs.timer.stop('Potential matrix')

        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        wfs.timer.start('Atomic Hamiltonian')
        Mstart = wfs.basis_functions.Mstart
        Mstop = wfs.basis_functions.Mstop
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(hamiltonian.dH_asp[a][kpt.s]), wfs.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), wfs.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            gemm(1.0, dHP_iM, P_Mi[Mstart:Mstop], 1.0, H_MM)
        wfs.timer.stop('Atomic Hamiltonian')
        wfs.timer.start('Distribute overlap matrix')
        H_MM = wfs.ksl.distribute_overlap_matrix(H_MM, root)
        wfs.timer.stop('Distribute overlap matrix')
        H_MM += wfs.T_qMM[kpt.q]
        return H_MM

    def iterate(self, hamiltonian, wfs):
        wfs.timer.start('LCAO eigensolver')
        for kpt in wfs.kpt_u:
            self.iterate_one_k_point(hamiltonian, wfs, kpt)
        wfs.timer.stop('LCAO eigensolver')

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_MM = self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt, root=0)
        S_MM = wfs.S_qMM[kpt.q]

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(wfs.bd.mynbands)
            
        if sl_diagonalize:
            assert scalapack()

        kpt.eps_n[0] = 42
        
        diagonalizationstring = self.diagonalizer.__class__.__name__
        wfs.timer.start(diagonalizationstring)
        self.diagonalizer.diagonalize(H_MM, kpt.C_nM, kpt.eps_n, S_MM)
        wfs.timer.stop(diagonalizationstring)

        if (kpt.eps_n[0] == 42):
            raise RuntimeError('LCAO diagonalization failed! You may want to check your structure.')    

        wfs.timer.start('Calculate projections')
        # P_ani are not strictly necessary as required quantities can be
        # evaluated directly using P_aMi.  We should probably get rid
        # of the places in the LCAO code using P_ani directly
        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')
        wfs.timer.stop('Calculate projections')

    def estimate_memory(self, mem, dtype):
        pass 
        # self.diagonalizer.estimate_memory(mem, dtype)
