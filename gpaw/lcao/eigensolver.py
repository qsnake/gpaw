import numpy as np
from gpaw.utilities import unpack
from gpaw.utilities.lapack import diagonalize
from gpaw.utilities.blas import gemm
from gpaw.utilities import scalapack
from gpaw import sl_diagonalize, extra_parameters
import gpaw.mpi as mpi


class SLDiagonalizer:
    """ScaLAPACK diagonalizer using redundantly distributed arrays."""
    def __init__(self, root=0):
        self.root = root
        # Keep buffers?

    def diagonalize(self, H_MM, S_MM, eps_n, kpt):
        return diagonalize(H_MM, eps_n, b=S_MM, root=self.root)


class LapackDiagonalizer:
    """Serial diagonalizer."""
    def __init__(self):
        pass # keep buffers?

    def diagonalize(self, H_MM, S_MM, eps_n, kpt):
        return diagonalize(H_MM, eps_n, S_MM)


class LCAO:
    """Eigensolver for LCAO-basis calculation"""

    def __init__(self, diagonalizer=None):
        self.error = 0.0
        self.linear_kpts = None
        self.eps_n = None
        self.S_MM = None
        self.H_MM = None
        self.timer = None
        self.mynbands = None
        self.band_comm = None
        self.world = None
        self.diagonalizer = None
        self.has_initialized = False # XXX

    def initialize(self, kpt_comm, gd, band_comm, dtype, nao, mynbands, world,
                   diagonalizer=None):
        self.kpt_comm = kpt_comm
        self.gd = gd
        self.band_comm = band_comm
        self.dtype = dtype
        self.nao = nao
        self.mynbands = mynbands
        self.world = world
        if diagonalizer is None:
            diagonalizer = LapackDiagonalizer()
        self.diagonalizer = diagonalizer
        self.has_initialized = True # XXX
        assert self.H_MM is None # Right now we're not sure whether
        # this will work when reusing

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt):
        assert self.has_initialized
        s = kpt.s
        q = kpt.q
        if self.H_MM is None:
            nao = self.nao
            mynao = wfs.S_qMM.shape[1]
            self.eps_n = np.empty(nao)
            self.S_MM = np.empty((mynao, nao), self.dtype)
            self.H_MM = np.empty((mynao, nao), self.dtype)
            self.timer = wfs.timer
            #self.linear_dependence_check(wfs)

        self.timer.start('LCAO: potential matrix')

        wfs.basis_functions.calculate_potential_matrix(hamiltonian.vt_sG[s],
                                                       self.H_MM, q)

        self.timer.stop('LCAO: potential matrix')

        # Add atomic contribution
        #
        #           --   a     a  a*
        # H      += >   P    dH  P
        #  mu nu    --   mu i  ij nu j
        #           aij
        #
        Mstart = wfs.basis_functions.Mstart
        Mstop = wfs.basis_functions.Mstop
        for a, P_Mi in kpt.P_aMi.items():
            dH_ii = np.asarray(unpack(hamiltonian.dH_asp[a][s]), P_Mi.dtype)
            dHP_iM = np.zeros((dH_ii.shape[1], P_Mi.shape[0]), P_Mi.dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dH_ii, 0.0, dHP_iM, 'c')
            if Mstart != -1:
                P_Mi = P_Mi[Mstart:Mstop]
            gemm(1.0, dHP_iM, P_Mi, 1.0, self.H_MM)
        self.gd.comm.sum(self.H_MM)
        self.H_MM += wfs.T_qMM[q]

    def iterate(self, hamiltonian, wfs):
        for kpt in wfs.kpt_u:
            self.iterate_one_k_point(hamiltonian, wfs, kpt)

    def iterate_one_k_point(self, hamiltonian, wfs, kpt):

        if self.band_comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        self.calculate_hamiltonian_matrix(hamiltonian, wfs, kpt)
        self.S_MM[:] = wfs.S_qMM[kpt.q]

        bandrank = self.band_comm.rank
        bandsize = self.band_comm.size
        mynbands = self.mynbands
        n1 = bandrank * mynbands
        n2 = n1 + mynbands

        if kpt.eps_n is None:
            kpt.eps_n = np.empty(mynbands)
            
        if sl_diagonalize:
            assert mpi.parallel
            assert scalapack()

        self.eps_n[0] = 42
        
        diagonalizationstring = self.diagonalizer.__class__.__name__
        self.timer.start(diagonalizationstring)

        info = self.diagonalizer.diagonalize(self.H_MM, self.S_MM, self.eps_n,
                                             kpt)
        if info != 0:
            raise RuntimeError('Failed to diagonalize: info=%d' % info)
        self.timer.stop(diagonalizationstring)

        if not extra_parameters.get('blacs'):
            if bandrank == 0:
                self.gd.comm.broadcast(self.H_MM[:wfs.nbands], 0)
                self.gd.comm.broadcast(self.eps_n[:wfs.nbands], 0)
            wfs.bd.distribute(self.H_MM[:wfs.nbands], kpt.C_nM)
            wfs.bd.distribute(self.eps_n[:wfs.nbands], kpt.eps_n)

        assert kpt.eps_n[0] != 42

        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            gemm(1.0, kpt.P_aMi[a], kpt.C_nM, 0.0, P_ni, 'n')

    def estimate_memory(self, mem):
        # XXX forward to diagonalizer
        itemsize = np.array(1, self.dtype).itemsize
        mem.subnode('H, work [2*MM]', self.nao * self.nao * itemsize)

