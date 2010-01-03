"""Module for high-level BLACS interface.

Array index symbol conventions:

* M, N: indices in global array
* m, n: indices in local array

Note that we take into account C vs. F ordering at the blacs grid
and descriptor level here. It will still be necessary to switch
uplo='U' to 'L', trans='N' to 'T', etc.
"""

import numpy as np

from gpaw import sl_diagonalize
from gpaw.mpi import SerialCommunicator
from gpaw.utilities.blas import gemm, r2k
from gpaw.utilities.blacs import scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_ex, pblas_simple_gemm
from gpaw.utilities.timing import nulltimer
import _gpaw


INACTIVE = -1
BLOCK_CYCLIC_2D = 1


class SLDenseLinearAlgebra:
    """ScaLAPACK Dense Linear Algebra."""
    def __init__(self, gd, bd, cols2blocks, blocks2cols, timer=nulltimer):
        self.gd = gd
        self.bd = bd
        assert cols2blocks.dstdescriptor == blocks2cols.srcdescriptor
        self.indescriptor = cols2blocks.srcdescriptor
        self.blockdescriptor = cols2blocks.dstdescriptor
        self.outdescriptor = blocks2cols.dstdescriptor
        self.cols2blocks = cols2blocks
        self.blocks2cols = blocks2cols
        self.timer = timer
    
    def diagonalize(self, H_mm, C_nM, eps_n, S_mm=None):
        if S_mm is None:
            self._standard_diagonalize(H_mm, C_nM, eps_n) #XXX H_mm or H_Nn?
        else:
            self._general_diagonalize(H_mm, S_mm, C_nM, eps_n)

    def _standard_diagonalize(self, H_Nn, C_Nn, eps_n):
        indescriptor = self.indescriptor
        outdescriptor = self.outdescriptor
        blockdescriptor = self.blockdescriptor

        dtype = H_Nn.dtype
        eps_n = np.empty(C_nN.shape[-1])

        # XXX where should inactive ranks be sorted out?
        if not indescriptor:
           shape = indescriptor.shape
           H_Nn = np.zeros(shape, dtype=dtype)
        
        H_nn = blockdescriptor.zeros(dtype=dtype)
        C_nn = blockdescriptor.zeros(dtype=dtype)
        C_nN = outdescriptor.zeros(dtype=dtype)

        self.cols2blocks.redistribute(H_Nn, H_Nn)
        blockdescriptor.general_diagonalize_ex(H_nn, C_nn, eps_M, UL='U',
                                               iu=self.bd.nbands)
        self.blocks2cols.redistribute(C_nn, C_nN) # XXX redist only nM somehow

        if outdescriptor:
            assert self.gd.comm.rank == 0
            bd = self.bd
            bd.distribute(eps_n[:bd.nbands], eps_n)
        else:
            assert self.gd.comm.rank != 0

        self.gd.comm.broadcast(C_nN, 0)
        self.gd.comm.broadcast(eps_n, 0)

    def _general_diagonalize(self, H_mm, S_mm, C_nM, eps_n):
        indescriptor = self.indescriptor
        outdescriptor = self.outdescriptor
        blockdescriptor = self.blockdescriptor

        dtype = S_mm.dtype
        eps_M = np.empty(C_nM.shape[-1])
        C_mm = blockdescriptor.zeros(dtype=dtype)
        self.timer.start('General diagonalize ex')
        blockdescriptor.general_diagonalize_ex(H_mm, S_mm.copy(), C_mm, eps_M,
                                               UL='U', iu=self.bd.nbands)
        self.timer.stop('General diagonalize ex')
        C_mM = outdescriptor.zeros(dtype=dtype)
        self.timer.start('Redistribute coefs')
        self.blocks2cols.redistribute(C_mm, C_mM)
        self.timer.stop('Redistribute coefs')
        self.timer.start('Send coefs to domains')
        if outdescriptor:
            assert self.gd.comm.rank == 0
            bd = self.bd
            C_nM[:] = C_mM[:bd.mynbands, :]
            bd.distribute(eps_M[:bd.nbands], eps_n)
        else:
            assert self.gd.comm.rank != 0

        self.gd.comm.broadcast(C_nM, 0)
        self.gd.comm.broadcast(eps_n, 0)
        self.timer.stop('Send coefs to domains')


class BlacsGrid:
    """Class representing a 2D grid of processors sharing a Blacs context."""
    def __init__(self, comm, nprow, npcol, order='R'):
        assert nprow > 0
        assert npcol > 0
        assert len(order) == 1
        assert order in 'CcRr'

        if isinstance(comm, SerialCommunicator):
            raise ValueError('you forgot mpi AGAIN')
        if comm is None: # if and only if rank is not part of the communicator
            context = INACTIVE
        else:
            if nprow * npcol > comm.size:
                raise ValueError('Impossible: %dx%d Blacs grid with %d CPUs'
                                 % (nprow, npcol, comm.size))
            # This call may not return INACTIVE
            context = _gpaw.new_blacs_context(comm.get_c_object(),
                                              npcol, nprow, order)
        
        self.context = context
        self.comm = comm
        self.nprow = nprow
        self.npcol = npcol
        self.ncpus = nprow * npcol
        self.order = order

    def new_descriptor(self, M, N, mb, nb, rsrc=0, csrc=0):
        return BlacsDescriptor(self, M, N, mb, nb, rsrc, csrc)

    def is_active(self):
        """Whether context is active on this rank."""
        return self.context != INACTIVE

    def __nonzero__(self):
        return self.is_active()

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[comm:size=%d,rank=%d; context=%d; %dx%d]'
        string = template % (classname, self.comm.size, self.comm.rank, 
                             self.context, self.nprow, self.npcol)
        return string
    
    def __del__(self):
        if self.is_active():
            _gpaw.blacs_destroy(self.context)


class MatrixDescriptor:
    """Class representing a 2D matrix shape.  Base class for parallel
    matrix descriptor with BLACS.  This class is by itself serial."""
    
    def __init__(self, M, N):
        self.shape = (M, N)
    
    def __nonzero__(self):
        return self.shape[0] != 0 and self.shape[1] != 0

    def zeros(self, n=(), dtype=float):
        return self._new_array(np.zeros, n, dtype)

    def empty(self, n=(), dtype=float):
        return self._new_array(np.empty, n, dtype)

    def _new_array(self, func, n, dtype):
        if isinstance(n, int):
            n = n,
        shape = n + self.shape
        return func(shape, dtype)

    def check(self, a_mn):
        return a_mn.shape == self.shape and a_mn.flags.contiguous

    def checkassert(self, a_mn):
        ok = self.check(a_mn)
        if not ok:
            if not a_mn.flags.contiguous:
                msg = 'Matrix with shape %s is not contiguous' % (a_mn.shape,)
            else:
                msg = ('%s-descriptor incompatible with %s-matrix' %
                       (self.shape, a_mn.shape))
            raise AssertionError(msg)


class BlacsDescriptor(MatrixDescriptor):
    """Class representing a 2D matrix distributed on a blacs grid.

    The global shape is M by N, being distributed on the specified BlacsGrid
    such that mb and nb are rows and columns on each processor.
    
    The following chart describes how different ranks (there are 4
    ranks in this example, 0 through 3) divide the matrix into blocks.
    This is called 2D block cyclic distribution::

        +--+--+--+--+..+--+
        | 0| 1| 0| 1|..| 1|
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+
        | 0| 1| 0| 1|..| 1|
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+
        ...................
        ...................
        +--+--+--+--+..+--+
        | 2| 3| 2| 3|..| 3|
        +--+--+--+--+..+--+

    Also refer to:
    http://acts.nersc.gov/scalapack/hands-on/datadist.html
        
    """
    def __init__(self, blacsgrid, M, N, mb, nb, rsrc, csrc):
        assert M > 0
        assert N > 0
        assert 1 <= mb <= M
        assert 1 <= nb <= N
        assert 0 <= rsrc < blacsgrid.nprow
        assert 0 <= csrc < blacsgrid.npcol
        
        self.blacsgrid = blacsgrid
        self.M = M # global size 1
        self.N = N # global size 2
        self.mb = mb # block cyclic distr dim 1
        self.nb = nb # and 2.  How many rows or columns are on this processor
        # more info:
        # http://www.netlib.org/scalapack/slug/node75.html
        self.rsrc = rsrc
        self.csrc = csrc
        
        if 1:#blacsgrid.is_active():
            locN, locM = _gpaw.get_blacs_shape(self.blacsgrid.context,
                                               self.N, self.M,
                                               self.nb, self.mb, 
                                               self.csrc, self.rsrc)
            self.lld  = max(1, locN) # max 1 is nonsensical, but appears
                                     # to be required by PBLAS
        else:
            locN, locM = 0, 0
            self.lld = 0
        
        MatrixDescriptor.__init__(self, max(0, locM), max(0, locN))
        
        self.active = locM > 0 and locN > 0 # inactive descriptor can
                                            # exist on an active OR
                                            # inactive blacs grid
        
        self.bshape = (self.mb, self.nb) # Shape of one block
        self.gshape = (M, N) # Global shape of array


    def asarray(self):
        arr = np.array([BLOCK_CYCLIC_2D, self.blacsgrid.context, 
                        self.N, self.M, self.nb, self.mb, self.csrc, self.rsrc,
                        self.lld], np.int32)
        return arr

    def __repr__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %s, block %s, lld %d, loc %s]'
        string = template % (classname, self.blacsgrid.context,
                             self.gshape,
                             self.bshape, self.lld, self.shape)
        return string

    def diagonalize_ex(self, H_nn, C_nn, eps_n, UL='U', iu=None):
        scalapack_diagonalize_ex(self, H_nn, C_nn, eps_n, UL, iu=iu)

    def general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M, UL='U', iu=None):
        scalapack_general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M,
                                         UL, iu=iu)


class Redistributor:
    """Class for redistributing BLACS matrices on different contexts."""
    def __init__(self, supercomm, srcdescriptor, dstdescriptor):
        self.supercomm = supercomm
        self.srcdescriptor = srcdescriptor
        self.dstdescriptor = dstdescriptor
    
    def redistribute_submatrix(self, src_mn, dst_mn, subM, subN):
        # self.supercomm must be a supercommunicator of the communicators
        # corresponding to the context of srcmatrix as well as dstmatrix.
        # We should verify this somehow.
        dtype = src_mn.dtype
        assert dtype == dst_mn.dtype
        
        isreal = (dtype == float)
        assert dtype == float or dtype == complex

        self.srcdescriptor.checkassert(src_mn)
        self.dstdescriptor.checkassert(dst_mn)
        
        _gpaw.scalapack_redist(self.srcdescriptor.asarray(), 
                               self.dstdescriptor.asarray(),
                               src_mn.T, dst_mn.T,
                               self.supercomm.get_c_object(),
                               subN, subM, isreal)
    
    def redistribute(self, src_mn, dst_mn):
        subM, subN = self.srcdescriptor.gshape
        self.redistribute_submatrix(src_mn, dst_mn, subM, subN)


def parallelprint(comm, obj):
    import sys
    for a in range(comm.size):
        if a == comm.rank:
            print 'rank=%d' % a
            print obj
            print
            sys.stdout.flush()
        comm.barrier()


class BlacsBandDescriptor:
    # this class 'describes' all the Realspace/Blacs-related stuff
    def __init__(self, world, gd, bd, kpt_comm):
        ncpus, mcpus, blocksize = sl_diagonalize[:3]
        
        bcommsize = bd.comm.size
        gcommsize = gd.comm.size
        bcommrank = bd.comm.rank
        
        shiftks = kpt_comm.rank * bcommsize * gcommsize
        column_ranks = shiftks + np.arange(bcommsize) * gcommsize
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        columncomm = world.new_communicator(column_ranks)
        blockcomm = world.new_communicator(block_ranks)

        self.bd = bd
        nbands = self.bd.nbands
        mynbands = self.bd.mynbands

        # Create 1D and 2D BLACS grid
        columngrid = BlacsGrid(bd.comm, 1, bcommsize)
        blockgrid  = BlacsGrid(blockcomm, ncpus, mcpus)

        # 1D layout
        Nndescriptor = columngrid.new_descriptor(nbands, nbands, nbands,
                                                 mynbands)
        
        # 2D layout
        nndescriptor = blockgrid.new_descriptor(nbands, nbands, blocksize,
                                                blocksize)

        self.Nndescriptor = Nndescriptor
        self.nndescriptor = nndescriptor
        self.Nn2nn = Redistributor(blockcomm, Nndescriptor, nndescriptor)
        self.nn2Nn = Redistributor(blockcomm, nndescriptor, Nndescriptor)


class BlacsOrbitalDescriptor: # XXX can we find a less confusing name?
    # This class 'describes' all the LCAO/Blacs-related stuff
    def __init__(self, world, gd, bd, kpt_comm, nao, timer=nulltimer):
        ncpus, mcpus, blocksize = sl_diagonalize[:3]

        bcommsize = bd.comm.size
        gcommsize = gd.comm.size
        bcommrank = bd.comm.rank

        # XXX these things can probably be obtained in a more programmatically
        # convenient way
        shiftks = kpt_comm.rank * bcommsize * gcommsize
        column_ranks = shiftks + np.arange(bcommsize) * gcommsize
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        columncomm = world.new_communicator(column_ranks)
        blockcomm = world.new_communicator(block_ranks)

        nbands = bd.nbands
        mynbands = bd.mynbands
        mynao = -((-nao) // bcommsize)
        self.nao = nao
        self.mynao = mynao

        # Range of basis functions for BLACS distribution of matrices:
        self.Mmax = nao
        self.Mstart = bcommrank * mynao
        self.Mstop = min(self.Mstart + mynao, self.Mmax)

        # Column layout for one matrix per band rank:
        columngrid = BlacsGrid(bd.comm, bcommsize, 1)
        self.mMdescriptor = columngrid.new_descriptor(nao, nao, mynao, nao)
        self.nMdescriptor = columngrid.new_descriptor(nbands, nao, mynbands,
                                                      nao)

        # Column layout for one matrix in total (only on grid masters):
        single_column_grid = BlacsGrid(columncomm, bcommsize, 1)
        mM_unique_descriptor = single_column_grid.new_descriptor(nao, nao,
                                                                 mynao, nao)
        # nM_unique_descriptor is meant to hold the coefficients after
        # diagonalization.  BLACS requires it to be nao-by-nao, but
        # we only fill meaningful data into the first nbands columns.
        #
        # The array will then be trimmed and broadcast across
        # the grid descriptor's communicator.
        nM_unique_descriptor = single_column_grid.new_descriptor(nao, nao,
                                                                 mynbands, nao)

        # Fully blocked grid for diagonalization with many CPUs:
        blockgrid = BlacsGrid(blockcomm, mcpus, ncpus)
        mmdescriptor = blockgrid.new_descriptor(nao, nao, blocksize, blocksize)

        self.mM_unique_descriptor = mM_unique_descriptor
        self.mmdescriptor = mmdescriptor
        #self.nMdescriptor = nMdescriptor
        self.mM2mm = Redistributor(blockcomm, mM_unique_descriptor,
                                   mmdescriptor)
        self.mm2nM = Redistributor(blockcomm, mmdescriptor,
                                   nM_unique_descriptor)

        self.world = world
        self.gd = gd
        self.bd = bd
        self.timer = timer

    def get_diagonalizer(self):
        return SLDenseLinearAlgebra(self.gd, self.bd, self.mM2mm, self.mm2nM,
                                    self.timer)

    def get_overlap_descriptor(self):
        return self.mMdescriptor

    def get_diagonalization_descriptor(self):
        return self.mmdescriptor

    def get_coefficient_descriptor(self):
        return self.nMdescriptor

    def distribute_overlap_matrix(self, S1_qmM):
        xshape = S1_qmM.shape[:-2]
        nm, nM = S1_qmM.shape[-2:]
        S1_qmM = S1_qmM.reshape(-1, nm, nM)
        
        blockdesc = self.mmdescriptor
        coldesc = self.mM_unique_descriptor
        S_qmm = blockdesc.zeros(len(S1_qmM), S1_qmM.dtype)
        
        # XXX ugly hack
        # TODO distribute T_qMM in the same way.  Hack eigensolver
        # as appropriate
        self.timer.start('Distribute overlap matrix')
        S_qmM = coldesc.zeros(len(S1_qmM), S1_qmM.dtype)
        for S_mM, S_mm, S1_mM in zip(S_qmM, S_qmm, S1_qmM):
            if self.gd.comm.rank == 0:
                S_mM[:] = S1_mM
            self.mM2mm.redistribute(S_mM, S_mm)
        self.timer.stop('Distribute overlap matrix')
        return S_qmm.reshape(xshape + blockdesc.shape)

    def calculate_density_matrix(self, f_n, C_nM, rho_mM=None):
        nbands = self.bd.nbands
        mynbands = self.bd.mynbands
        nao = self.nao
        
        if rho_mM is None:
            rho_mM = self.mMdescriptor.zeros()
        
        Cf_nM = C_nM * f_n[:, None]
        pblas_simple_gemm(self.nMdescriptor, self.nMdescriptor,
                          self.mMdescriptor, Cf_nM, C_nM, rho_mM, transa='T')
        return rho_mM


class OrbitalDescriptor:
    def __init__(self, gd, bd, nao):
        self.gd = gd # XXX shouldn't be necessary
        self.bd = bd
        self.mMdescriptor = MatrixDescriptor(nao, nao)
        self.nMdescriptor = MatrixDescriptor(bd.mynbands, nao)
        
        self.Mstart = 0
        self.Mstop = nao
        self.Mmax = nao
        self.mynao = nao
        self.nao = nao

    def get_diagonalizer(self):
        if sl_diagonalize:
            from gpaw.lcao.eigensolver import SLDiagonalizer
            diagonalizer = SLDiagonalizer(self.gd, self.bd)
        else:
            from gpaw.lcao.eigensolver import LapackDiagonalizer
            diagonalizer = LapackDiagonalizer(self.gd, self.bd)
        return diagonalizer

    def get_overlap_descriptor(self):
        return self.mMdescriptor

    def get_diagonalization_descriptor(self):
        return self.mMdescriptor

    def get_coefficent_descriptor(self):
        return self.nMdescriptor

    def distribute_overlap_matrix(self, S_qMM):
        return S_qMM

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # XXX Should not conjugate, but call gemm(..., 'c')
        # Although that requires knowing C_Mn and not C_nM.
        # that also conforms better to the usual conventions in literature
        Cf_Mn = C_nM.T.conj() * f_n
        gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
        self.bd.comm.sum(rho_MM)
        return rho_MM

    def alternative_calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # Alternative suggestion. Might be faster. Someone should test this
        C_Mn = C_nM.T.copy()
        r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
        tri2full(rho_MM)
        return rho_MM
