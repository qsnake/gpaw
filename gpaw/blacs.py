"""Module for high-level BLACS interface.

Array index symbol conventions::
 * M, N: indices in global array
 * m, n: indices in local array

"""

import numpy as np

from gpaw.mpi import SerialCommunicator
from gpaw.utilities.blacs import scalapack_diagonalize_ex
import _gpaw


INACTIVE = -1
BLOCK_CYCLIC_2D = 1


class SLEXDiagonalizer:
    """ScaLAPACK Expert Driver diagonalizer."""
    def __init__(self, gd, bd, cols2blocks, blocks2cols):
        self.gd = gd
        self.bd = bd
        assert cols2blocks.dstdescriptor == blocks2cols.srcdescriptor
        self.indescriptor = cols2blocks.srcdescriptor
        self.blockdescriptor = cols2blocks.dstdescriptor
        self.outdescriptor = blocks2cols.dstdescriptor
        self.cols2blocks = cols2blocks
        self.blocks2cols = blocks2cols
    
    def diagonalize(self, H_mM, S_mM, eps_M, kpt):
        indescriptor = self.indescriptor
        outdescriptor = self.outdescriptor
        blockdescriptor = self.blockdescriptor

        dtype = H_mM.dtype

        # XXX where should inactive ranks be sorted out?
        if not indescriptor:
            H_mM = np.zeros((0, 0))
            S_mM = np.zeros((0, 0))
        
        S_mm = blockdescriptor.zeros(dtype)
        H_mm = blockdescriptor.zeros(dtype)
        C_mm = blockdescriptor.zeros(dtype)
        C_nM = outdescriptor.zeros(dtype)

        self.cols2blocks.redistribute(S_mM, S_mm)
        self.cols2blocks.redistribute(H_mM, H_mm)
        blockdescriptor.diagonalize_ex(H_mm, S_mm, C_mm, eps_M, 'U')
        self.blocks2cols.redistribute(C_mm, C_nM)

        if outdescriptor:
            assert self.gd.comm.rank == 0
            bd = self.bd
            kpt.C_nM[:] = C_nM[:bd.mynbands, :]
            bd.distribute(eps_M[:bd.nbands], kpt.eps_n)
        else:
            assert self.gd.comm.rank != 0

        self.gd.comm.broadcast(kpt.C_nM, 0)
        self.gd.comm.broadcast(kpt.eps_n, 0)
        return 0


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
            # (which is not entirely sensible.  That rank must then belong in
            # a different communicator.  Which one?)
            context = INACTIVE
        else:
            if nprow * npcol > comm.size:
                raise ValueError('Impossible: %dx%d Blacs grid with %d CPUs'
                                 % (nprow, npcol, comm.size))
            # This call may also return INACTIVE
            context = _gpaw.new_blacs_context(comm, npcol, nprow, order)
        
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
        _gpaw.blacs_destroy(self.context)


class NonBlacsGrid:
    def __init__(self):
        pass

    def new_descriptor(self, M, N, mb, nb, rsrc=0, csrc=0):
        desc = MatrixDescriptor(M, N)
        return desc

class MatrixDescriptor:
    """Class representing a 2D matrix shape.  Base class for parallel
    matrix descriptor with BLACS.  This class is by itself serial."""
    
    def __init__(self, M, N):
        self.shape = (M, N)
    
    def __nonzero__(self):
        return self.shape[0] != 0 and self.shape[1] != 0

    def zeros(self, dtype=float):
        return np.zeros(self.shape, dtype)

    def empty(self, dtype=float):
        return np.empty(self.shape, dtype)

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
    
    XXX rsrc, csrc?

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
        
    """
    def __init__(self, blacsgrid, M, N, mb, nb, rsrc, csrc):
        assert M > 0
        assert N > 0
        assert 0 < mb <= M
        assert 0 < nb <= N
        # asserts on rsrc, csrc?
        
        self.blacsgrid = blacsgrid
        self.M = M # global size 1
        self.N = N # global size 2
        self.mb = mb # block cyclic distr dim 1
        self.nb = nb # and 2.  How many rows or columns are on this processor
        # more info:
        # http://www.netlib.org/scalapack/slug/node75.html
        self.rsrc = rsrc
        self.csrc = csrc
        
        if blacsgrid.is_active():
            locN, locM = _gpaw.get_blacs_shape(self.blacsgrid.context,
                                               self.N, self.M,
                                               self.nb, self.mb, 
                                               self.rsrc, self.csrc)
        else:
            locN, locM = 0, 0
        
        MatrixDescriptor.__init__(self, locM, locN)
        
        self.active = locM > 0 and locN > 0
        
        self.bshape = (self.mb, self.nb) # Shape of one block
        self.gshape = (M, N) # Global shape of array
        self.lld  = locN # lld = 'local leading dimension'

    def asarray(self):
        arr = np.array([BLOCK_CYCLIC_2D, self.blacsgrid.context, 
                        self.N, self.M, self.nb, self.mb, self.rsrc, self.csrc,
                        max(0, self.lld)], np.int32)
        return arr

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %s, block %s, lld %d, loc %s]'
        string = template % (classname, self.blacsgrid.context,
                             self.gshape,
                             self.bshape, self.lld, self.shape)
        return string

    def diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M, UL='U'):
        self.checkassert(H_mm)
        self.checkassert(S_mm)
        self.checkassert(C_mm)
        scalapack_diagonalize_ex(self, H_mm.T, S_mm.T, C_mm.T, eps_M, UL)


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
                               self.supercomm, subN, subM, isreal)
    
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


class BlacsOrbitalDescriptor: # XXX can we find a less confusing name?
    # This class 'describes' all the LCAO/Blacs-related stuff
    def __init__(self, supercomm, gd, bd, kpt_comm, nao):
        from gpaw import sl_diagonalize
        ncpus, mcpus, blocksize = sl_diagonalize[:3]

        bcommsize = bd.comm.size
        gcommsize = gd.comm.size
        bcommrank = bd.comm.rank
        
        shiftks = kpt_comm.rank * bcommsize * gcommsize
        stripe_ranks = shiftks + np.arange(bcommsize) * gcommsize
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        stripecomm = supercomm.new_communicator(stripe_ranks)
        blockcomm = supercomm.new_communicator(block_ranks)

        mynao = -((-nao) // bcommsize)
        # Range of basis functions for BLACS distribution of matrices:
        self.Mmax = nao
        self.Mstart = bcommrank * mynao
        self.Mstop = min(self.Mstart + mynao, self.Mmax)
        
        stripegrid = BlacsGrid(stripecomm, bcommsize, 1)
        blockgrid = BlacsGrid(blockcomm, mcpus, ncpus)

        # Striped layout
        mMdescriptor = stripegrid.new_descriptor(nao, nao, mynao, nao)

        # Blocked layout
        mmdescriptor = blockgrid.new_descriptor(nao, nao, blocksize, blocksize)

        # Striped layout but only nbands by nao (nbands <= nao)
        nMdescriptor = stripegrid.new_descriptor(nao, nao, bd.mynbands, nao)

        self.mMdescriptor = mMdescriptor
        self.mmdescriptro = mmdescriptor
        self.nMdescriptor = nMdescriptor
        self.mM2mm = Redistributor(supercomm, mMdescriptor, mmdescriptor)
        self.mm2nM = Redistributor(supercomm, mmdescriptor, nMdescriptor)
        
        self.gd = gd
        self.bd = bd

    def get_diagonalizer(self):
        return SLEXDiagonalizer(self.gd, self.bd, self.mM2mm, self.mm2nM)
