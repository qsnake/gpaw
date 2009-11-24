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
    def __init__(self, supercomm, stripecomm, blockcomm, gd, bd, ncpu,
                 mcpu, blocksize, nao):
        self.supercomm = supercomm
        self.bd = bd
        self.gd = gd
        self.nao = nao
        
        bcommsize = bd.comm.size
        gcommsize = gd.comm.size
        
        
        mynbands = self.bd.mynbands
        nbands = self.bd.nbands
                
        columngrid = BlacsGrid(stripecomm, 1, bcommsize)
        blockgrid = BlacsGrid(blockcomm, ncpu, mcpu)
        
        myncolumns = -((-nao) // bcommsize)
        self.indescriptor = columngrid.new_descriptor(nao, nao, nao,
                                                      myncolumns)
        self.blockdescriptor = blockgrid.new_descriptor(nao, nao, blocksize,
                                                        blocksize)
        self.outdescriptor = columngrid.new_descriptor(nao, nao, nao, mynbands)
        
        self.cols2blocks = Redistributor(blockcomm, self.indescriptor,
                                         self.blockdescriptor)
        self.blocks2cols = Redistributor(blockcomm, self.blockdescriptor,
                                         self.outdescriptor)
    
    def diagonalize(self, H_mM, S_mM, eps_M, kpt):
        indescriptor = self.indescriptor
        outdescriptor = self.outdescriptor
        blockdescriptor = self.blockdescriptor

        dtype = H_mM.dtype

        S_Mm = indescriptor.zeros(dtype=dtype)
        H_Mm = indescriptor.zeros(dtype=dtype)

        if indescriptor:
            S_Mm.T[:] = S_mM
            H_Mm.T[:] = H_mM
        
        S_mm = blockdescriptor.zeros(dtype)
        H_mm = blockdescriptor.zeros(dtype)
        C_mm = blockdescriptor.zeros(dtype)
        C_Mn = outdescriptor.zeros(dtype)
        
        self.cols2blocks.redistribute(S_Mm, S_mm)
        self.cols2blocks.redistribute(H_Mm, H_mm)

        blockdescriptor.diagonalize_ex(H_mm, S_mm, C_mm, eps_M, 'U')

        self.blocks2cols.redistribute(C_mm, C_Mn)

        if outdescriptor:
            assert self.gd.comm.rank == 0
            bd = self.bd
            kpt.C_nM[:] = C_Mn[:, :bd.mynbands].T
            bd.distribute(eps_M[:bd.nbands], kpt.eps_n)
        else:
            assert self.gd.comm.rank != 0

        self.gd.comm.broadcast(kpt.C_nM, 0)
        self.gd.comm.broadcast(kpt.eps_n, 0)
        return 0


class BlacsGrid:
    """Class representing a 2D grid of processors sharing a Blacs context."""
    def __init__(self, comm, nprow, npcol, order='R'):
        if isinstance(comm, SerialCommunicator):
            raise ValueError('you forgot mpi AGAIN')
        if comm is None: # if and only if rank is not part of the communicator
            # (which is not entirely sensible.  That rank must then belong in
            # a different communicator.  Which one?)
            context = INACTIVE
        else:
            # This call may also return INACTIVE
            context = _gpaw.new_blacs_context(comm, nprow, npcol, order)
            assert nprow * npcol <= comm.size
        assert nprow > 0
        assert npcol > 0
        assert len(order) == 1
        assert order in 'CcRr'
        
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
        return np.zeros(self.shape, dtype, order='F') #XXX

    def empty(self, dtype=float):
        return np.empty(self.shape, dtype, order='F') #XXX

    def check(self, a_mn):
        return a_mn.shape == self.shape and a_mn.flags.f_contiguous #XXX
        

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
            locM, locN = _gpaw.get_blacs_shape(self.blacsgrid.context,
                                               self.M, self.N,
                                               self.mb, self.nb, 
                                               self.rsrc, self.csrc)
        else:
            locM, locN = 0, 0
        
        MatrixDescriptor.__init__(self, locM, locN)
        
        self.active = locM > 0 and locN > 0
        
        self.bshape = (self.mb, self.nb) # Shape of one block
        self.gshape = (M, N) # Global shape of array
        self.lld  = locM # lld = 'local leading dimension'

    def asarray(self):
        arr = np.array([BLOCK_CYCLIC_2D, self.blacsgrid.context, 
                        self.M, self.N, self.mb, self.nb, self.rsrc, self.csrc,
                        max(0, self.lld)], np.int32)
        return arr

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %dx%d, mb %dx%d, lld %d, locM/N %dx%d]'
        string = template % (classname, self.blacsgrid.context,
                             self.M, self.N, 
                             self.mb, self.nb, self.lld, self.locM, self.locN)
        return string

    def diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M, UL='U'):
        assert self.check(H_mm)
        assert self.check(S_mm)
        assert self.check(C_mm)
        scalapack_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M, UL)


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

        assert self.srcdescriptor.check(src_mn)
        assert self.dstdescriptor.check(dst_mn)
        
        _gpaw.scalapack_redist(self.srcdescriptor.asarray(), 
                               self.dstdescriptor.asarray(),
                               src_mn, dst_mn,
                               self.supercomm, subM, subN, isreal)
    
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

