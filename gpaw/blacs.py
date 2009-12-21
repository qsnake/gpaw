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
from gpaw.utilities.blacs import scalapack_general_diagonalize_ex
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
    
    def diagonalize(self, H_mM, S_mm, C_nM, eps_n):
        indescriptor = self.indescriptor
        outdescriptor = self.outdescriptor
        blockdescriptor = self.blockdescriptor

        dtype = H_mM.dtype
        eps_M = np.empty(C_nM.shape[-1])

        # XXX where should inactive ranks be sorted out?
        if not indescriptor:
            shape = indescriptor.shape
            H_mM = np.zeros(shape, dtype=dtype)
        
        H_mm = blockdescriptor.zeros(dtype=dtype)
        C_mm = blockdescriptor.zeros(dtype=dtype)
        C_mM = outdescriptor.zeros(dtype=dtype)

        self.cols2blocks.redistribute(H_mM, H_mm)
        blockdescriptor.diagonalize_ex(H_mm, S_mm.copy(), C_mm, eps_M, UL='U',
                                       iu=self.bd.nbands)
        self.blocks2cols.redistribute(C_mm, C_mM) # XXX redist only nM somehow

        if outdescriptor:
            assert self.gd.comm.rank == 0
            bd = self.bd
            C_nM[:] = C_mM[:bd.mynbands, :]
            bd.distribute(eps_M[:bd.nbands], eps_n)
        else:
            assert self.gd.comm.rank != 0

        self.gd.comm.broadcast(C_nM, 0)
        self.gd.comm.broadcast(eps_n, 0)
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

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %s, block %s, lld %d, loc %s]'
        string = template % (classname, self.blacsgrid.context,
                             self.gshape,
                             self.bshape, self.lld, self.shape)
        return string

    def diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M, UL='U', iu=None):
        self.checkassert(H_mm)
        self.checkassert(S_mm)
        self.checkassert(C_mm)
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
    def __init__(self, world, gd, bd, kpt_comm, nao):
        ncpus, mcpus, blocksize = sl_diagonalize[:3]

        bcommsize = bd.comm.size
        gcommsize = gd.comm.size
        bcommrank = bd.comm.rank
        
        shiftks = kpt_comm.rank * bcommsize * gcommsize
        stripe_ranks = shiftks + np.arange(bcommsize) * gcommsize
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        stripecomm = world.new_communicator(stripe_ranks)
        blockcomm = world.new_communicator(block_ranks)

        mynao = -((-nao) // bcommsize)
        self.mynao = mynao

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
        self.mmdescriptor = mmdescriptor
        self.nMdescriptor = nMdescriptor
        self.mM2mm = Redistributor(blockcomm, mMdescriptor, mmdescriptor)
        self.mm2nM = Redistributor(blockcomm, mmdescriptor, nMdescriptor)
        
        self.gd = gd
        self.bd = bd

    def get_diagonalizer(self):
        return SLEXDiagonalizer(self.gd, self.bd, self.mM2mm, self.mm2nM)

    def get_overlap_descriptor(self):
        return self.mMdescriptor

    def get_diagonalization_descriptor(self):
        return self.mmdescriptor

    def get_coefficient_descriptor(self):
        return self.nMdescriptor

    def distribute_overlap_matrix(self, S1_qmM):
        blockdesc = self.mmdescriptor
        coldesc = self.mMdescriptor
        S_qmm = blockdesc.zeros(len(S1_qmM), S1_qmM.dtype)
        
        # XXX ugly hack
        S_qmM = coldesc.zeros(len(S1_qmM), S1_qmM.dtype)
        for S_mM, S_mm, S1_mM in zip(S_qmM, S_qmm, S1_qmM):
            if self.gd.comm.rank == 0:
                S_mM[:] = S1_mM
            self.mM2mm.redistribute(S_mM, S_mm)
        return S_qmm


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

    def get_diagonalizer(self):
        if sl_diagonalize:
            from gpaw.lcao.eigensolver import SLDiagonalizer
            diagonalizer = SLDiagonalizer()
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
