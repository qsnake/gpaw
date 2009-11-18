import numpy as np

from gpaw.mpi import SerialCommunicator
from gpaw.utilities.blacs import scalapack_diagonalize_ex
import _gpaw


INACTIVE = -1
BLOCK_CYCLIC_2D = 1


class SLEXDiagonalizer:
    """ScaLAPACK Expert Driver diagonalizer."""
    def __init__(self, supercomm, kpt_comm, gd, bd, ncpu, mcpu, blocksize,
                 nao):
        self.supercomm = supercomm
        self.bd = bd
        self.kpt_comm = kpt_comm
        self.gd = gd
        
        bcommsize = bd.comm.size
        gcommsize = gd.comm.size
        
        shiftks = kpt_comm.rank * bcommsize * gcommsize
        
        mynbands = self.bd.mynbands
        nbands = self.bd.nbands
        
        stripe_ranks = shiftks + np.arange(bcommsize) * gcommsize
        block_ranks = shiftks + np.arange(bcommsize * gcommsize)
        stripecomm = supercomm.new_communicator(stripe_ranks)
        blockcomm = supercomm.new_communicator(block_ranks)
        
        columngrid = BlacsGrid(stripecomm, 1, bcommsize)
        blockgrid = BlacsGrid(blockcomm, ncpu, mcpu)
        
        myncolumns = -((-nao) // bcommsize)
        self.indescriptor = columngrid.new_descriptor(nao, nao, nao,
                                                      myncolumns)
        self.blockdescriptor = blockgrid.new_descriptor(nao, nao, blocksize,
                                                        blocksize)
        self.outdescriptor = columngrid.new_descriptor(nao, nao, nao, mynbands)
        self.redistributor = Redistributor(blockcomm)

    def diagonalize(self, H_MM, S_MM, eps_n, kpt):
        descriptor1 = self.indescriptor
        descriptor1b = self.outdescriptor
        descriptor2 = self.blockdescriptor

        redistributor = self.redistributor

        dtype = H_MM.dtype
        
        colS = descriptor1.new_matrix(dtype)
        colH = descriptor1.new_matrix(dtype)
        if colS.A_mn.shape != (0, 0):
            assert colS.A_mn.T.flags.contiguous # A_mn is Fortran ordered
            # This is not a 'true' transpose, it should be a regular copy
            # due to the Fortran/C ordering
            colS.A_mn.T[:] = S_MM
            colH.A_mn.T[:] = H_MM

        sqrS = descriptor2.new_matrix(dtype)
        sqrH = descriptor2.new_matrix(dtype)

        dst_colH = descriptor1b.new_matrix(dtype)

        redistributor.redistribute(colS, sqrS)
        redistributor.redistribute(colH, sqrH)

        eps_n[:], H1_MM = scalapack_diagonalize_ex(sqrH.A_mn, descriptor2,
                                                   'U', sqrS.A_mn)
        if H1_MM is not None:
            sqrH.A_mn[:] = H1_MM

        redistributor.redistribute(sqrH, dst_colH)
        H_MM = dst_colH.A_mn

        if H_MM.shape != (0, 0):
            assert self.gd.comm.rank == 0
            bd = self.bd
            kpt.C_nM[:] = H_MM[:, :bd.mynbands].T
            bd.distribute(eps_n[:bd.nbands], kpt.eps_n)
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

    def is_active(self):
        """Whether context is active on this rank."""
        return self.context != INACTIVE

    def new_descriptor(self, M, N, mb, nb, rsrc=0, csrc=0):
        return BlacsDescriptor(self, M, N, mb, nb, rsrc, csrc)

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[comm:size=%d,rank=%d; context=%d; %dx%d]'
        string = template % (classname, self.comm.size, self.comm.rank, 
                             self.context, self.nprow, self.npcol)
        return string
    
    def __del__(self):
        _gpaw.blacs_destroy(self.context)


class BlacsDescriptor:
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
        context = blacsgrid.context
        
        if blacsgrid.is_active():
            locM, locN = _gpaw.get_blacs_shape(context, self.M, self.N,
                                               self.mb, self.nb, 
                                               self.rsrc, self.csrc)
        else:
            locM, locN = 0, 0
        self.locM = locM
        self.locN = locN
        self.lld  = locM # lld = 'local leading dimension'

    def new_matrix(self, dtype):
        return BlacsMatrix(self, dtype)

    def asarray(self):
        arr = np.array([BLOCK_CYCLIC_2D, self.blacsgrid.context, 
                        self.M, self.N, self.mb, self.nb, self.rsrc, self.csrc,
                        max(0, self.lld)], np.int32)
        return arr

    def __str__(self):
        classname = self.__class__.__name__
        template = '%s[context=%d, glob %dx%d, mb %dx%d, lld %d, locM/N %dx%d]'
        string = template % (classname, self.blacsgrid.context, self.M, self.N, 
                             self.mb, self.nb, self.lld, self.locM, self.locN)
        return string


class BlacsMatrix:
    def __init__(self, blacs_descriptor, dtype):
        self.descriptor = blacs_descriptor
        self.dtype = dtype
        locM = blacs_descriptor.locM
        locN = blacs_descriptor.locN
        # Could be empty, but we don't want to challenge the ability of certain
        # libraries to handle random NaNs.
        self.A_mn = np.zeros((locM, locN), dtype=dtype, order='Fortran') # XXX

    # Parallel operations to be implemented as methods

class Redistributor:
    """Class for redistributing BLACS matrices on different contexts."""
    def __init__(self, supercomm):
        self.supercomm = supercomm
    
    def redistribute_submatrix(self, srcmatrix, dstmatrix, subM, subN):
        # self.supercomm must be a supercommunicator of the communicators
        # corresponding to the context of srcmatrix as well as dstmatrix.
        # We should verify this somehow.
        src_mn = srcmatrix.A_mn
        dst_mn = dstmatrix.A_mn
        dtype = src_mn.dtype
        assert dtype == dst_mn.dtype
        srcdesc = srcmatrix.descriptor
        
        isreal = (dtype == float)
        assert dtype == float or dtype == complex
        
        _gpaw.scalapack_redist(srcmatrix.descriptor.asarray(), 
                               dstmatrix.descriptor.asarray(),
                               srcmatrix.A_mn, dstmatrix.A_mn,
                               self.supercomm, subM, subN, isreal)

    def redistribute(self, srcmatrix, dstmatrix):
        subM = srcmatrix.descriptor.M
        subN = srcmatrix.descriptor.N
        self.redistribute_submatrix(srcmatrix, dstmatrix, subM, subN)



class BlacsParallelization:
    def __init__(self, master_comm, kpt_comm, gd, bd):
        self.master_comm = master_comm
        self.kpt_comm = kpt_comm
        self.gd = gd
        self.bd = bd
        
        self.stripegrid = BlacsGrid(band_comm, 1, band_comm.size)
        self.squaregrid = BlacsGrid()

    def get_diagonalizer(self):
        return SLEXDiagonalizer()



def parallelprint(comm, obj):
    import sys
    for a in range(comm.size):
        if a == comm.rank:
            print 'rank=%d' % a
            print obj
            print
            sys.stdout.flush()
        comm.barrier()




def simpletest(M=16, N=16):
    from gpaw.mpi import world
    ncpus = world.size

    grid0 = BlacsGrid(world, 1, 1)
    grid1 = BlacsGrid(world, 1, ncpus)
    #grid2 = BlacsGrid(world, 2, 2)

    desc1 = grid1.new_descriptor(M, N, M, ncpus, 0, 0)
    parallelprint(world, desc1)
    mat1 = desc1.new_matrix(float)
    A_mn = mat1.A_mn
    A_mn[:] = world.rank

    desc0 = grid0.new_descriptor(M, N, M, N, 0, 0)
    mat0 = desc0.new_matrix(float)
    mat0.A_mn[:] = world.size + 1

    redistributor = Redistributor(world)
    redistributor.redistribute(mat1, mat0)
    parallelprint(world, mat1.A_mn)
    print
    print
    parallelprint(world, mat0.A_mn)

def main():
    simpletest()
    #othertest()

def othertest():
    from gpaw.mpi import world
    bg = BlacsGrid(world, 2, 2)
    print bg.context
    descriptor = bg.new_descriptor(16, 16, 8, 8)
    print descriptor.lld
    import sys
    m = descriptor.new_matrix(float)
    m.A_mn[:] = world.rank
    sys.stdout.flush()
    world.barrier()

    parallelprint(world, m.A_mn)

if __name__ == '__main__':
    main()
