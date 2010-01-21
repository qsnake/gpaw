"""Module for high-level BLACS interface.

Usage
=====

A BLACS grid is a logical grid of processors.  To use BLACS, first
create a BLACS grid.  If comm contains 8 or more ranks, this example
will work::

  from gpaw.mpi import world
  from gpaw.blacs import BlacsGrid
  grid = BlacsGrid(world, 4, 2)

Use the processor grid to create various descriptors for distributed
arrays::

  block_desc = grid.new_descriptor(500, 500, 64, 64)
  local_desc = grid.new_descriptor(500, 500, 500, 500)

The first descriptor describes 500 by 500 arrays distributed amongst
the 8 CPUs of the BLACS grid in blocks of 64 by 64 elements (which is
a sensible block size).  That means each CPU has many blocks located
all over the array::

  print world.rank, block_desc.shape, block_desc.gshape

Here block_desc.shape is the local array shape while gshape is the
global shape.  The local array shape varies a bit on each CPU as the
block distribution may be slightly uneven.

The second descriptor, local_desc, has a block size equal to the
global size of the array, and will therefore only have one block.
This block will then reside on the first CPU -- local_desc therefore
represents non-distributed arrays.  Let us instantiate some arrays::

  H_MM = local_desc.empty()
  
  if world.rank == 0:
      assert H_MM.shape == (500, 500)
      H_MM[:, :] = calculate_hamiltonian_or_something()
  else:
      assert H_MM.shape[0] == 0 or H_MM.shape[1] == 0

  H_mm = block_desc.empty()
  print H_mm.shape # many elements on all CPUs

We can then redistribute the local H_MM into H_mm::

  from gpaw.blacs import Redistributor
  redistributor = Redistributor(world, local_desc, block_desc)
  redistributor.redistribute(H_MM, H_mm)
  
Now we can run parallel linear algebra on H_mm.  This will diagonalize
H_mm, place the eigenvectors in C_mm and the eigenvalues globally in
eps_M::

  eps_M = np.empty(500)
  C_mm = block_desc.empty()
  block_desc.diagonalize_ex(H_mm, C_mm, eps_M)

We can redistribute C_mm back to the master process if we want::

  C_MM = local_desc.empty()
  redistributor2 = Redistributor(world, block_desc, local_desc)
  redistributor2.redistribute(C_mm, C_MM)

If somebody wants to do all this more easily, they will probably write
a function for that.

List of interesting classes
===========================

 * BlacsGrid
 * BlacsDescriptor
 * Redistributor

The other classes in this module are coded specifically for GPAW and
are inconvenient to use otherwise.

The module gpaw.utilities.blacs contains several functions like gemm,
gemv and r2k.  These functions may or may not have appropriate
docstings, and may use Fortran-like variable naming.  Also, either
this module or gpaw.utilities.blacs will be renamed at some point.

"""

import numpy as np

from gpaw import sl_diagonalize
from gpaw.mpi import SerialCommunicator, serial_comm
from gpaw.utilities.blas import gemm, r2k, gemmdot
from gpaw.utilities.blacs import scalapack_general_diagonalize_ex, \
    scalapack_diagonalize_ex, pblas_simple_gemm
from gpaw.utilities.timing import nulltimer
from gpaw.utilities.tools import tri2full
import _gpaw


INACTIVE = -1
BLOCK_CYCLIC_2D = 1


class BlacsGrid:
    """Class representing a 2D grid of processors sharing a Blacs context.
        
    A BLACS grid defines a logical M by N ordering of a collection of
    CPUs.  A BLACS grid can be used to create BLACS descriptors.  On
    an npcol by nprow BLACS grid, a matrix is distributed amongst M by
    N CPUs along columns and rows, respectively, while the matrix
    shape and blocking properties are determined by the descriptors.

    Use the method new_descriptor() to create any number of BLACS
    descriptors sharing the same CPU layout.
    
    Most matrix operations require the involved matrices to all be on
    the same BlacsGrid.  Use a Redistributor to redistribute matrices
    from one BLACS grid to another if necessary.

    Parameters::

      * comm:  MPI communicator for CPUs of the BLACS grid or None.  A BLACS
        grid may use all or some of the CPUs of the communicator.
      * nprow:  Number of CPU rows.
      * npcol: Number of CPU columns.
      * order: 'R' or 'C', meaning rows or columns.  I'm not sure what this 
        does, it probably interchanges the meaning of rows and columns. XXX

    Complicated stuff
    -----------------

    It may be useful to know that a BLACS grid is said to be active
    and will evaluate to True on any process where comm is not None
    *and* comm.rank < nprow * npcol.  Otherwise it is considered
    inactive and evaluates to False.  Ranks where a grid is inactive
    never do anything at all.

    BLACS identifies each grid by a unique ID number called the
    context (frequently abbreviated ConTxt).  Grids on inactive ranks
    have context -1."""
    def __init__(self, comm, nprow, npcol, order='R'):
        assert nprow > 0
        assert npcol > 0
        assert len(order) == 1
        assert order in 'CcRr'

        if isinstance(comm, SerialCommunicator):
            raise ValueError('Instance of SerialCommunicator not supported')
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
        """Create a new descriptor from this BLACS grid.

        See documentation for BlacsDescriptor.__init__."""
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
    matrix descriptor with BLACS."""
    
    def __init__(self, M, N):
        self.shape = (M, N)
    
    def __nonzero__(self):
        return self.shape[0] != 0 and self.shape[1] != 0

    def zeros(self, n=(), dtype=float):
        """Return array of zeroes with the correct size on all CPUs.

        The last two dimensions will be equal to the shape of this
        descriptor.  If specified as a tuple, can have any preceding
        dimension."""
        return self._new_array(np.zeros, n, dtype)

    def empty(self, n=(), dtype=float):
        """Return array of zeros with the correct size on all CPUs.

        See zeros()."""
        return self._new_array(np.empty, n, dtype)

    def _new_array(self, func, n, dtype):
        if isinstance(n, int):
            n = n,
        shape = n + self.shape
        return func(shape, dtype)

    def check(self, a_mn):
        """Check that specified array is compatible with this descriptor."""
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
    """Class representing a 2D matrix distribution on a blacs grid.

    A BlacsDescriptor represents a particular shape and distribution
    of matrices.  A BlacsDescriptor has a global matrix shape and a
    rank-dependent local matrix shape.  The local shape is not
    necessarily equal on all ranks.

    A numpy array is said to be compatible with a BlacsDescriptor if,
    on all ranks, the shape of the numpy array is equal to the local
    shape of the BlacsDescriptor.  Compatible arrays can be created
    conveniently with the zeros() and empty() methods.

    An array with a global shape of M by N is distributed such that
    each process gets a number of distinct blocks of size mb by nb.
    The blocks on one process generally reside in very different areas
    of the matrix to improve load balance.
    
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

    Parameters:
     * blacsgrid: the BLACS grid of processors to distribute matrices.
     * M: global row count
     * N: global column count
     * mb: number of rows per block
     * nb: number of columns per block
     * rsrc: rank on which the first row is stored
     * csrc: rank on which the first column is stored

    Complicated stuff
    -----------------
    
    If there is trouble with matrix shapes, the below caveats are
    probably the reason.

    Depending on layout, a descriptor may have a local shape of zero
    by N or something similar.  If the row blocksize is 7, the global
    row count is 10, and the blacs grid contains 3 row processes: The
    first process will have 7 rows, the next will have 3, and the last
    will have 0.  The shapes in this case must still be correctly
    given to BLACS functions, which can be confusing.
    
    A blacs descriptor must also give the correct local leading
    dimension (lld), which is the local array size along the
    memory-contiguous direction in the matrix, and thus equal to the
    local column number, *except* when local shape is zero, but the
    implementation probably works.

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
        """Return a nine-element array representing this descriptor.
        
        In the C/Fortran code, a BLACS descriptor is represented by a
        special array of arcane nature.  The value of asarray() must 
        generally be passed to BLACS functions in the C code."""
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
        """Diagonalize symmetric matrix using Expert Driver algorithm.

        Solves the eigenvalue equation::

          H_nn C_nn = eps_n C_nn

        Diagonalizes H_nn and writes eigenvectors to C_nn.  Both H_nn
        and C_nn must be compatible with this descriptor.  Values in
        H_nn will be overwritten.

        Eigenvalues are written to the global array eps_n.

        UL can be either 'U' (default) or 'L', meaning that the
        matrices are taken to be upper or lower triangular.

        If the integer iu is specified, calculates only eigenvectors
        corresponding to the lowest iu eigenvalues."""
        scalapack_diagonalize_ex(self, H_nn, C_nn, eps_n, UL, iu=iu)

    def general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M, UL='U', iu=None):
        """Solve generalized eigenvalue problem.
        
        Solves::

          H_mm C_mm = S_mm C_mm eps_M.

        H_mm, C_mm and S_mm are distributed arrays all compatible with
        this descriptor.  eps_M is a global output array of
        eigenvalues.  H_mm and S_mm will be overwritten.

        See also documentation for diagonalize_ex."""
        scalapack_general_diagonalize_ex(self, H_mm, S_mm, C_mm, eps_M,
                                         UL, iu=iu)


class Redistributor:
    """Class for redistributing BLACS matrices on different contexts."""
    def __init__(self, supercomm, srcdescriptor, dstdescriptor):
        """Create redistributor.

        Source and destination descriptors may reside on different
        BLACS grids, but the descriptors should describe arrays with
        the same number of elements.  

        The communicators of the BLACS grid of srcdescriptor as well
        as that of dstdescriptor *must* both be subcommunicators of
        supercomm."""
        self.supercomm = supercomm
        self.srcdescriptor = srcdescriptor
        self.dstdescriptor = dstdescriptor
    
    def redistribute_submatrix(self, src_mn, dst_mn, subM, subN):
        """Redistribute submatrix into other submatrix.  

        A bit more general than redistribute().  See also redistribute()."""
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
        """Redistribute src_mn to dst_mn.

        src_mn must be compatible with the source descriptor of this
        redistributor, while dst_mn must be compatible with the
        destination descriptor."""
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


class SLDenseLinearAlgebra:
    """ScaLAPACK Dense Linear Algebra.

    This class is instantiated in LCAO.  Not for casual use."""
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
    def __init__(self, world, gd, bd, kpt_comm, nao, ncpus, mcpus, blocksize,
                 timer=nulltimer):
        
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
        naoblocksize = -((-nao) // bcommsize)
        self.nao = nao

        # Range of basis functions for BLACS distribution of matrices:
        self.Mmax = nao
        self.Mstart = bcommrank * naoblocksize
        self.Mstop = min(self.Mstart + naoblocksize, self.Mmax)
        self.mynao = self.Mstop - self.Mstart

        # Column layout for one matrix per band rank:
        columngrid = BlacsGrid(bd.comm, bcommsize, 1)
        self.mMdescriptor = columngrid.new_descriptor(nao, nao, naoblocksize,
                                                      nao)
        self.nMdescriptor = columngrid.new_descriptor(nbands, nao, mynbands,
                                                      nao)
        assert self.mMdescriptor.shape == (self.mynao, nao)

        #parallelprint(world, (mynao, self.mMdescriptor.shape))

        # Column layout for one matrix in total (only on grid masters):
        single_column_grid = BlacsGrid(columncomm, bcommsize, 1)
        mM_unique_descriptor = single_column_grid.new_descriptor(nao, nao,
                                                                 naoblocksize,
                                                                 nao)
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

        self.orbital_comm = bd.comm
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

    def distribute_overlap_matrix(self, S_qmM):
        xshape = S_qmM.shape[:-2]
        nm, nM = S_qmM.shape[-2:]
        S_qmM = S_qmM.reshape(-1, nm, nM)
        
        blockdesc = self.mmdescriptor
        coldesc = self.mM_unique_descriptor
        S_qmm = blockdesc.zeros(len(S_qmM), S_qmM.dtype)

        if not coldesc: # XXX ugly way to sort out inactive ranks
            S_qmM = coldesc.zeros(len(S_qmM), S_qmM.dtype)
        
        self.timer.start('Distribute overlap matrix')
        for S_mM, S_mm in zip(S_qmM, S_qmm):
            self.mM2mm.redistribute(S_mM, S_mm)
        self.timer.stop('Distribute overlap matrix')
        return S_qmm.reshape(xshape + blockdesc.shape)

    def get_overlap_matrix_shape(self):
        return self.mmdescriptor.shape

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

    def get_transposed_density_matrix(self, f_n, C_nM, rho_mM=None):
        # XXX for the complex case, find out whether this or the other
        # method should be changed
        return self.calculate_density_matrix(f_n, C_nM, rho_mM)
        

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
        self.orbital_comm = serial_comm

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

    def get_overlap_matrix_shape(self):
        return self.nao, self.nao

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        # Only a madman would use a non-transposed density matrix.
        # Maybe we should use the get_transposed_density_matrix instead
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # XXX Should not conjugate, but call gemm(..., 'c')
        # Although that requires knowing C_Mn and not C_nM.
        # that also conforms better to the usual conventions in literature
        Cf_Mn = C_nM.T.conj() * f_n
        gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
        self.bd.comm.sum(rho_MM)
        return rho_MM

    def get_transposed_density_matrix(self, f_n, C_nM, rho_MM=None):
        return self.calculate_density_matrix(f_n, C_nM, rho_MM).T.copy()

        #if rho_MM is None:
        #    rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        #C_Mn = C_nM.T.copy()
        #gemm(1.0, C_Mn, f_n[np.newaxis, :] * C_Mn, 0.0, rho_MM, 'c')
        #self.bd.comm.sum(rho_MM)
        #return rho_MM

    def alternative_calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # Alternative suggestion. Might be faster. Someone should test this
        C_Mn = C_nM.T.copy()
        r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
        tri2full(rho_MM)
        return rho_MM
