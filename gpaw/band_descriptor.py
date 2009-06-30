# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Band-descriptors for blocked/strided groups.

This module contains classes defining two kinds of band groups:

* Blocked groups with contiguous band indices.
* Strided groups with evenly-spaced band indices.
"""

import numpy as np

from gpaw import debug
import gpaw.mpi as mpi

NONBLOCKING = False

class BandDescriptor:
    """Descriptor-class for 

    A ``BandDescriptor`` object holds information on how functions, such
    as wave functions and corresponding occupation numbers, are divided
    into groups according to band indices. The main information here is
    how many bands are stored on each processor and who gets what.

    There are methods for tasks such as allocating arrays, performing
    rotation- and mirror-symmetry operations and integrating functions
    over space.  All methods work correctly also when the domain is
    parallelized via domain decomposition.

    This is how a 12 band array is laid out in memory on 3 cpu's::

      a) Blocked groups       b) Strided groups

           3    7   11             9   10   11
     myn   2 \  6 \ 10       myn   6    7    8
      |    1  \ 5  \ 9        |    3    4    5
      |    0    4    8        |    0    1    2
      |                       |
      +----- band_rank        +----- band_rank

    Example:

     >>> a = np.zeros((3, 4))
     >>> a.ravel()[:] = range(12)
     >>> a
     array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
     >>> b = np.zeros((4, 3))
     >>> b.ravel()[:] = range(12)
     >>> b.T
     array([[ 0,  3,  6,  9],
            [ 1,  4,  7, 10],
            [ 2,  5,  8, 11]])
     """
    
    def __init__(self, nbands, comm=None, strided=False):
        """Construct band-descriptor object.

        Parameters:

        nbands: int
            Number of bands.
        comm: MPI-communicator
            Communicator for band-groups.
        strided: bool
            Enable strided band distribution for better
            load balancing with many unoccupied bands.

        Note that if comm.size is 1, then all bands are contained on
        a single CPU and blocked/strided grouping loses its meaning.

        Attributes:

        ============  ======================================================
        ``nbands``    Number of bands in total.
        ``mynbands``  Number of bands on this CPU.
        ``beg``       Beginning of band indices in group (inclusive).
        ``end``       End of band indices in group (exclusive).
        ``step``      Stride for band indices between ``beg`` and ``end``.
        ``comm``      MPI-communicator for band distribution.
        ============  ======================================================
        """
        
        if comm is None:
            comm = mpi.serial_comm
        self.comm = comm
        self.rank = self.comm.rank

        self.nbands = nbands

        if self.nbands % self.comm.size != 0:
            raise RuntimeError('Cannot distribute %d bands to %d processors' %
                               (self.nbands, self.comm.size))

        self.mynbands = self.nbands // self.comm.size
        self.strided = strided

        nslice = self.get_slice(self.comm.size)
        self.beg, self.end, self.step = nslice.indices(self.nbands)

    def __len__(self):
        return self.mynbands

    def get_slice(self, band_rank=None):
        if band_rank is None:
            band_rank = self.comm.rank

        if self.strided:
            nstride = self.comm.size
            nslice = slice(band_rank, None, nstride)
        else:
            n0 = band_rank * self.mynbands
            nslice = slice(n0, n0 + self.mynbands)
        return nslice

    def get_band_indices(self, band_rank=None):
        nslice = self.get_slice(band_rank)
        return np.arange(*nslice.indices(self.nbands))

    def get_band_ranks(self):
        rank_n = np.empty(self.nbands, dtype=int)
        for band_rank in range(self.comm.size):
            nslice = self.get_slice(band_rank)
            rank_n[nslice] = band_rank
        assert (rank_n >= 0).all() and (rank_n < self.comm.size).all()
        return rank_n

    def who_has(self, n):
        if self.strided:
            myn, band_rank = divmod(n, self.comm.size)
        else:
            band_rank, myn = divmod(n, self.mynbands)

        return band_rank, myn

    def global_index(self, myn, band_rank=None):
        if band_rank is None:
            band_rank = self.comm.rank

        if self.strided:
            n = band_rank + myn * self.comm.size
        else:
            n = band_rank * self.mynbands + myn

        return n

    def get_size_of_global_array(self):
        return (self.nbands,)

    def zeros(self, n=(), dtype=float, global_array=False):
        """Return new zeroed 3D array for this domain.

        The type can be set with the ``dtype`` keyword (default:
        ``float``).  Extra dimensions can be added with ``n=dim``.  A
        global array spanning all domains can be allocated with
        ``global_array=True``."""
        #TODO XXX doc
        return self._new_array(n, dtype, True, global_array)
    
    def empty(self, n=(), dtype=float, global_array=False):
        """Return new uninitialized 3D array for this domain.

        The type can be set with the ``dtype`` keyword (default:
        ``float``).  Extra dimensions can be added with ``n=dim``.  A
        global array spanning all domains can be allocated with
        ``global_array=True``."""
        #TODO XXX doc
        return self._new_array(n, dtype, False, global_array)
        
    def _new_array(self, n=(), dtype=float, zero=True, global_array=False):
        if global_array:
            shape = self.get_size_of_global_array()
        else:
            shape = (self.mynbands,)
            
        if isinstance(n, int):
            n = (n,)

        shape = tuple(shape) + n

        if zero:
            return np.zeros(shape, dtype)
        else:
            return np.empty(shape, dtype)

    def collect(self, a_nx, broadcast=False):
        """Collect distributed array to master-CPU or all CPU's."""
        if self.comm.size == 1:
            return a_nx

        xshape = a_nx.shape[1:]

        # Optimization for blocked groups
        if not self.strided:
            if broadcast:
                A_nx = self.empty(xshape, a_nx.dtype, global_array=True)
                self.comm.all_gather(a_nx, A_nx)
                return A_nx

            if self.rank == 0:
                A_nx = self.empty(xshape, a_nx.dtype, global_array=True)
            else:
                A_nx = None
            self.comm.gather(a_nx, 0, A_nx)
            return A_nx

        # Collect all arrays on the master:
        if self.rank != 0:
            # There can be several sends before the corresponding receives
            # are posted, so use syncronous send here
            self.comm.ssend(a_nx, 0, 3011)
            if broadcast:
                A_nx = self.empty(xshape, a_nx.dtype, global_array=True)
                self.comm.broadcast(A_nx, 0)
                return A_nx
            else:
                return None

        # Put the band groups from the slaves into the big array
        # for the whole collection of bands:
        A_nx = self.empty(xshape, a_nx.dtype, global_array=True)
        for band_rank in range(self.comm.size):
            if band_rank != 0:
                a_nx = self.empty(xshape, a_nx.dtype, global_array=False)
                self.comm.receive(a_nx, band_rank, 3011)
            A_nx[self.get_slice(band_rank), ...] = a_nx

        if broadcast:
            self.comm.broadcast(A_nx, 0)
        return A_nx

    def distribute(self, B_nx, b_nx):
        """ distribute full array B_nx to band groups, result in
        b_nx. b_nx must be allocated."""

        if self.comm.size == 1:
            b_nx[:] = B_nx
            return

        # Optimization for blocked groups
        if not self.strided:
            self.comm.scatter(B_nx, b_nx, 0)
            return

        if self.rank != 0:
            self.comm.receive(b_nx, 0, 421)
            return
        else:
            requests = []
            for band_rank in range(self.comm.size):
                if band_rank != 0:
                    a_nx = B_nx[self.get_slice(band_rank), ...].copy()
                    request = self.comm.send(a_nx, r, 421, NONBLOCKING)
                    # Remember to store a reference to the
                    # send buffer (a_nx) so that is isn't
                    # deallocated:
                    requests.append((request, a_nx))
                else:
                    b_nx[:] = B_nx[self.get_slice(), ...]
                        
            for request, a_nx in requests:
                self.comm.wait(request)

    def matrix_assembly(self, A_qnn, A_NN, hermitian):
        if self.comm.size == 1:
            self.blockwise_assign(A_qnn, A_NN, 0, hermitian)
            return

        if self.rank == 0:
            for band_rank in range(self.comm.size):
                if band_rank > 0:
                    self.comm.receive(A_qnn, band_rank, 13)
                self.blockwise_assign(A_qnn, A_NN, band_rank, hermitian)
        else:
            self.comm.send(A_qnn, 0, 13)

    def blockwise_assign(self, A_qnn, A_NN, band_rank, hermitian):
        if hermitian:
            self.triangular_blockwise_assign(A_qnn, A_NN, band_rank)
        else:
            self.full_blockwise_assign(A_qnn, A_NN, band_rank)

    def triangular_blockwise_assign(self, A_qnn, A_NN, band_rank):
        N = self.mynbands
        B = self.comm.size
        assert band_rank in xrange(B)

        if B == 1:
            # Only fill in the lower part
            mask = np.tri(N, dtype=bool)
            A_NN[mask] = A_qnn.reshape((N,N))[mask]
            return

        # A_qnn[q2,myn1,myn2] on rank q1 is the q2'th overlap calculated
        # between <psi_n1| and A|psit_n2> where n1 <-> (q1,myn1) and 
        # n2 <-> ((q1+q2)%B,myn2) since we've sent/recieved q2 times.
        q1 = band_rank
        Q = B // 2 + 1

        # Note that for integer inequalities, these relations are useful (X>0):
        #     A*X > B   <=>   A > B//X   ^   A*X <= B   <=>   A <= B//X

        if self.strided:
            A_nbnb = A_NN.reshape((N, B, N, B))

            for q2 in range(Q):
                # n1 = (q1+q2)%B + myn1*B   ^   n2 = q1 + myn2*B
                #
                # We seek the lower triangular part i.e. n1 >= n2
                #   <=>   (myn2-myn1)*B <= (q1+q2)%B-q1
                #   <=>   myn2-myn1 <= dq//B
                dq = (q1+q2)%B-q1 # within ]-B; Q[ so dq//B is -1 or 0

                # Create mask for lower part of current block
                mask = np.tri(N, N, dq//B, dtype=bool)
                if debug:
                    m1,m2 = np.indices((N,N))
                    assert (mask == (m1 >= m2 - dq//B)).all()

                # Copy lower part of A_qnn[q2] to its rightfull place
                A_nbnb[:, (q1+q2)%B, :, q1][mask] = A_qnn[q2][mask]

                # Negate and transpose mask to get complementary mask
                mask = ~mask.T

                # Copy upper part of Hermitian conjugate of A_qnn[q2]
                A_nbnb[:, q1, :, (q1+q2)%B][mask] = A_qnn[q2].T.conj()[mask]
        else:
            A_bnbn = A_NN.reshape((B, N, B, N))

            # Optimization for the first block
            if q1 == 0:
                A_bnbn[:Q, :, 0] = A_qnn
                return

            for q2 in range(Q):
                # n1 = ((q1+q2)%B)*N + myn1   ^   n2 = q1*N + myn2
                #
                # We seek the lower triangular part i.e. n1 >= n2
                #   <=>   ((q1+q2)%B-q1)*N >= myn2-myn1
                #   <=>   myn2-myn1 <= dq*N
                #   <=>   entire block if dq > 0,
                #   ...   myn2 <= myn1 if dq == 0,
                #   ...   copy nothing if dq < 0
                if q1 + q2 < B:
                    A_bnbn[q1 + q2, :, q1] = A_qnn[q2]
                else:
                    A_bnbn[q1, :, q1 + q2 - B] = A_qnn[q2].T.conj()

    def full_blockwise_assign(self, A_qnn, A_NN, band_rank):
        N = self.mynbands
        B = self.comm.size
        assert band_rank in xrange(B)

        if B == 1:
            A_NN[:] = A_qnn.reshape((N,N))
            return

        # A_qnn[q2,myn1,myn2] on rank q1 is the q2'th overlap calculated
        # between <psi_n1| and A|psit_n2> where n1 <-> (q1,myn1) and 
        # n2 <-> ((q1+q2)%B,myn2) since we've sent/recieved q2 times.
        q1 = band_rank

        if self.strided:
            A_nbnb = A_NN.reshape((N, B, N, B))
            for q2 in range(B):
                A_nbnb[:, (q1+q2)%B, :, q1] = A_qnn[q2]
        else:
            A_bnbn = A_NN.reshape((B, N, B, N))
            for q2 in range(B):
                A_bnbn[(q1+q2)%B, :, q1] = A_qnn[q2]

    def extract_block(self, A_NN, q1, q2):
        N = self.mynbands
        B = self.comm.size

        if B == 1:
            return A_NN

        if self.strided:
            A_nbnb = A_NN.reshape((N, B, N, B))
            return A_nbnb[:, q1, :, q2].copy() # last dim must have unit stride
        else:
            A_bnbn = A_NN.reshape((B, N, B, N))
            return A_bnbn[q1, :, q2]

