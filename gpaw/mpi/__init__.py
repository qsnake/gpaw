# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys
import time
import atexit
import pickle
import numpy as np

from gpaw import debug
from gpaw import dry_run as dry_run_size
from gpaw.utilities import is_contiguous
from gpaw.utilities import scalapack, gcd
from gpaw.utilities.tools import md5_array

import _gpaw

MASTER = 0


class _Communicator:
    def __init__(self, comm, parent=None):
        """Construct a wrapper of the C-object for any MPI-communicator.

        Parameters:

        comm: MPI-communicator
            Communicator.

        Attributes:

        ============  ======================================================
        ``size``      Number of ranks in the MPI group.
        ``rank``      Number of this CPU in the MPI group.
        ``parent``    Parent MPI-communicator.
        ============  ======================================================
        """
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.parent = parent #XXX check C-object against comm.parent?

    def new_communicator(self, ranks):
        """Create a new MPI communicator for a subset of ranks in a group.
        Must be called with identical arguments by all relevant processes.

        Note that a valid communicator is only returned to the processes
        which are included in the new group; other ranks get None returned.

        Parameters:

        ranks: ndarray (type int)
            List of integers of the ranks to include in the new group.
            Note that these ranks correspond to indices in the current
            group whereas the rank attribute in the new communicators
            correspond to their respective index in the subset.

        """
        assert is_contiguous(ranks, int)
        sranks = np.sort(ranks)
        # Are all ranks in range?
        assert 0 <= sranks[0] and sranks[-1] < self.size
        # No duplicates:
        for i in range(len(sranks) - 1):
            assert sranks[i] != sranks[i + 1]
        assert len(ranks) > 0
        
        comm = self.comm.new_communicator(ranks)
        if comm is None:
            # This cpu is not in the new communicator:
            return None
        else:
            return _Communicator(comm, parent=self)

    def sum(self, a, root=-1):
        """Perform summation by MPI reduce operations of numerical data.

        Parameters:

        a: ndarray or value (type int, float or complex)
            Numerical data to sum over all ranks in the communicator group.
            If the data is a single value of type int, float or complex,
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the sum of
            the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float, complex)):
            return self.comm.sum(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float or tc == complex
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.sum(a, root)

    def product(self, a, root=-1):
        """Do multiplication by MPI reduce operations of numerical data.

        Parameters:

        a: ndarray or value (type int or float)
            Numerical data to multiply across all ranks in the communicator
            group. NB: Find the global product from the local products.
            If the data is a single value of type int or float (no complex),
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the product
            of the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float)):
            return self.comm.product(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.product(a, root)

    def max(self, a, root=-1):
        """Find maximal value by an MPI reduce operation of numerical data.

        Parameters:

        a: ndarray or value (type int or float)
            Numerical data to find the maximum value of across all ranks in
            the communicator group. NB: Find global maximum from local max.
            If the data is a single value of type int or float (no complex),
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the max of
            the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float)):
            assert isinstance(a, float)
            return self.comm.max(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.max(a, root)

    def min(self, a, root=-1):
        """Find minimal value by an MPI reduce operation of numerical data.

        Parameters:

        a: ndarray or value (type int or float)
            Numerical data to find the minimal value of across all ranks in
            the communicator group. NB: Find global minimum from local min.
            If the data is a single value of type int or float (no complex),
            the result is returned because the input argument is immutable.
            Otherwise, the reduce operation is carried out in-place such
            that the elements of the input array will represent the min of
            the equivalent elements across all processes in the group.
        root: int (default -1)
            Rank of the root process, on which the outcome of the reduce
            operation is valid. A root rank of -1 signifies that the result
            will be distributed back to all processes, i.e. a broadcast.

        """
        if isinstance(a, (int, float)):
            assert isinstance(a, float)
            return self.comm.min(a, root)
        else:
            tc = a.dtype
            assert tc == int or tc == float
            assert is_contiguous(a, tc)
            assert root == -1 or 0 <= root < self.size
            self.comm.min(a, root)

    def scatter(self, a, b, root):
        """Distribute data from one rank to all other processes in a group.

        Parameters:

        a: ndarray (ignored on all ranks different from root; use None)
            Source of the data to distribute, i.e. send buffer on root rank.
        b: ndarray
            Destination of the distributed data, i.e. local receive buffer.
            The size of this array multiplied by the number of process in
            the group must match the size of the source array on the root.
        root: int
            Rank of the root process, from which the source data originates.

        The reverse operation is ``gather``.

        Example::

          # The master has all the interesting data. Distribute it.
          if comm.rank == 0:
              data = np.random.normal(size=N*comm.size)
          else:
              data = None
          mydata = np.empty(N, dtype=float)
          comm.scatter(data, mydata, 0)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Extract my part directly
              mydata[:] = data[0:N]
              # Distribute parts to the slaves
              for rank in range(1, comm.size):
                  buf = data[rank*N:(rank+1)*N]
                  comm.send(buf, rank, tag=123)
          else:
              # Receive from the master
              comm.receive(mydata, 0, tag=123)

        """
        if self.rank == root:
            assert a.dtype == b.dtype
            assert a.size == self.size * b.size
            assert a.flags.contiguous
        assert b.flags.contiguous
        assert 0 <= root < self.size
        self.comm.scatter(a, b, root)

    def all_gather(self, a, b):
        """Gather data from all ranks onto all processes in a group.

        Parameters:

        a: ndarray
            Source of the data to gather, i.e. send buffer of this rank.
        b: ndarray
            Destination of the distributed data, i.e. receive buffer.
            The size of this array must match the size of the distributed
            source arrays multiplied by the number of process in the group.

        Example::

          # All ranks have parts of interesting data. Gather on all ranks.
          mydata = np.random.normal(size=N)
          data = np.empty(N*comm.size, dtype=float)
          comm.all_gather(mydata, data)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Insert my part directly
              data[0:N] = mydata
              # Gather parts from the slaves
              buf = np.empty(N, dtype=float)
              for rank in range(1, comm.size):
                  comm.receive(buf, rank, tag=123)
                  data[rank*N:(rank+1)*N] = buf
          else:
              # Send to the master
              comm.send(mydata, 0, tag=123)
          # Broadcast from master to all slaves
          comm.broadcast(data, 0)

        """
        tc = a.dtype
        assert a.flags.contiguous
        assert b.flags.contiguous
        assert b.dtype == a.dtype
        assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                a.size * self.size == b.size)
        self.comm.all_gather(a, b)

    def gather(self, a, root, b=None):
        """Gather data from all ranks onto a single process in a group.

        Parameters:

        a: ndarray
            Source of the data to gather, i.e. send buffer of this rank.
        root: int
            Rank of the root process, on which the data is to be gathered.
        b: ndarray (ignored on all ranks different from root; default None)
            Destination of the distributed data, i.e. root's receive buffer.
            The size of this array must match the size of the distributed
            source arrays multiplied by the number of process in the group.

        The reverse operation is ``scatter``.

        Example::

          # All ranks have parts of interesting data. Gather it on master.
          mydata = np.random.normal(size=N)
          if comm.rank == 0:
              data = np.empty(N*comm.size, dtype=float)
          else:
              data = None
          comm.gather(mydata, 0, data)

          # .. which is equivalent to ..

          if comm.rank == 0:
              # Extract my part directly
              data[0:N] = mydata
              # Gather parts from the slaves
              buf = np.empty(N, dtype=float)
              for rank in range(1, comm.size):
                  comm.receive(buf, rank, tag=123)
                  data[rank*N:(rank+1)*N] = buf
          else:
              # Send to the master
              comm.send(mydata, 0, tag=123)

        """
        assert a.flags.contiguous
        assert 0 <= root < self.size
        if root == self.rank:
            assert b.flags.contiguous and b.dtype == a.dtype
            assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                    a.size * self.size == b.size)
            self.comm.gather(a, root, b)
        else:
            assert b is None
            self.comm.gather(a, root)

    def broadcast(self, a, root):
        """Share data from a single process to all ranks in a group.

        Parameters:

        a: ndarray
            Data, i.e. send buffer on root rank, receive buffer elsewhere.
            Note that after the broadcast, all ranks have the same data.
        root: int
            Rank of the root process, from which the data is to be shared.

        Example::

          # All ranks have parts of interesting data. Take a given index.
          mydata[:] = np.random.normal(size=N)

          # Who has the element at global index 13? Everybody needs it!
          index = 13
          root, myindex = divmod(index, N)
          element = np.empty(1, dtype=float)
          if comm.rank == root:
              # This process has the requested element so extract it
              element[:] = mydata[myindex]

          # Broadcast from owner to everyone else
          comm.broadcast(element, root)

          # .. which is equivalent to ..

          if comm.rank == root:
              # We are root so send it to the other ranks
              for rank in range(comm.size):
                  if rank != root:
                      comm.send(element, rank, tag=123)
          else:
              # We don't have it so receive from root
              comm.receive(element, root, tag=123)

        """
        assert 0 <= root < self.size
        assert is_contiguous(a)
        self.comm.broadcast(a, root)

    def sendreceive(self, a, dest, b, src, sendtag=123, recvtag=123):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        assert 0 <= src < self.size
        assert src != self.rank
        assert is_contiguous(b)
        return self.comm.sendreceive(a, dest, b, src, sendtag, recvtag)

    def send(self, a, dest, tag=123, block=True):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        if not block:
            pass #assert sys.getrefcount(a) > 3
        return self.comm.send(a, dest, tag, block)

    def ssend(self, a, dest, tag=123):
        assert 0 <= dest < self.size
        assert dest != self.rank
        assert is_contiguous(a)
        return self.comm.ssend(a, dest, tag)

    def receive(self, a, src, tag=123, block=True):
        assert 0 <= src < self.size
        assert src != self.rank
        assert is_contiguous(a)
        return self.comm.receive(a, src, tag, block)

    def test(self, request):
        """Test whether a non-blocking MPI operation has completed. A boolean
        is returned immediately and the request is not modified in any way.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        return self.comm.test(request)

    def testall(self, requests):
        """Test whether non-blocking MPI operations have completed. A boolean
        is returned immediately but requests may have been deallocated as a
        result, provided they have completed before or during this invokation.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        return self.comm.testall(requests) # may deallocate requests!

    def wait(self, request):
        """Wait for a non-blocking MPI operation to complete before returning.

        Parameters:

        request: MPI request
            Request e.g. returned from send/receive when block=False is used.

        """
        self.comm.wait(request)

    def waitall(self, requests):
        """Wait for non-blocking MPI operations to complete before returning.

        Parameters:

        requests: list
            List of MPI requests e.g. aggregated from returned requests of
            multiple send/receive calls where block=False was used.

        """
        self.comm.waitall(requests)

    def abort(self, errcode):
        """Terminate MPI execution environment of all tasks in the group.
        This function only returns in the advent of an error occurring.

        Parameters:

        errcode: int
            Error code to return to the invoking environment.

        """
        return self.comm.abort(errcode)

    def name(self):
        """Return the name of the processor as a string."""
        return self.comm.name()

    def barrier(self):
        """Block execution until all process have reached this point."""
        self.comm.barrier()

    def diagonalize(self, a, w,
                    nprow=1, npcol=1, mb=32, root=0,
                    b=None):
        if b is None:
            return self.comm.diagonalize(a, w, nprow, npcol, mb, root)
        else:
            return self.comm.diagonalize(a, w, nprow, npcol, mb, root, b)

    def inverse_cholesky(self, a,
                         nprow=1, npcol=1, mb=32, root=0):
        return self.comm.inverse_cholesky(a, nprow, npcol, mb, root)

    def get_members(self):
        """Return the subset of processes which are members of this MPI group
        in terms of the ranks they are assigned on the parent communicator.
        For the world communicator, this is all integers up to ``size``.

        Example::

          >>> world.rank, world.size
          (3, 4)
          >>> world.get_members()
          array([0, 1, 2, 3])
          >>> comm = world.new_communicator(array([2, 3]))
          >>> comm.rank, comm.size
          (1, 2)
          >>> comm.get_members()
          array([2, 3])
          >>> comm.get_members()[comm.rank] == world.rank
          True

        """
        return self.comm.get_members()

    def get_c_object(self):
        """Return the C-object wrapped by this debug interface.

        Whenever a communicator object is passed to C code, that object
        must be a proper C-object - *not* e.g. this debug wrapper.  For
        this reason.  The C-communicator object has a get_c_object()
        implementation which returns itself; thus, always call
        comm.get_c_object() and pass the resulting object to the C code.
        """
        c_obj = self.comm.get_c_object()
        assert type(c_obj) is _gpaw.Communicator
        return c_obj


# Serial communicator
class SerialCommunicator:
    size = 1
    rank = 0

    def __init__(self, parent=None):
        self.parent = parent

    def sum(self, array, root=-1):
        if isinstance(array, (int, float, complex)):
            return array

    def scatter(self, s, r, root):
        r[:] = s

    def max(self, value, root=-1):
        return value

    def broadcast(self, buf, root):
        pass

    def send(self, buff, root, tag=123, block=True):
        pass

    def barrier(self):
        pass

    def gather(self, a, root, b):
        b[:] = a

    def all_gather(self, a, b):
        b[:] = a

    def new_communicator(self, ranks):
        if self.rank not in ranks:
            return None
        return SerialCommunicator(parent=self)

    def test(self, request):
        return 1

    def testall(self, requests):
        return 1

    def wait(self, request):
        raise NotImplementedError('Calls to mpi wait should not happen in '
                                  'serial mode')

    def waitall(self, request):
        raise NotImplementedError('Calls to mpi waitall should not happen in '
                                  'serial mode')

    def get_members(self):
        return np.array([0])

    def get_c_object(self):
        raise NotImplementedError('Should not get C-object for serial comm')


serial_comm = SerialCommunicator()

try:
    world = _gpaw.Communicator()
except AttributeError:
    world = serial_comm

class DryRunCommunicator(SerialCommunicator):
    def __init__(self, size=1, parent=None):
        self.size = size
        self.parent = parent
    
    def new_communicator(self, ranks):
        return DryRunCommunicator(len(ranks), parent=self)

    def get_c_object(self):
        return None # won't actually be passed to C

if dry_run_size > 1:
    world = DryRunCommunicator(dry_run_size)


if debug:
    serial_comm = _Communicator(serial_comm)
    world = _Communicator(world)


size = world.size
rank = world.rank
parallel = (size > 1)


def distribute_cpus(parsize, parsize_bands, nspins, nibzkpts, comm=world):
    """Distribute k-points/spins to processors.

    Construct communicators for parallelization over
    k-points/spins and for parallelization using domain
    decomposition."""

    size = comm.size
    rank = comm.rank

    ntot = nspins * nibzkpts * parsize_bands
    if parsize is None:
        ndomains = size // gcd(ntot, size)
    elif type(parsize) is int:
        ndomains = parsize
    else:
        parsize_c = parsize
        ndomains = parsize_c[0] * parsize_c[1] * parsize_c[2]

    r0 = (rank // ndomains) * ndomains
    ranks = np.arange(r0, r0 + ndomains)
    domain_comm = comm.new_communicator(ranks)

    r0 = rank % (ndomains * parsize_bands)
    ranks = np.arange(r0, r0 + size, ndomains * parsize_bands)
    kpt_comm = comm.new_communicator(ranks)

    r0 = rank % ndomains + kpt_comm.rank * (ndomains * parsize_bands)
    ranks = np.arange(r0, r0 + (ndomains * parsize_bands), ndomains)
    band_comm = comm.new_communicator(ranks)

    assert size == domain_comm.size * kpt_comm.size * band_comm.size
    assert nspins * nibzkpts % kpt_comm.size == 0

    return domain_comm, kpt_comm, band_comm


def compare_atoms(atoms, comm=world):
    """Check whether atoms objects are identical on all processors."""
    # Construct fingerprint:
    fingerprint = np.array([md5_array(array, numeric=True) for array in
                             [atoms.positions,
                              atoms.cell,
                              atoms.pbc * 1.0,
                              atoms.get_initial_magnetic_moments()]])
    # Compare fingerprints:
    fingerprints = np.empty((comm.size, 4), fingerprint.dtype)
    comm.all_gather(fingerprint, fingerprints)
    mismatches = fingerprints.ptp(0)

    if debug:
        dumpfile = 'compare_atoms'
        for i in np.argwhere(mismatches).ravel():
            itemname = ['positions', 'cell', 'pbc', 'magmoms'][i]
            itemfps = fingerprints[:, i]
            itemdata = [atoms.positions,
                        atoms.cell,
                        atoms.pbc * 1.0,
                        atoms.get_initial_magnetic_moments()][i]
            if comm.rank == 0:
                print 'DEBUG: compare_atoms failed for %s' % itemname
                itemfps.dump('%s_fps_%s.pickle' % (dumpfile, itemname))
            itemdata.dump('%s_r%04d_%s.pickle' % (dumpfile, comm.rank, 
                                                  itemname))

    return not mismatches.any()

def broadcast(obj, root=0, comm=world):
    """Broadcast a Python object across an MPI communicator and return it."""
    if comm.rank == root:
        assert obj is not None
        string = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    else:
        assert obj is None
        string = None
    string = broadcast_string(string, root, comm)
    if comm.rank == root:
        return obj
    else:
        return pickle.loads(string)

def broadcast_string(string=None, root=0, comm=world):
    """Broadcast a Python string across an MPI communicator and return it.
    NB: Strings are immutable objects in Python, so the input is unchanged."""
    if comm.rank == root:
        assert isinstance(string, str)
        n = np.array(len(string), int)
    else:
        assert string is None
        n = np.zeros(1, int)
    comm.broadcast(n, root)
    if comm.rank == root:
        string = np.fromstring(string, np.int8)
    else:
        string = np.zeros(n, np.int8)
    comm.broadcast(string, root)
    return string.tostring()

def send_string(string, rank, comm=world):
    comm.send(np.array(len(string)), rank)
    comm.send(np.fromstring(string, np.int8), rank)

def receive_string(rank, comm=world):
    n = np.array(0)
    comm.receive(n, rank)
    string = np.empty(n, np.int8)
    comm.receive(string, rank)
    return string.tostring()

def ibarrier(timeout=None, root=0, tag=123, comm=world):
    """Non-blocking barrier returning a list of requests to wait for.
    An optional time-out may be given, turning the call into a blocking
    barrier with an upper time limit, beyond which an exception is raised."""
    requests = []
    byte = np.ones(1, dtype=np.int8)
    if comm.rank == root:
        for rank in range(0,root) + range(root+1,comm.size): #everybody else
            rbuf, sbuf = np.empty_like(byte), byte.copy()
            requests.append(comm.send(sbuf, rank, tag=2 * tag + 0, 
                                      block=False))
            requests.append(comm.receive(rbuf, rank, tag=2 * tag + 1,
                                         block=False))
    else:
        rbuf, sbuf = np.empty_like(byte), byte
        requests.append(comm.receive(rbuf, root, tag=2 * tag + 0, block=False))
        requests.append(comm.send(sbuf, root, tag=2 * tag + 1, block=False))

    if comm.size == 1 or timeout is None:
        return requests

    t0 = time.time()
    while not comm.testall(requests): # automatic clean-up upon success
        if time.time() - t0 > timeout:
            raise RuntimeError('MPI barrier timeout.')
    return []

def run(iterators):
    """Run through list of iterators one step at a time."""
    if not isinstance(iterators, list):
        # It's a single iterator - empty it:
        for i in iterators:
            pass
        return

    if len(iterators) == 0:
        return

    while True:
        try:
            results = [iter.next() for iter in iterators]
        except StopIteration:
            return results

# Shut down all processes if one of them fails.
if parallel and not (dry_run_size > 1):
    # This is a true parallel calculation
    def cleanup(sys=sys, time=time, world=world):
        error = getattr(sys, 'last_type', None)
        if error:
            sys.stdout.flush()
            sys.stderr.write(('GPAW CLEANUP (node %d): %s occurred.  ' +
                              'Calling MPI_Abort!\n') % (world.rank, error))
            sys.stderr.flush()
            # Give other nodes a moment to crash by themselves (perhaps
            # producing helpful error messages)
            time.sleep(3)
            world.abort(42)

    atexit.register(cleanup)
    
