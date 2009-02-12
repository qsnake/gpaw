# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys

import numpy as npy

from gpaw import debug
from gpaw import dry_run as dry_run_size
from gpaw.utilities import is_contiguous
from gpaw.utilities import scalapack
import _gpaw


MASTER = 0

# Serial communicator
class SerialCommunicator:
    size = 1
    rank = 0
    def sum(self, array, root=-1):
        if isinstance(array, (float, complex)):
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
        return self

    def cart_create(self, dimx, dimy, dimz, periodic):
        return self

serial_comm = SerialCommunicator()

try:
    world = _gpaw.Communicator()
except:
    world = serial_comm

if dry_run_size > 1:
    class DryRunCommunicator(SerialCommunicator):
        def new_communicator(self, ranks):
            new_comm = DryRunCommunicator()
            new_comm.size = len(ranks)
            return new_comm
    world = DryRunCommunicator()
    world.size = dry_run_size

size = world.size
rank = world.rank
parallel = (size > 1)

if debug:
    class _Communicator:
        def __init__(self, comm):
            self.comm = comm
            self.size = comm.size
            self.rank = comm.rank

        def new_communicator(self, ranks):
            assert is_contiguous(ranks, int)
            sranks = npy.sort(ranks)
            # Are all ranks in range?
            assert 0 <= sranks[0] and sranks[-1] < self.size
            # No duplicates:
            for i in range(len(sranks) - 1):
                assert sranks[i] != sranks[i + 1]
            comm = self.comm.new_communicator(ranks)
            if comm is None:
                # This cpu is not in the new communicator:
                return None
            else:
                return _Communicator(comm)

        def sum(self, array, root=-1):
            if isinstance(array, (float, complex)):
                return self.comm.sum(array, root)
            else:
                tc = array.dtype
                assert tc == float or tc == complex
                assert is_contiguous(array, tc)
                assert root == -1 or 0 <= root < self.size
                self.comm.sum(array, root)

        def scatter(self, s, r, root):
            """Call MPI_Scatter.

            Distribute *s* array from *root* to *r*."""

            assert s.dtype == r.dtype
            assert s.size == self.size * r.size
            assert s.flags.contiguous
            assert r.flags.contiguous
            assert 0 <= root < self.size
            self.comm.scatter(s, r, root)

        def max(self, array, root=-1):
            if isinstance(array, (float, complex)):
                assert isinstance(array, float)
                return self.comm.max(array, root)
            else:
                tc = array.dtype
                assert tc == float or tc == complex
                assert is_contiguous(array, tc)
                assert root == -1 or 0 <= root < self.size
                self.comm.max(array, root)

        def all_gather(self, a, b):
            tc = a.dtype
            assert a.flags.contiguous
            assert b.flags.contiguous
            assert b.dtype == a.dtype
            assert (b.shape[0] == self.size and a.shape == b.shape[1:] or
                    a.size * self.size == b.size)
            self.comm.all_gather(a, b)

        def gather(self, a, root, b=None):
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

        def broadcast(self, buf, root):
            assert 0 <= root < self.size
            assert is_contiguous(buf)
            self.comm.broadcast(buf, root)

        def send(self, a, dest, tag=123, block=True):
            assert 0 <= dest < self.size
            assert dest != self.rank
            assert is_contiguous(a)
            if not block:
                assert sys.getrefcount(a) > 3
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

        def wait(self, request):
            self.comm.wait(request)

        def abort(self, errcode):
            self.comm.abort(errcode)

        def name(self):
            return self.comm.name()

        def barrier(self):
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

        def cart_create(self, dimx, dimy, dimz, periodic):
            return self.comm.cart_create(dimx, dimy, dimz, periodic)


    world = _Communicator(world)
    serial_comm = _Communicator(serial_comm)

def broadcast_string(string=None, root=MASTER, comm=world):
    if rank == root:
        assert isinstance(string, str)
        n = npy.array(len(string), int)
    else:
        assert string is None
        n = npy.zeros(1, int)
    comm.broadcast(n, root)
    if rank == root:
        string = npy.fromstring(string, int8)
    else:
        string = npy.zeros(n, int8)
    comm.broadcast(string, root)
    return string.tostring()


def all_gather_array(comm, a): #???
    # Gather array into flat array
    shape = (comm.size,) + npy.shape(a)
    all = npy.zeros(shape)
    comm.all_gather(a, all)
    return all.ravel()


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
