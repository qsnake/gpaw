# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import os
import sys

import Numeric as num

from gpaw import debug, parallel
from gpaw.utilities import is_contiguous
import _gpaw


MASTER = 0

# Serial communicator
class SerialCommunicator:
    size = 1
    rank = 0
    def sum(self, array, root=-1):
        if isinstance(array, (float, complex)):
            return array
        
    def broadcast(self, buf, root):
        pass

    def send(self, buff, root, tag=123, block=True):
        pass

    def barrier(self):
        pass
    
serial_comm = SerialCommunicator()
if debug:
    serial_comm.comm = serial_comm # cycle? XXX

try:
    world = _gpaw.Communicator()
except:
    world = serial_comm

size = world.size
rank = world.rank
parallel = (size > 1)

if parallel and debug:
    class _Communicator:
        def __init__(self, comm):
            self.comm = comm
            self.size = comm.size
            self.rank = comm.rank

        def new_communicator(self, ranks):
            assert is_contiguous(ranks, num.Int)
            sranks = num.sort(ranks)
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
                assert isinstance(array, float)
                return self.comm.sum(array, root)
            else:
                tc = array.typecode()
                assert tc == num.Float or tc == num.Complex
                assert is_contiguous(array, tc)
                assert root == -1 or 0 <= root < self.size
                self.comm.sum(array, root)

        def all_gather(self, a, b):
            tc = a.typecode()
            assert is_contiguous(a, tc)
            assert is_contiguous(b, tc)
            assert b.shape[0] == self.size
            assert a.shape == b.shape[1:]
            self.comm.all_gather(a, b)

        def gather(self, a, root, b=None):
            tc = a.typecode()
            assert is_contiguous(a, tc)
            assert 0 <= root < self.size
            if root == self.rank:
                assert is_contiguous(b, tc)
                assert b.shape[0] == self.size
                assert a.shape == b.shape[1:]
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
            
        def receive(self, a, src, tag=123, block=True):
            assert 0 <= src < self.size
            assert src != self.rank
            assert is_contiguous(a)
            return self.comm.receive(a, src, tag, block)
            
        def wait(self, request):
            self.comm.wait(request)

    world = _Communicator(world)


def broadcast_string(string=None, root=MASTER, comm=world):
    if rank == root:
        assert isinstance(string, str)
        n = num.array(len(string), num.Int)
    else:
        assert string is None
        n = num.zeros(1, num.Int)
    comm.broadcast(n, root)
    if rank == root:
        string = num.fromstring(string, num.Int8)
    else:
        string = num.zeros(n, num.Int8)
    comm.broadcast(string, root)
    return string.tostring()


def all_gather_array(comm, a):
    # Gather array into flat array
    shape = (comm.size,) + num.shape(a)
    all = num.zeros(shape, num.Float)
    comm.all_gather(a, all)
    return all.flat
