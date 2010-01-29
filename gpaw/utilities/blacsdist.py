#!/usr/bin/env python

import sys
from optparse import OptionParser

from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.mpi import world, SerialCommunicator
from gpaw.utilities import devnull

#if world.rank != 0:
#    sys.stdout = devnull
#    sys.stderr = devnull

def build_parser():
    description = ('Print distribution layout of BLACS matrix.  '
                   'Each printed element will be the rank corresponding to'
                   ' the element in question.')
    usage = 'mpirun -np MNCPU gpaw-python %prog [OPTION] MCPUxNCPU'
    p = OptionParser(usage=usage)
    p.add_option('--matrix', default='32x12', metavar='MxN',
                 help='global matrix shape [%default]')
    p.add_option('--blocksize', default='4x4', metavar='MBxNB',
                 help='block (local matrix) shape [%default]')
    return p

def test(comm, M, N, mcpus, ncpus, mb, nb):
    grid0 = BlacsGrid(comm, 1, 1)
    desc0 = grid0.new_descriptor(M, N, M, N, 0, 0)
    A_mn = desc0.zeros(dtype=float)
    A_mn[:] = comm.size + 1

    grid1 = BlacsGrid(comm, mcpus, ncpus)
    desc1 = grid1.new_descriptor(M, N, mb, nb, 0, 0) # ???
    B_mn = desc1.zeros(dtype=float)
    B_mn[:] = comm.rank

    if comm.rank == 0:
        msg = 'Slices of global matrix indices by rank'
        print msg
        print '-' * len(msg)

    for rank in range(comm.size):
        comm.barrier()
        if rank == comm.rank:
            print 'Rank %d:' % rank
            last_Nstart = 0
            for Mstart, Mstop, Nstart, Nstop, block in desc1.my_blocks(B_mn):
                if Nstart < last_Nstart:
                    print
                print '[%3d:%3d, %3d:%3d]' % (Mstart, Mstop, Nstart, Nstop),
                last_Nstart = Nstart
                assert (block == comm.rank).all()
                #print block
                #print
            print
            print
        comm.barrier()
    
    redistributor = Redistributor(comm, desc1, desc0)
    redistributor.redistribute(B_mn, A_mn)

    if comm.rank == 0:
        msg = 'Rank where each element of the global matrix is stored'
        print msg
        print '-' * len(msg)
        print A_mn

def main():
    parser = build_parser()
    opts, args = parser.parse_args()

    if isinstance(world, SerialCommunicator):
        print >> sys.stderr, ('Please run in parallel using gpaw-python '
                              'or in serial with --help.')
                              
        raise SystemExit

    if len(args) != 1:
        print >> sys.stderr, ('Please provide exactly one argument, e.g. "2x2"'
                              ' or "4x1" if using four CPUs.')
        world.barrier()
        raise SystemExit

    M, N = map(int, opts.matrix.split('x'))
    mcpus, ncpus = map(int, args[0].split('x'))

    blocksize = opts.blocksize
    mb, nb = map(int, blocksize.split('x'))

    if mcpus * ncpus > world.size:
        print >> sys.stderr, ('Requested %d by %d cpus, but commsize is only '
                              '%d' % (mcpus, ncpus, world.size))
        raise SystemExit

    assert M > 0 and N > 0
    assert mcpus > 0 and ncpus > 0
    assert mb > 0 and nb > 0
    test(world, M, N, mcpus, ncpus, mb, nb)

if __name__ == '__main__':
    main()
