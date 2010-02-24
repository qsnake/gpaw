#!/usr/bin/env python

import numpy as np

from gpaw.mpi import world
from gpaw.utilities import scalapack
from gpaw.utilities.blacsdist import test

if __name__ in ['__main__', '__builtin__']:
    if not scalapack(True):
        print('Not built with ScaLAPACK. Test does not apply.')
    else:
        M, N = 10, 10
        mb, nb = 2, 2
        mcpus = int(np.ceil(world.size**0.5))
        ncpus = world.size // mcpus

        if world.rank == 0:
            print 'world size:   ', world.size
            print 'M x N:        ', M, 'x', N
            print 'mcpus x ncpus:', mcpus, 'x', ncpus
            print 'mb x nb:      ', mb, 'x', nb
            print

        test(world, M, N, mcpus, ncpus, mb, nb)
