# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import division
from math import pi
import Numeric as num

from gpaw import debug
from gpaw.utilities import is_contiguous
import _gpaw


class Transformer:
    def __init__(self, gdin, gdout, nn, typecode):
        self.typecode = typecode
        neighbor_cd = gdin.domain.neighbor_cd

        if gdin.comm.size > 1:
            comm = gdin.comm
            if debug:
                # Get the C-communicator from the debug-wrapper:
                comm = comm.comm
        else:
            comm = None

        pad_cd = num.empty((3, 2), num.Int)
        neighborpad_cd = num.empty((3, 2), num.Int)
        skip_cd = num.empty((3, 2), num.Int)
        
        if gdin.N_c == 2 * gdout.N_c:
            # Restriction:
            pad_cd[:, 0] = 2 * nn - 1 - 2 * gdout.beg_c + gdin.beg_c
            pad_cd[:, 1] = 2 * nn - 2 + 2 * gdout.end_c - gdin.end_c
            neighborpad_cd[:, 0] = 2 * nn - 2 + 2 * gdout.beg_c - gdin.beg_c
            neighborpad_cd[:, 1] = 2 * nn - 1 - 2 * gdout.end_c + gdin.end_c
            interpolate = False
        else:
            assert gdout.N_c == 2 * gdin.N_c
            # Interpolation:
            pad_cd[:, 0] = nn - 1 - gdout.beg_c // 2 + gdin.beg_c
            pad_cd[:, 1] = nn + gdout.end_c // 2 - gdin.end_c
            neighborpad_cd[:, 0] = nn + gdout.beg_c // 2 - gdin.beg_c
            neighborpad_cd[:, 1] = nn - 1 - gdout.end_c // 2 + gdin.end_c
            skip_cd[:, 0] = gdout.beg_c % 2
            skip_cd[:, 1] = gdout.end_c % 2
            interpolate = True

        if 0:
            import mpi
            print mpi.rank,pad_cd
            print mpi.rank, neighborpad_cd
            print mpi.rank,skip_cd
            print mpi.rank, neighbor_cd
            print mpi.rank, gdin.n_c
            print mpi.rank, gdout.n_c
            
        self.transformer = _gpaw.Transformer(
            gdin.n_c, 2 * nn, pad_cd, neighborpad_cd, skip_cd, neighbor_cd,
            typecode == num.Float, comm, interpolate)
        
        self.ngpin = tuple(gdin.n_c)
        self.ngpout = tuple(gdout.n_c)
        assert typecode in [num.Float, num.Complex]

    def apply(self, input, output, phases=None):
        assert is_contiguous(input, self.typecode)
        assert is_contiguous(output, self.typecode)
        assert input.shape == self.ngpin
        assert output.shape == self.ngpout
        self.transformer.apply(input, output, phases)


def Interpolator(gd, nn, typecode=num.Float):
    return Transformer(gd, gd.refine(), nn, typecode)
def Restrictor(gd, nn, typecode=num.Float):
    return Transformer(gd, gd.coarsen(), nn, typecode)


def coefs(k, p):
    for i in range(0, k * p, p):
        print '%2d' % i,
        for x in range((k // 2 - 1) * p, k // 2 * p + 1):
            n = 1
            d = 1
            for j in range(0, k * p, p):
                if j == i:
                    continue
                n *= x - j
                d *= i - j
            print '%14.10f' % (n / d),
        print
