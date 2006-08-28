# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import division
from math import pi
import gpaw.transrotation as tr
import Numeric as num

from gpaw import debug
from gpaw.utilities import is_contiguous
import _gpaw


class Transformer:
    def __init__(self, gd, number, typecode, interpolate):
        self.typecode = typecode
        neighbor_cd = gd.domain.neighbor_cd

        if gd.comm.size > 1:
            comm = gd.comm
            if debug:
                # Get the C-communicator from the debug-wrapper:
                comm = comm.comm
        else:
            comm = None

        self.transformer = _gpaw.Transformer(
            # Why asarray here? XXX
            num.asarray(gd.n_c, num.Int), number + 1, neighbor_cd,
            typecode == num.Float, comm, interpolate)
        
        if gd.domain.angle is not None:
            angle = gd.domain.angle
            if gd.comm.size > 1:
                raise NotImplementedError
            c1, pval1, pfrom1, pto1, pval2, pfrom2, pto2 = tr.RotationCoef(gd.n_c[1], angle)
            self.transformer.set_rotation(angle, c1, pval1, pfrom1, pto1,
                                          pval2, pfrom2, pto2, 0)
            
        self.ngpin = tuple(gd.n_c)
        assert typecode in [num.Float, num.Complex]

    def apply(self, input, output, phases=None):
        assert is_contiguous(input, self.typecode)
        assert is_contiguous(output, self.typecode)
        assert input.shape == self.ngpin
        assert output.shape == self.ngpout
        self.transformer.apply(input, output, phases)


class _Interpolator(Transformer):
    def __init__(self, gd, order, typecode=num.Float):
        Transformer.__init__(self, gd, order, typecode, interpolate=True)
        self.ngpout = tuple(2 * num.array(self.ngpin))


class _Restrictor(Transformer):
    def __init__(self, gd, order, typ=num.Float):
        Transformer.__init__(self, gd, order, typ, interpolate=False)
        self.ngpout = tuple(num.array(self.ngpin) / 2)


if debug:
    Interpolator = _Interpolator
    Restrictor = _Restrictor
else:
    def Interpolator(gd, order, typecode=num.Float):
        return _Interpolator(gd, order, typecode).transformer
    def Restrictor(gd, order, typecode=num.Float):
        return _Restrictor(gd, order, typecode).transformer    


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
