# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from __future__ import division
from math import pi
import numpy as np

from gpaw import debug
from gpaw.utilities import is_contiguous
import _gpaw


class _Transformer:
    def __init__(self, gdin, gdout, nn=1, dtype=float):
        self.dtype = dtype
        neighbor_cd = gdin.neighbor_cd

        if gdin.comm.size > 1:
            comm = gdin.comm
            if debug:
                # Get the C-communicator from the debug-wrapper:
                comm = comm.comm
        else:
            comm = None

        pad_cd = np.empty((3, 2), int)
        neighborpad_cd = np.empty((3, 2), int)
        skip_cd = np.empty((3, 2), int)

        if (gdin.N_c == 2 * gdout.N_c).all():
            # Restriction:
            pad_cd[:, 0] = 2 * nn - 1 - 2 * gdout.beg_c + gdin.beg_c
            pad_cd[:, 1] = 2 * nn - 2 + 2 * gdout.end_c - gdin.end_c
            neighborpad_cd[:, 0] = 2 * nn - 2 + 2 * gdout.beg_c - gdin.beg_c
            neighborpad_cd[:, 1] = 2 * nn - 1 - 2 * gdout.end_c + gdin.end_c
            self.interpolate = False
        else:
            assert (gdout.N_c == 2 * gdin.N_c).all()
            # Interpolation:
            pad_cd[:, 0] = nn - 1 - gdout.beg_c // 2 + gdin.beg_c
            pad_cd[:, 1] = nn + gdout.end_c // 2 - gdin.end_c
            neighborpad_cd[:, 0] = nn + gdout.beg_c // 2 - gdin.beg_c
            neighborpad_cd[:, 1] = nn - 1 - gdout.end_c // 2 + gdin.end_c
            skip_cd[:, 0] = gdout.beg_c % 2
            skip_cd[:, 1] = gdout.end_c % 2
            self.interpolate = True

        assert np.alltrue(pad_cd.ravel() >= 0)

        self.transformer = _gpaw.Transformer(gdin.n_c, 2 * nn, pad_cd, 
                                             neighborpad_cd, skip_cd,
                                             neighbor_cd, dtype == float, comm,
                                             self.interpolate)
        
        self.ngpin = tuple(gdin.n_c)
        self.ngpout = tuple(gdout.n_c)
        assert dtype in [float, complex]

    def apply(self, input, output, phases=None):
        assert is_contiguous(input, self.dtype)
        assert is_contiguous(output, self.dtype)
        assert input.shape == self.ngpin
        assert output.shape == self.ngpout
        self.transformer.apply(input, output, phases)


def Transformer(gdin, gdout, nn=1, dtype=float):
    if nn != 9:
        t = _Transformer(gdin, gdout, nn, dtype)
        interpolate = t.interpolate
        if not debug:
            t = t.transformer
        return TransformerWrapper(gdin, gdout, t, interpolate)
    class T:
        def apply(self, input, output, phases=None):
            output[:] = input
        def estimate_memory(self, mem):
            mem.set('Nulltransformer', 0)
    return T()

def multiple_transform_apply(transformerlist, inputs, outputs, phases=None):
    return _gpaw.multiple_transform_apply(transformerlist, inputs, outputs, 
                                          phases)


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
            print '%14.16f' % (n / d),
        print


class TransformerWrapper:
    def __init__(self, gdin, gdout, transformer, interpolate):
        self.gdin = gdin
        self.gdout = gdout
        self.transformer = transformer
        self.interpolate = interpolate
        
    def apply(self, input, output, phases=None):
        self.transformer.apply(input, output, phases)

    def estimate_memory(self, mem):
        # Read transformers.c for details
        inbytes = self.gdin.bytecount()
        outbytes = self.gdout.bytecount()
        mem.subnode('buf', inbytes)
        if self.interpolate:
            mem.subnode('buf2 interp', 16 * inbytes)
        else:
            mem.subnode('buf2 restrict', 4 * outbytes)
