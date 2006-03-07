# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Utility functions and classes."""

from math import sqrt

import Numeric as num

import _gridpaw
from gridpaw import debug


# Error function:
erf = _gridpaw.erf


# Factorials:
fac = [1, 1, 2, 6, 24, 120, 720, 5040, 40320,
       362880, 3628800, 39916800, 479001600]


def contiguous(array, typecode):
    """Convert a sequence to a contiguous Numeric array."""
    array = num.asarray(array, typecode)
    if array.iscontiguous():
        return array
    else:
        return num.array(array)


def is_contiguous(array, typecode=None):
    """Check for contiguity and type."""
    if typecode is None:
        return array.iscontiguous()
    else:
        return array.iscontiguous() and array.typecode() == typecode


# Radial-grid Hartree solver:
if debug:
    def hartree(l, nrdr, a, vr):
        assert is_contiguous(nrdr, num.Float)
        assert is_contiguous(vr, num.Float)
        assert nrdr.shape == vr.shape and len(vr.shape) == 1
        return _gridpaw.hartree(l, nrdr, a, vr)
else:
    hartree = _gridpaw.hartree


def unpack(M):
    assert is_contiguous(M, num.Float)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = num.zeros((n, n), num.Float)
    _gridpaw.unpack(M, M2)
    return M2

    
def unpack2(M):
    assert is_contiguous(M, num.Float)
    n = int(sqrt(0.25 + 2.0 * len(M)))
    M2 = num.zeros((n, n), num.Float)
    _gridpaw.unpack(M, M2)
    M2 *= 0.5
    M2.flat[0::n + 1] *= 2
    return M2

    
def pack(M2):
    n = len(M2)
    M = num.zeros(n * (n + 1) / 2, M2.typecode())
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = 2 * M2[r, c]
            assert abs(M2[r, c] - M2[c, r]) < 1e-10 # ?????
            p += 1
    assert p == len(M)
    return M


def check_unit_cell(cell):
    """Check that the unit cell (3*3 matrix) is orthorhombic (diagonal)."""
    c = cell.copy()
    # Zero the diagonal:
    c.flat[::4] = 0.0
    if num.sometrue(c.flat):
        raise RuntimeError('Unit cell not orthorhombic')
    

class DownTheDrain:
    """Definition of a stream that throws away all output."""
    
    def write(self, string):
        pass
    
    def flush(self):
        pass


"""
class OutputFilter:
    def __init__(self, out, threshold, level=500):
        self.threshold = threshold
        self.verbosity = verbosity

    def write(self, string):
        if kfdce

"""

def run_threaded(tasks):
    """Run list of tasks in small steps.

    Given a list of ``tasks`` (generators), take on step in each and
    repeat that until eash generator is one.  This function is used
    for parallelization by running different tasks in separate
    threads."""

    try:
        while True:
            for task in tasks:
                task.next()
    except StopIteration:
        pass


def warning(msg):
    r"""Put string in a box.

    >>> print Warning('Watch your step!')
     /\/\/\/\/\/\/\/\/\/\/\
     \                    /
     /  WARNING:          \
     \  Watch your step!  /
     /                    \
     \/\/\/\/\/\/\/\/\/\/\/
    """
    
    n = len(msg)
    if n % 2 == 1:
        n += 1
        msg += ' '
    bar = (n / 2 + 3) * '/\\'
    space = (n / 2 + 2) * '  '
    format = ' %s\n \\%s/\n /  WARNING:%s\\\n \\  %s  /\n /%s\\\n %s/'
    return format % (bar, space, space[10:], msg, space, bar[1:])


def center(atoms):
    pos = atoms.GetCartesianPositions()
    cntr = 0.5 * (num.minimum.reduce(pos) + num.maximum.reduce(pos))
    cell = num.diagonal(atoms.GetUnitCell())
    atoms.SetCartesianPositions(pos - cntr + 0.5 * cell)


# Function used by test-suite:
def equal(a, b, e=0):
    assert abs(a - b) <= e, '%f != %f (error: %f)' % (a, b, a - b)

