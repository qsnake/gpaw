# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import sqrt

import Numeric as num

from gridpaw import _gridpaw


erf = _gridpaw.erf
#XXXfrom libgridpaw import erf

def contiguous(array, type):
    """Convert a sequence to a contiguous Numeric array."""
    array = num.asarray(array, type)
    if array.iscontiguous():
        return array
    else:
        return num.array(array)

def is_contiguous(array, type=None):
    """Check for contiguity and type."""
    if type is None:
        return array.iscontiguous()
    else:
        return array.iscontiguous() and array.typecode() == type


# These should use BLAS:
def scale_add_to(dwf, s, wf):
    wf += s * dwf
    
def square_scale_add_to(wf, s, rho):
    rho.flat[:] += s * wf.flat**2


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
    c = cell.copy()
    # Zero the diagonal:
    c.flat[::4] = 0.0
    if num.sometrue(c.flat):
        raise RuntimeError, 'Unit cell not orthorhombic'
    

class DownTheDrain:
    def write(self, string):
        pass
    
    def flush(self):
        pass


def run_threaded(tasks):
    try:
        while True:
            for task in tasks:
                task.next()
    except StopIteration:
        pass


def warning(msg):
    r"""Warning(msg) -> msg in a box.

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

