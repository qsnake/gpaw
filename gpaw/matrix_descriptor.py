import numpy as np


class MatrixDescriptor:
    """Class representing a 2D matrix shape.  Base class for parallel
    matrix descriptor with BLACS."""
    
    def __init__(self, M, N):
        self.shape = (M, N)
    
    def __nonzero__(self):
        return self.shape[0] != 0 and self.shape[1] != 0

    def zeros(self, n=(), dtype=float):
        """Return array of zeroes with the correct size on all CPUs.

        The last two dimensions will be equal to the shape of this
        descriptor.  If specified as a tuple, can have any preceding
        dimension."""
        return self._new_array(np.zeros, n, dtype)

    def empty(self, n=(), dtype=float):
        """Return array of zeros with the correct size on all CPUs.

        See zeros()."""
        return self._new_array(np.empty, n, dtype)

    def _new_array(self, func, n, dtype):
        if isinstance(n, int):
            n = n,
        shape = n + self.shape
        return func(shape, dtype)

    def check(self, a_mn):
        """Check that specified array is compatible with this descriptor."""
        return a_mn.shape == self.shape and a_mn.flags.contiguous

    def checkassert(self, a_mn):
        ok = self.check(a_mn)
        if not ok:
            if not a_mn.flags.contiguous:
                msg = 'Matrix with shape %s is not contiguous' % (a_mn.shape,)
            else:
                msg = ('%s-descriptor incompatible with %s-matrix' %
                       (self.shape, a_mn.shape))
            raise AssertionError(msg)
