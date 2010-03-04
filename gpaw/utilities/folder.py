import numpy as np

from gpaw.gauss import Gauss, Lorentz

class Folder:
    """Fold a function with normalised Gaussians or Lorentzians"""
    def __init__(self, width,
                 folding='Gauss'):
        self.width = width
        if folding == 'Gauss':
            self.func = Gauss(width)
        elif folding == 'Lorentz':
            self.func = Lorentz(width)
        elif folding == None:
            self.func = None
        else:
            raise RuntimeError('unknown folding "' + folding + '"')

    def fold(self, x, y, dx=None, min=None, max=None):
        X = np.array(x)
        assert len(X.shape) == 1
        Y = np.array(y)
        assert X.shape[0] == Y.shape[0]

        if self.func is None:
            xl = X
            yl = Y
        else:
            if min is None:
                min = np.min(X) - 4 * self.width
            if max is None:
                max = np.max(X) + 4 * self.width
            if dx is None:
                dx = self.width / 4.

            xl = np.arange(min, max + 0.5 * dx, dx)
                
            # weight matrix
            weightm = np.empty((xl.shape[0], X.shape[0]))
            for i, x in enumerate(X):
                weightm[:, i] = self.func.get(xl - x)

            yl = np.dot(weightm, Y)
            
        return xl, yl

