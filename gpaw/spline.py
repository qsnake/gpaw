# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import numpy as npy

from gpaw.utilities import contiguous, divrl
import _gpaw


class Spline:
    def __init__(self, l, rmax, f_g, r_g=None, beta=None, points=25):
        """Spline(l, rcut, list) -> Spline object

        The integer l gives the angular momentum quantum number and
        the list contains the spline values from r=0 to r=rcut.
        """
        assert 0.0 < rmax

        if beta is None:
            f_g = contiguous(f_g, float)
        else:
            f_g = divrl(f_g, l, r_g)
            r = 1.0 * rmax / points * npy.arange(points + 1)
            ng = len(f_g)
            g = (ng * r / (beta + r) + 0.5).astype(int)
            g = npy.clip(g, 1, ng - 2)
            r1 = npy.take(r_g, g - 1)
            r2 = npy.take(r_g, g)
            r3 = npy.take(r_g, g + 1)
            x1 = (r - r2) * (r - r3) / (r1 - r2) / (r1 - r3)
            x2 = (r - r1) * (r - r3) / (r2 - r1) / (r2 - r3)
            x3 = (r - r1) * (r - r2) / (r3 - r1) / (r3 - r2)
            f1 = npy.take(f_g, g - 1)
            f2 = npy.take(f_g, g)
            f3 = npy.take(f_g, g + 1)
            f_g = f1 * x1 + f2 * x2 + f3 * x3

        f_g[-1] = 0.0
        self.spline = _gpaw.Spline(l, rmax, f_g)

    def get_cutoff(self):
        """Return the radial cutoff."""
        return self.spline.get_cutoff()

    def get_angular_momentum_number(self):
        """Return the angular momentum quantum number."""
        return self.spline.get_angular_momentum_number()

    def get_value_and_derivative(self):
        """Return the value and derivative."""
        return self.spline.get_value_and_derivative()

    def __call__(self, r):
        assert r >= 0.0
        return self.spline(r)
