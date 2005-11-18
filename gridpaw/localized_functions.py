# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Atomic-centered localized functions.
"""

from math import pi, cos, sin

import Numeric as num

from gridpaw import debug
from gridpaw.utilities import is_contiguous
from gridpaw.rotation import transrotation
import _gridpaw


MASTER = 0


def create_localized_functions(functions, gd, spos_c, onohirose=5,
                               typecode=num.Float, cut=False,
                               forces=True, lfbc=None):
    """Create `LocFuncs` object.

    From a list of splines, a grid-descriptor and a scaled position,
    create a `LocFuncs` object.  If this domain does not contribute to
    the localized functions, ``None`` is returned.

    ============= ======================== ===================================
    keyword       type
    ============= ======================== ===================================
    ``onohirose`` ``int``                  Grid point density used for
                                           Ono-Hirose double-grid
                                           technique (5 is default and 1 is
                                           off).
    ``typecode``  ``Float`` or ``Complex`` Type of arrays to operate on.
    ``cut``       ``bool``                 Allow functions to cut boundaries
                                           when not periodic.
    ``forces``    ``bool``                 Calculate derivatives.
    ``lfbc``      `LocFuncBroadcaster`     Parallelization ...
    ============= ======================== ===================================
    """

    lfs = LocFuncs(functions, gd, spos_c, onohirose,
                    typecode, cut, forces, lfbc)

    if len(lfs.box_b) > 0:
        return lfs
    else:
        # No boxes in this domain:
        return None


class LocFuncs:
    """Class to handle atomic-centered localized functions."""
    def __init__(self, functions, gd, spos_c, onohirose,
                 typecode, cut, forces, lfbc):
        """Create `LocFuncs` object.

        Use `create_localized_functions()` to create this object."""
        
        self.angle = gd.domain.angle
        angle = self.angle
        # We assume that all functions have the same cut-off:
        rcut = functions[0].get_cutoff()

        p = onohirose
        assert p > 0

        k = 6
        if p != 1:
            rcut += (k / 2 - 1.0 / p) * max(gd.h_c)

        box_b = gd.get_boxes(spos_c, rcut, cut)

        self.box_b = []
        self.sdisp_bc = num.zeros((len(box_b), 3), num.Float)
        b = 0
        for beg_c, end_c, sdisp_c in box_b:
            if angle is None:
                rspos_c = spos_c
            else:
                da = angle * sdisp_c[0]
                tspos_c = spos_c - 0.5
                rspos_c = num.array(
                    [tspos_c[0],
                     tspos_c[1] * cos(da) - tspos_c[2] * sin(da),
                     tspos_c[1] * sin(da) + tspos_c[2] * cos(da)]) + 0.5
                                      
            box = LocalizedFunctions(functions, beg_c, end_c,
                                     rspos_c, sdisp_c, gd,
                                     p, k, typecode, forces, lfbc)
            self.box_b.append(box)
            self.sdisp_bc[b] = sdisp_c
            b += 1
        
        self.ni = 0
        self.niD = 0
        for radial in functions:
            l = radial.get_angular_momentum_number()
            self.ni += 2 * l + 1; 
            self.niD += 3 + l * (1 + 2 * l)

        if angle is not None:
            nb = len(self.box_b)
            self.R_bii = num.zeros((nb, self.ni, self.ni), num.Float)
            self.R_biiT= num.zeros((nb, self.ni, self.ni), num.Float)
            for b, sdisp_c in enumerate(self.sdisp_bc):
                n1 = 0
                for radial in functions:
                    l = radial.get_angular_momentum_number()
                    n2 = n1 + 2 * l + 1
                    self.R_bii[b, n1:n2, n1:n2]=self.rmatrix(
                        sdisp_c[0] * angle, l)
                    self.R_biiT[b, n1:n2, n1:n2]=self.rmatrix(
                        -sdisp_c[0] * angle, l)
                    n1 = n2
                               
        self.typecode = typecode

        self.set_communicator(gd.comm, MASTER)

        self.phase_kb = None

    def rmatrix(self, da, l):
        c = cos(da)
        s = sin(da)
        if l == 0:
            return num.asarray([1])
        if l == 1:
            r = num.asarray([
                [1, 0, 0],
                [0, c,-s],
                [0, s, c]])
            return r
        if l == 2:
##            r = transrotation(2, -da)
            r = num.transpose(transrotation(2, da))
##             r = num.asarray([
##                 [c,       0, -s,           0,           0],
##                 [0, c*c-s*s,  0,    -0.5*s*c,    -0.5*s*c],
##                 [s,       0,  c,           0,           0],
##                 [0,   2*s*c,  0,   1-0.5*s*s,    -0.5*s*s],
##                 [0,   6*s*c,  0,    -1.5*s*s, c*c-0.5*s*s]
##                 ])
            return r
        
    def set_communicator(self, comm, root):
        """Set MPI-communicator and master CPU."""
        self.comm = comm
        self.root = root

    def set_phase_factors(self, k_kc):
        self.phase_kb = num.exp(2j * pi *
                                num.innerproduct(k_kc, self.sdisp_bc))
        
    def add(self, a_xg, coef_xi, k=None, communicate=False):
        """Add localized functions to extended arrays.

        Add the product of ``coef_xi`` and the localized functions to
        ``a_xg``.  With Block boundary-condtions, ``k`` is used to
        index the phase-factors.  If ``communicate`` is false,
        ``coef_xi`` will be broadcasted from the root-CPU."""
        
        if communicate:
            if coef_xi is None:
                shape = a_xg.shape[:-3] + (self.ni,)
                coef_xi = num.zeros(shape, self.typecode)
            self.comm.broadcast(coef_xi, self.root)

        if (k is None or self.phase_kb is None) and self.angle is None:
            # No k-points, no rotation
            for box in self.box_b:
                box.add(coef_xi, a_xg)
        elif self.angle is None:
            # K-points, no rotation
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.add(coef_xi / phase, a_xg)
        elif (k is None or self.phase_kb is None) and self.angle is not None:
            # Rotation, but no k-points
            for box, R_ii in zip(self.box_b, self.R_biiT):
                box.add(num.dot(coef_xi, R_ii), a_xg) 
        else:
            # Rotation and k-points
            for box, phase, R_ii in zip(self.box_b, self.phase_kb[k], self.R_biiT):
                box.add(num.dot(coef_xi, R_ii) / phase, a_xg)
                
    def integrate(self, a_xg, result_xi, k=None, derivatives=False):
        """Calculate integrals of arrays times localized functions.

        Return the integral of extended arrays times localized
        functions in ``result_xi``.  Correct phase-factors are used if
        the **k**-point index ``k`` is not ``None`` (Block
        boundary-condtions).  If ``derivatives`` is true (defaults to
        false), the *x*- *y*- and *z*-derivatives are calculated
        instead."""
        
        if derivatives:
            shape = a_xg.shape[:-3] + (self.niD,)
        else:
            shape = a_xg.shape[:-3] + (self.ni,)
            
        tmp_xi = num.zeros(shape, self.typecode)
        if result_xi is None:
            result_xi = num.zeros(shape, self.typecode)
            
        if (k is None or self.phase_kb is None) and self.angle is None:
            # No k-points, no rotation
            for box in self.box_b:
                box.integrate(a_xg, tmp_xi, derivatives)
                result_xi += tmp_xi                
        elif self.angle is None:
            # K-points, no rotation
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.integrate(a_xg, tmp_xi, derivatives)
                result_xi += phase * tmp_xi
        elif (k is None or self.phase_kb is None) and self.angle is not None:
            # Rotation, no k-points
            for box, R_ii in zip(self.box_b, self.R_bii):
                box.integrate(a_xg, tmp_xi, derivatives)
                result_xi += num.dot(tmp_xi, R_ii)
        else:
            # Rotation and k-points
            for box, phase, R_ii in zip(self.box_b, self.phase_kb[k], self.R_bii):
                box.integrate(a_xg, tmp_xi, derivatives)
                result_xi += phase * num.dot(tmp_xi, R_ii)
               
        self.comm.sum(result_xi, self.root)

    def add_density(self, n_G, f_i):
        """Add atomic electron density to extended density array.

        Special method for adding the atomic electron density
        calculated from atomic orbitals and occupation numbers
        ``f_i``."""
        for box in self.box_b:
            box.add_density(n_G, f_i)


class LocalizedFunctionsWrapper:
    """Python wrapper class for C-extension: ``LocalizedFunctions``.

    This class is used for construction of the C-object and for adding
    type-checking to the C-methods."""
    
    def __init__(self, functions, beg_c, end_c, spos_c, sdisp_c, gd,
                 p, k, typecode, forces, locfuncbcaster):
        """Construct a ``LocalizedFunctions`` C-object.

        Evaluate function values from a list of splines
        (``functions``) inside a box between grid points ``beg_c``
        (included) to ``end_c`` (not included).  The functions are
        centered at the scaled position ``spos_c`` displaced by
        ``sdisp_c`` (in units of lattice vectors), and ``gd`` is the
        grid-descriptor.

        ``p`` is the number of grid points used for the Ono-Hirose
        double-grid technique and ``k`` is the order of the
        interpolation.  Derivatives are calculated with
        ``forces=True``."""

        assert typecode in [num.Float, num.Complex]

        # Who evaluates the function values?
        if locfuncbcaster is None:
            # I do!
            compute = True
        else:
            # One of the CPU's in the k-point communicator does it,
            # and will later broadcast to the others:
            compute = locfuncbcaster.next()
            
        size_c = end_c - beg_c
        corner_c = beg_c - gd.beg0_c
        pos_c = (beg_c - (spos_c - sdisp_c) * gd.N_c) * gd.h_c

        self.lfs = _gridpaw.LocalizedFunctions(
            [function.spline for function in functions],
            size_c, gd.n_c, corner_c, gd.h_c, pos_c, p, k,
            typecode == num.Float, forces, compute)
        
        if locfuncbcaster is not None:
            locfuncbcaster.add(self.lfs)

        self.ni = 0   # number of functions
        self.niD = 0  # number of derivatives
        for function in functions:
            l = function.get_angular_momentum_number()
            self.ni += 2 * l + 1; 
            self.niD += 3 + l * (1 + 2 * l)

        self.shape = tuple(gd.n_c)
        self.typecode = typecode

    def integrate(self, a_xg, result_xi, derivatives=False):
        """Calculate integrals of arrays times localized functions.

        Return the interal of extended arrays times localized
        functions in ``result_xi``.  If ``derivatives`` is true
        (defaults to false), the *x*- *y*- and *z*-derivatives are
        calculated instead."""
        
        assert is_contiguous(a_xg, self.typecode)
        assert is_contiguous(result_xi, self.typecode)
        assert a_xg.shape[:-3] == result_xi.shape[:-1]
        assert a_xg.shape[-3:] == self.shape
        if derivatives:
            assert result_xi.shape[-1] == self.niD
        else:
            assert result_xi.shape[-1] == self.ni
        self.lfs.integrate(a_xg, result_xi, derivatives)

    def add(self, coef_xi, a_xg):
        """Add localized functions to extended arrays.

        Add the product of ``coef_xi`` and the localized functions to
        ``a_xg``."""
        
        assert is_contiguous(a_xg, self.typecode)
        assert is_contiguous(coef_xi, self.typecode)
        assert a_xg.shape[:-3] == coef_xi.shape[:-1]
        assert a_xg.shape[-3:] == self.shape
        assert coef_xi.shape[-1] == self.ni
        self.lfs.add(coef_xi, a_xg)

    def add_density(self, n_G, f_i):
        """Add atomic electron density to extended density array.

        Special method for adding the atomic electron density
        calculated from atomic orbitals and occupation numbers
        ``f_i``."""
        
        assert is_contiguous(n_G, num.Float)
        assert is_contiguous(f_i, num.Float)
        assert n_G.shape == self.shape
        assert f_i.shape == (self.ni,)
        self.lfs.add_density(n_G, f_i)


if debug:
    # Add type and sanity checks:
    LocalizedFunctions = LocalizedFunctionsWrapper
else:
    # Just use the bare C-object for efficiency:
    def LocalizedFunctions(functions, beg_c, end_c, spos_c, sdisp_c, gd,
                           p, k, typecode, forces, locfuncbcaster):
        return LocalizedFunctionsWrapper(functions, beg_c, end_c, spos_c,
                                         sdisp_c, gd, p, k,
                                         typecode, forces, locfuncbcaster).lfs


class LocFuncBroadcaster:
    """..."""
    def __init__(self, comm):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.reset()

    def reset(self):
        self.lfs = []
        self.root = 0

    def next(self):
        compute = (self.root == self.rank)
        self.root = (self.root + 1) % self.size
        return compute

    def add(self, lf):
        self.lfs.append(lf)
    
    def broadcast(self):
        if self.size > 1:
            for root, lf in enumerate(self.lfs):
                lf.broadcast(self.comm, root % self.size)
        self.reset()


