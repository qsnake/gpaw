# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Atomic-centered localized functions.
"""

from math import pi, cos, sin

import Numeric as num

from gridpaw import enumerate
from gridpaw import debug
from gridpaw.utilities import contiguous, is_contiguous
import _gridpaw


MASTER = 0


def create_localized_functions(functions, gd, spos_i, onohirose=5,
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

    lfs = LocFuncs(functions, gd, spos_i, onohirose,
                    typecode, cut, forces, lfbc)

    if len(lfs.box_b) > 0:
        return lfs
    else:
        # No boxes in this domain:
        return None


class LocFuncs:
    """Class to handle atomic-centered localized functions."""
    def __init__(self, functions, gd, spos_i, onohirose,
                 typecode, cut, forces, lfbc):
        """Create `LocFuncs` object.

        Use `creat_localized_functions()` to create this object."""
        
        angle = gd.domain.angle

        # We assume that all functions have the same cut-off:
        rcut = functions[0].get_cutoff()

        p = onohirose
        assert p > 0

        k = 6
        if p != 1:
            rcut += (k / 2 - 1.0 / p) * max(gd.h_i)

        box_b = gd.get_boxes(spos_i, rcut, cut)

        self.box_b = []
        self.disp_bi = num.zeros((len(box_b), 3), num.Float)
        b = 0
        for beg_i, end_i, disp_i in box_b:
            if angle is None:
                rspos_i = spos_i
            else:
                da = angle * disp_i[0]
                tspos_i = spos_i - 0.5
                rspos_i = num.array(
                    [tspos_i[0],
                     tspos_i[1] * cos(da) - tspos_i[2] * sin(da),
                     tspos_i[1] * sin(da) + tspos_i[2] * cos(da)]) + 0.5
                                      
            box = LocalizedFunctions(functions, beg_i, end_i,
                                     rspos_i, disp_i, gd,
                                     p, k, typecode, forces, lfbc)
            self.box_b.append(box)
            self.disp_bi[b] = disp_i
            b += 1
        
        self.ni = 0
        self.niD = 0
        for radial in functions:
            l = radial.get_angular_momentum_number()
            self.ni += 2 * l + 1; 
            self.niD += 3 + l * (1 + 2 * l)
        self.typecode = typecode

        self.set_communicator(gd.comm, MASTER)

        self.phases = {}

    def set_communicator(self, comm, root):
        """Set MPI-communicator and master CPU."""
        self.comm = comm
        self.root = root
        
    def add(self, a_xg, coef_xi, k_i=None, communicate=False):
        """Add localized functions to extended arrays.

        Add the product of ``coef_xi`` and the localized functions to
        ``a_xg``.  With Block boundary-condtions, ``k_i`` is used for
        the phase-factors.  If ``communicate`` is false, ``coef_xi``
        will be broadcasted from the root-CPU."""
        
        if communicate:
            if coef_xi is None:
                shape = grids.shape[:-3] + (self.ni,)
                coef_xi = num.zeros(shape, self.typecode)
            self.comm.broadcast(coef_xi, self.root)
            
        if k_i is None:
            for box in self.box_b:
                box.add(coef_xi, a_xg)
        else:
            if self.phases.has_key(id(k_i)):
                phase_b = self.phases[id(k_i)]
            else:
                phase_b = num.exp(-2j * pi * num.dot(self.disp_bi, k_i))
                self.phases[id(k_i)] = phase_b
                
            for box, phase in zip(self.box_b, phase_b):
                box.add(coef_xi * phase, a_xg)

    def integrate(self, a_xg, result_xi, k_i=None, derivatives=False):
        """Calculate integrals of arrays times localized functions.

        Return the interal of extended arrays times localized
        functions in ``result_xi``.  Correct phase-factors are used if
        the scaled **k**-point ``k_i`` is not ``None`` (Block
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
            
        if k_i is None:
            for box in self.box_b:
                box.multiply(a_xg, tmp_xi, derivatives)
                result_xi += tmp_xi
        else:
            if self.phases.has_key(id(k_i)):
                phase_b = self.phases[id(k_i)]
            else:
                phase_b = num.exp(-2j * pi * num.dot(self.disp_bi, k_i))
                self.phases[id(k_i)] = phase_b
            for box, phase in zip(self.box_b, phase_b):
                box.multiply(a_xg, tmp_xi, derivatives)
                result_xi += phase * tmp_xi

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
    
    def __init__(self, functions, beg_i, end_i, spos_i, disp_i, gd,
                 p, k, typecode, forces, locfuncbcaster):
        """Construct a ``LocalizedFunctions`` C-object.

        Evaluate function values from a list of splines
        (``functions``) inside a box between grid points ``beg_i``
        (included) to ``end_i`` (not included).  The functions are
        centered at the scaled position ``spos_i`` displaced by
        ``disp_i`` (in units of lattice vectors), and ``gd`` is the
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
            
        size_i = end_i - beg_i
        corner_i = beg_i - gd.beg0_i
        pos_i = (beg_i - (spos_i - disp_i) * gd.N_i) * gd.h_i

        self.lfs = _gridpaw.LocalizedFunctions(
            [function.spline for function in functions],
            size_i, gd.n_i, corner_i, gd.h_i, pos_i, p, k,
            typecode == num.Float, forces, compute)
        
        if locfuncbcaster is not None:
            locfuncbcaster.add(self.lfs)

        self.ni = 0   # number of functions
        self.niD = 0  # number of derivatives
        for function in functions:
            l = function.get_angular_momentum_number()
            self.ni += 2 * l + 1; 
            self.niD += 3 + l * (1 + 2 * l)

        self.shape = tuple(gd.n_i)
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
            assert results.shape[-1] == self.ni
        self.lfs.multiply(a_xg, result_xi, derivatives)

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
    def LocalizedFunctions(functions, beg_i, end_i, spos_i, disp_i, gd,
                           p, k, typecode, forces, locfuncbcaster):
        return LocalizedFunctionsWrapper(functions, beg_i, end_i, spos_i,
                                         disp_i, gd, p, k,
                                         typecode, forces, locfuncbcaster).lfs


class LocFuncBroadcaster:
    """"""
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


