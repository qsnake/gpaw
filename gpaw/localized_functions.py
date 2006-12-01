# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Atomic-centered localized functions.
"""

from math import pi, cos, sin

import Numeric as num
from multiarray import innerproduct as inner # avoid the dotblas version!

from gpaw import debug
from gpaw.utilities import is_contiguous
import _gpaw


MASTER = 0


def create_localized_functions(functions, gd, spos_c,
                               typecode=num.Float, cut=False,
                               forces=True, lfbc=None):
    """Create `LocFuncs` object.

    From a list of splines, a grid-descriptor and a scaled position,
    create a `LocFuncs` object.  If this domain does not contribute to
    the localized functions, ``None`` is returned.

    ============= ======================== ===================================
    keyword       type
    ============= ======================== ===================================
    ``typecode``  ``Float`` or ``Complex`` Type of arrays to operate on.
    ``cut``       ``bool``                 Allow functions to cut boundaries
                                           when not periodic.
    ``forces``    ``bool``                 Calculate derivatives.
    ``lfbc``      `LocFuncBroadcaster`     Parallelization ...
    ============= ======================== ===================================
    """

    lfs = LocFuncs(functions, gd, spos_c,
                    typecode, cut, forces, lfbc)

    if len(lfs.box_b) > 0:
        return lfs
    else:
        # No boxes in this domain:
        return None


class LocFuncs:
    """Class to handle atomic-centered localized functions."""
    def __init__(self, functions, gd, spos_c,
                 typecode, cut, forces, lfbc):
        """Create `LocFuncs` object.

        Use `create_localized_functions()` to create this object."""
        
        # We assume that all functions have the same cut-off:
        rcut = functions[0].get_cutoff()

        box_b = gd.get_boxes(spos_c, rcut, cut)

        self.box_b = []
        self.sdisp_bc = num.zeros((len(box_b), 3), num.Float)
        b = 0
        for beg_c, end_c, sdisp_c in box_b:
            box = LocalizedFunctions(functions, beg_c, end_c,
                                     spos_c, sdisp_c, gd,
                                     typecode, forces, lfbc)
            self.box_b.append(box)
            self.sdisp_bc[b] = sdisp_c
            b += 1
        
        self.ni = 0
        for radial in functions:
            l = radial.get_angular_momentum_number()
            assert l <= 4, 'C-code only does l <= 4.'
            self.ni += 2 * l + 1

        self.typecode = typecode
        self.set_communicator(gd.comm, MASTER)
        self.phase_kb = None

    def set_communicator(self, comm, root):
        """Set MPI-communicator and master CPU."""
        self.comm = comm
        self.root = root

    def set_phase_factors(self, k_kc):
        self.phase_kb = num.exp(2j * pi * inner(k_kc, self.sdisp_bc))
        
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

        if k is None or self.phase_kb is None:
            # No k-points:
            for box in self.box_b:
                box.add(coef_xi, a_xg)
        else:
            # K-points:
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.add(coef_xi / phase, a_xg)
                                                
    def integrate(self, a_xg, result_xi, k=None):
        """Calculate integrals of arrays times localized functions.

        Return the integral of extended arrays times localized
        functions in ``result_xi``.  Correct phase-factors are used if
        the **k**-point index ``k`` is not ``None`` (Block
        boundary-condtions)."""
        
        shape = a_xg.shape[:-3] + (self.ni,)
        tmp_xi = num.zeros(shape, self.typecode)
        if result_xi is None:
            result_xi = num.zeros(shape, self.typecode)
            
        if k is None or self.phase_kb is None:
            # No k-points:
            for box in self.box_b:
                box.integrate(a_xg, tmp_xi)
                result_xi += tmp_xi                
        else:
            # K-points:
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.integrate(a_xg, tmp_xi)
                result_xi += phase * tmp_xi
               
        self.comm.sum(result_xi, self.root)

    def derivative(self, a_xg, result_xic, k=None):
        """Calculate derivatives of localized integrals.

        Return the *x*- *y*- and *z*-derivatives of the integral of
        extended arrays times localized functions in ``result_xi``.
        Correct phase-factors are used if the **k**-point index ``k``
        is not ``None`` (Block boundary-condtions)."""
        
        shape = a_xg.shape[:-3] + (self.ni, 3)
        tmp_xic = num.zeros(shape, self.typecode)
        if result_xic is None:
            result_xic = num.zeros(shape, self.typecode)
            
        if k is None or self.phase_kb is None:
            # No k-points:
            for box in self.box_b:
                box.derivative(a_xg, tmp_xic)
                result_xic += tmp_xic                
        else:
            # K-points:
            for box, phase in zip(self.box_b, self.phase_kb[k]):
                box.derivative(a_xg, tmp_xic)
                result_xic += phase * tmp_xic
               
        self.comm.sum(result_xic, self.root)

    def add_density(self, n_G, f_i):
        """Add atomic electron density to extended density array.

        Special method for adding the atomic electron density
        calculated from atomic orbitals and occupation numbers
        ``f_i``."""
        for box in self.box_b:
            box.add_density(n_G, f_i)

    def add_density2(self, n_G, D_p):
        """Add atomic electron density to extended density array.
        Special method for adding the atomic electron density
        calculated from atomic orbitals and density matrix
        ``D_p``. Returns integral of correction."""
        I = 0.0
        for box in self.box_b:
            I += box.add_density2(n_G, D_p)
        return I

    def normalize(self, I0):
        """Normalize localized function.
        The integral of the first function is normalized to the value
        ``I0``."""

        I = 0.0
        for box in self.box_b:
            I += box.norm()
        I = self.comm.sum(I)
        for box in self.box_b:
            box.scale(I0 / I)
        
class LocalizedFunctionsWrapper:
    """Python wrapper class for C-extension: ``LocalizedFunctions``.

    This class is used for construction of the C-object and for adding
    type-checking to the C-methods."""
    
    def __init__(self, functions, beg_c, end_c, spos_c, sdisp_c, gd,
                 typecode, forces, locfuncbcaster):
        """Construct a ``LocalizedFunctions`` C-object.

        Evaluate function values from a list of splines
        (``functions``) inside a box between grid points ``beg_c``
        (included) to ``end_c`` (not included).  The functions are
        centered at the scaled position ``spos_c`` displaced by
        ``sdisp_c`` (in units of lattice vectors), and ``gd`` is the
        grid-descriptor.

        Derivatives are calculated when ``forces=True``."""

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
        corner_c = beg_c - gd.beg_c
        pos_c = (beg_c - (spos_c - sdisp_c) * gd.N_c) * gd.h_c

        self.lfs = _gpaw.LocalizedFunctions(
            [function.spline for function in functions],
            size_c, gd.n_c, corner_c, gd.h_c, pos_c,
            typecode == num.Float, forces, compute)
        
        if locfuncbcaster is not None:
            locfuncbcaster.add(self.lfs)

        self.ni = 0   # number of functions
        for function in functions:
            l = function.get_angular_momentum_number()
            self.ni += 2 * l + 1; 

        self.shape = tuple(gd.n_c)
        self.typecode = typecode
        self.forces = forces
        
    def integrate(self, a_xg, result_xi):
        """Calculate integrals of arrays times localized functions.

        Return the integral of extended arrays times localized
        functions in ``result_xi``."""
        
        assert is_contiguous(a_xg, self.typecode)
        assert is_contiguous(result_xi, self.typecode)
        assert a_xg.shape[:-3] == result_xi.shape[:-1]
        assert a_xg.shape[-3:] == self.shape
        assert result_xi.shape[-1] == self.ni
        self.lfs.integrate(a_xg, result_xi)

    def derivative(self, a_xg, result_xic):
        """Calculate x-, y-, z-derivatives of localized integrals.

        Return the *x*- *y*- and *z*-derivatives of the integral of
        extended arrays times localized functions in
        ``result_xic``."""

        assert self.forces
        assert is_contiguous(a_xg, self.typecode)
        assert is_contiguous(result_xic, self.typecode)
        assert a_xg.shape[:-3] == result_xic.shape[:-2]
        assert a_xg.shape[-3:] == self.shape
        assert result_xic.shape[-2:] == (self.ni, 3)
        self.lfs.derivative(a_xg, result_xic)

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

    def add_density2(self, n_G, D_p):
        """Add atomic electron density to extended density array.
        Special method for adding the atomic electron density
        calculated from atomic orbitals and density matrix
        ``D_p``."""
        
        assert is_contiguous(n_G, num.Float)
        assert is_contiguous(D_p, num.Float)
        assert n_G.shape == self.shape
        assert D_p.shape == (self.ni * (self.ni + 1) / 2,)
        return self.lfs.add_density2(n_G, D_p)

    def norm(self):
        """Integral of the first function."""
        return self.lfs.norm()

    def scale(self, s):
        """Scale the first function."""
        self.lfs.scale(s)

if debug:
    # Add type and sanity checks:
    LocalizedFunctions = LocalizedFunctionsWrapper
else:
    # Just use the bare C-object for efficiency:
    def LocalizedFunctions(functions, beg_c, end_c, spos_c, sdisp_c, gd,
                           typecode, forces, locfuncbcaster):
        return LocalizedFunctionsWrapper(functions, beg_c, end_c, spos_c,
                                         sdisp_c, gd,
                                         typecode, forces, locfuncbcaster).lfs


class LocFuncBroadcaster:
    """..."""
    def __init__(self, comm):
        if debug:
            comm = comm.comm
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


