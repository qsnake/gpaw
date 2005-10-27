# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

from math import pi

import Numeric as num

from gridpaw import enumerate
from gridpaw import debug
from gridpaw.utilities import contiguous, is_contiguous
import gridpaw.utilities.mpi as mpi
import _gridpaw


MASTER = 0

KK = [6]

class _LocalizedFunctions:
    def __init__(self, radials, dims, dimsbig, corner,
                 h, pos, p, k, typecode, forces, locfuncbcaster):
        radials = [radial.spline for radial in radials]
        dims = contiguous(dims, num.Int)
        dimsbig = contiguous(dimsbig, num.Int)
        corner = contiguous(corner, num.Int)
        h = contiguous(h, num.Float)
        pos = contiguous(pos * h, num.Float)
        assert typecode in [num.Float, num.Complex]
        self.ngp = tuple(dimsbig)
        self.nfuncs = 0
        self.nfuncsD = 0
        for radial in radials:
            l = radial.get_angular_momentum_number()
            self.nfuncs += 2 * l + 1; 
            self.nfuncsD += 3 + l * (1 + 2 * l)
        self.typecode = typecode

        if locfuncbcaster is None:
            compute = True
        else:
            compute = locfuncbcaster.next()
            
        self.lfs = _gridpaw.LocalizedFunctions(
            radials, dims,
            dimsbig,
            corner,
            h, pos, p, k,
            typecode == num.Float,
            forces, compute)
        if locfuncbcaster is not None:
            locfuncbcaster.add(self.lfs)

    def multiply(self, arrays, results, derivatives=False):
        """multiply(arrays, results [, derivatives])

        Return the dot-produts of arrays and the localized functions
        in results. if derivatives is true (defaults to false), the
        x- y- and z-derivatives are calculated instead."""
        
        assert is_contiguous(arrays, self.typecode)
        assert is_contiguous(results, self.typecode)
        assert arrays.shape[:-3] == results.shape[:-1]
        assert arrays.shape[-3:] == self.ngp
        if derivatives:
            assert results.shape[-1] == self.nfuncsD
        else:
            assert results.shape[-1] == self.nfuncs
        self.lfs.multiply(arrays, results, derivatives)

    def add(self, coefs, arrays):
        """add(coefs, arrays)

        Add the product of coefs and the localized functions to
        arrays."""
        
        assert is_contiguous(arrays, self.typecode)
        assert is_contiguous(coefs, self.typecode)
        assert arrays.shape[:-3] == coefs.shape[:-1]
        assert arrays.shape[-3:] == self.ngp
        assert coefs.shape[-1] == self.nfuncs
        self.lfs.add(coefs, arrays)

    def add_density(self, n_G, f_i):
        """XXX"""
        
        assert is_contiguous(n_G, num.Float)
        assert is_contiguous(f_i, num.Float)
        assert n_G.shape == self.ngp
        assert f_i.shape == (self.nfuncs,)
        self.lfs.add_density(n_G, f_i)


if debug:
    LocalizedFunctions = _LocalizedFunctions
else:
    def LocalizedFunctions(radials, dims, dimsbig, corner,
                           h, pos, ninter, k, typecode, forces, lfbc):
        return _LocalizedFunctions(radials, dims, dimsbig, corner,
                                   h, pos, ninter, k, typecode,
                                   forces, lfbc).lfs


def LocFuncs(functions, gd, spos, npts=5, typecode=num.Float, cut=False,
             forces=True, lfbc=None):
    lf = _LocFuncs(functions, gd, spos, npts, typecode, cut, forces, lfbc)
    if len(lf.boxes) > 0:
        return lf
    else:
        return None


class _LocFuncs:
    def __init__(self, functions, gd, spos, npts, typecode, cut, forces, lfbc):
        
        # We assume that all functions have the same cut-off:
        rcut = functions[0].get_cutoff()
        if npts==0:
            npts=1#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        p = npts
        k = KK[0]
        if npts != 1:
            rcut += (k / 2 - 1.0 / p) * max(gd.h_i)
        boxes = gd.get_boxes(spos, rcut, cut)
        self.boxes = []
        self.displacements = num.zeros((len(boxes), 3), num.Float)
        b = 0
        angle = gd.domain.angle
        
        from math import cos, sin
        for beg_i, end_i, disp in boxes:
            if angle is None:
                rspos = spos
            else:
                da = angle*disp[0]
                tspos = spos-0.5
                rspos = num.array([tspos[0],
                                   tspos[1]*cos(da)-tspos[2]*sin(da),
                                   tspos[1]*sin(da) + tspos[2]*cos(da)])+0.5
                                      
            box = LocalizedFunctions(functions, end_i - beg_i,
                                     gd.n_i,
                                     beg_i - gd.beg0_i, gd.h_i,
                                     beg_i - (rspos - disp) * gd.N_i,
                                     npts, k, typecode, forces, lfbc)
            self.boxes.append(box)
            self.displacements[b] = disp
            b += 1
        
        self.nfuncs = 0
        self.nfuncsD = 0
        for radial in functions:
            l = radial.get_angular_momentum_number()
            self.nfuncs += 2 * l + 1; 
            self.nfuncsD += 3 + l * (1 + 2 * l)
        self.typecode = typecode

        self.set_communicator(gd.comm, MASTER)

    def set_communicator(self, comm, root):
        self.comm = comm
        self.root = root
        
    def add(self, grids, coefficients, kpt=None, communicate=False):
        if communicate and mpi.parallel:
            if coefficients is None:
                shape = grids.shape[:-3] + (self.nfuncs,)
                coefficients = num.zeros(shape, self.typecode)
            self.comm.broadcast(coefficients, self.root)
            
        if kpt is None:
            for box in self.boxes:
                box.add(coefficients, grids)
        else:
            # Should we store the phases for all possible kpoints? or should
            # we recalculate them????????
            phases = num.exp(-2j * pi * num.dot(self.displacements, kpt))
            for box, phase in zip(self.boxes, phases):
                box.add(coefficients * phase, grids)

    def multiply(self, grids, result, kpt=None, derivatives=False):
        if derivatives:
            shape = grids.shape[:-3] + (self.nfuncsD,)
        else:
            shape = grids.shape[:-3] + (self.nfuncs,)
            
        tmp = num.zeros(shape, self.typecode)
        if result is None:
            result = num.zeros(shape, self.typecode)
            
        if kpt is None:
            for box in self.boxes:
                box.multiply(grids, tmp, derivatives)
                result += tmp
        else:
            #!!!
            # Should we store the phases for all possible kpoints? or should
            # we recalculate them???????
            phases = num.exp(2j * pi * num.dot(self.displacements, kpt))
            for box, phase in zip(self.boxes, phases):
                box.multiply(grids, tmp, derivatives)
                result += phase * tmp

        if mpi.parallel:
            self.comm.sum(result, self.root)

    def add_density(self, n_G, f_i):
        for box in self.boxes:
            box.add_density(n_G, f_i)


class LocFuncBroadcaster:
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
