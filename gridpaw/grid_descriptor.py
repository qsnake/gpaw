# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Grid-descriptors

This module contains classes defining two kinds of grids:

* Uniform 3D grids.
* Radial grids.
"""

import Numeric as num

from math import pi, cos, sin
from cmath import exp

from gridpaw.utilities.complex import cc

# Be careful!  Python integers and arrays of integers behave differently:
assert (-1) % 3 == 2
assert (num.array([-1]) % 3)[0] == -1 # Grrrr...!!!!


MASTER = 0


class GridDescriptor:
    """Descriptor-class for uniform 3D grid

    A ``GridDescriptor`` object holds information on how functions, such
    as wave functions and electron densities, are discreticed in a
    certain domain in space.  The main information here is how many
    grid points are used in each direction of the unit cell.

    There are methods for tasks such as allocating arrays, performing
    rotation- and mirror-symmetry operations and integrating functions
    over space.  All methods work correctly also when the domain is
    parallelized via domain decomposition."""
    
    def __init__(self, domain, N_i):
        """Construct `GridDescriptor`

        A uniform 3D grid is defined by a ``Domain`` object and the
        number of grid points ``N_i`` in x, y, and z-directions (three
        integers)."""
        
        self.domain = domain
        self.comm = domain.comm
        self.rank = self.comm.rank

        self.N_i = num.array(N_i, num.Int)

        if num.sometrue(self.N_i % domain.parsize_i):
            raise ValueError('Bad number of CPUs!')

        self.myN_i = self.N_i / domain.parsize_i

        self.beg_i = domain.parpos_i * self.myN_i
        self.end_i = self.beg_i + self.myN_i
        self.beg0_i = self.beg_i.copy()

        for i in range(3):
            if not self.domain.periodic_i[i] and self.beg_i[i] == 0:
                self.beg_i[i] = 1
        
        self.h_i = domain.cell_i / N_i
        self.dv = self.h_i[0] * self.h_i[1] * self.h_i[2]

    def new_array(self, n=None, typecode=num.Float, zero=True):
        """Return new 3D array for this domain.

        The array will be zeroed unless ``zero=False`` is used.  The
        type can be set with the ``typecode`` keyword (default:
        ``Float``).  An extra dimension can be added with
        ``n=dim``."""

        shape = self.myN_i
        if n is not None:
            shape = (n,) + tuple(shape)
            
        if zero:
            return num.zeros(shape, typecode)
        else:
            return num.empty(shape, typecode)

    def is_healthy(self):
        """Sanity check for grid spacings."""
        return max(self.h_i) / min(self.h_i) < 1.3

    def integrate(self, a_g):
        """Integrate function in array over domain."""
        return self.comm.sum(num.sum(a_g.flat)) * self.dv
    
    def coarsen(self):
        """Return coarsened `GridDescriptor` object.

        Reurned descriptor has 2x2x2 fewer grid points."""
        
        if num.sometrue(self.myN_i % 2):
            raise ValueError('Grid %s not divisable by 2!' % self.myN_i)
        return GridDescriptor(self.domain, self.N_i / 2)

    def get_boxes(self, spos, rcut, cut=True):
        """Find boxes enclosing sphere."""
        N_i = self.N_i
        ncut = rcut / self.h_i
        npos = spos * N_i
        beg_i = num.ceil(npos - ncut).astype(num.Int)
        end_i   = num.ceil(npos + ncut).astype(num.Int)

        if cut:
            for i in range(3):
                if not self.domain.periodic_i[i]:
                    if beg_i[i] < 0:
                        beg_i[i] = 0
                    if end_i[i] > N_i[i]:
                        end_i[i] = N_i[i]
        else:
            for i in range(3):
                if not self.domain.periodic_i[i] and \
                       (beg_i[i] < 0 or end_i[i] > N_i[i]):
                    raise RuntimeError('Atom too close to boundary!')

        ranges = ([], [], [])
        
        for i in range(3):
            b = beg_i[i]
            e = b
            
            while e < end_i[i]:
                b0 = b % N_i[i]
               
                e = min(end_i[i], b + N_i[i] - b0)

                if b0 < self.beg_i[i]:
                    b1 = b + self.beg_i[i] - b0
                else:
                    b1 = b
                    
                e0 = b0 - b + e
                              
                if e0 > self.end_i[i]:
                    e1 = e - (e0 - self.end_i[i])
                else:
                    e1 = e
                if e1 > b1:
                    ranges[i].append((b1, e1))
                b = e
        
        boxes = []

        if self.domain.angle is None:
            for b0, e0 in ranges[0]:
                for b1, e1 in ranges[1]:
                    for b2, e2 in ranges[2]:
                        b = num.array((b0, b1, b2))
                        e = num.array((e0, e1, e2))
                        beg_i = num.array((b0 % N_i[0], b1 % N_i[1], b2 % N_i[2]))
                        end_i = beg_i + e - b
                        disp = (b - beg_i) / N_i
                        beg_i = num.maximum(beg_i, self.beg_i)
                        end_i = num.minimum(end_i, self.end_i)
                        if (beg_i[0] < end_i[0] and
                            beg_i[1] < end_i[1] and
                            beg_i[2] < end_i[2]):
                            boxes.append((beg_i, end_i, disp))

            return boxes
        else:
            #angle er on, derfor kun periodicitet i 1.akse
            #roter centrum.
         
            for b0, e0 in ranges[0]:
                b1, e1 = ranges[1][0]
                [(b2, e2)] = ranges[2]
                
                b = num.array((b0, b1, b2))
                e = num.array((e0, e1, e2))
                  
                beg_i = num.array((b0 % N_i[0], b1 % N_i[1], b2 % N_i[2]))
                end_i = beg_i + e - b
                
                disp = (b - beg_i) / N_i
                da = self.domain.angle*disp[0]
                
                beg_i = num.maximum(beg_i, self.beg_i)
                end_i = num.minimum(end_i, self.end_i)
                
                ###Noget her, foskydning?!?                  
                l = 0.5*(end_i - beg_i)
                c = 0.5*(end_i + beg_i) - 0.5 * N_i
                
                newc = num.array([c[0],c[1]*cos(da)-c[2]*sin(da),
                                 c[1]*sin(da) + c[2]*cos(da)])+0.5*N_i
                
                beg_i = num.floor(newc - l).astype(num.Int) - 1
                end_i = num.ceil(newc + l).astype(num.Int) + 1
                beg_i = num.maximum(beg_i, self.beg_i)
                end_i = num.minimum(end_i, self.end_i)                 

##                beg_i = self.beg_i.copy();print '.....'
##                end_i = self.end_i.copy()
                
                if (beg_i[0] < end_i[0] and
                    beg_i[1] < end_i[1] and
                    beg_i[2] < end_i[2]):
                    boxes.append((beg_i, end_i, disp))
            return boxes


    def mirror(self, a_g, i):
        """Apply mirror symmetry to array.

        The mirror plane goes through origo and is perpendicular to
        the ``i``'th axis: 0, 1, 2 -> *x*, *y*, *z*."""
        
        N = self.domain.parsize_i[i]
        if i == 0:
            b_g = a_g.copy()
        else:
            axes = [0, 1, 2]
            axes[i] = 0
            axes[0] = i
            b_g = num.transpose(a_g, axes).copy()
        n = self.domain.parpos_i[i]
        m = (-n) % N
        if n != m:
            rank = self.rank + (m - n) * self.domain.stride_i[i]
            request = self.comm.receive(b_g[0], rank, False)
            self.comm.send(b_g[0].copy(), rank)
            self.comm.wait(request)
        c_g = b_g[-1:0:-1].copy()
        m = N - n - 1
        if n != m:
            rank = self.rank + (m - n) * self.domain.stride_i[i]
            request = self.comm.receive(b_g[1:], rank, False)
            self.comm.send(c_g, rank)
            self.comm.wait(request)
        else:
            b_g[1:] = c_g
        if i == 0:
            return b_g
        else:
            return num.transpose(b_g, axes).copy()
                
    def swap_axes(self, a_g, axes):
        """Swap axes of array.

        The ``axes`` argument gives the new ordering of the axes.
        Example: With ``axes=(0, 2, 1)`` the *y*, *z* axes will be
        swapped."""
        
        assert num.alltrue(self.N_i == num.take(self.N_i, axes)), \
               'Can only swap axes with same length!'

        if self.comm.size == 1:
            return num.transpose(a_g, axes).copy()

        # Collect all arrays on the master, do the swapping, and
        # redistribute the result:
        if self.rank != MASTER:
            self.comm.gather(a_g, MASTER)
            b_g = self.new_array()
            self.comm.receive(b_g, MASTER)
            return b_g
        else:
            b_cg = self.new_array(self.comm.size)
            self.comm.gather(a_g, MASTER, b_cg)

            # Put the subdomains from the slaves into the big array
            # for the whole domain:
            B_g = num.zeros(self.N_i, num.Float)
            parsize_i = self.domain.parsize_i
            n0, n1, n2 = self.myN_i
            c = 0
            for nx in range(parsize_i[0]):
                for ny in range(parsize_i[1]):
                    for nz in range(parsize_i[2]):
                        B_g[nx * n0:(nx + 1) * n0,
                            ny * n1:(ny + 1) * n1,
                            nz * n2:(nz + 1) * n2] = b_cg[c]
                        c += 1
                        
            B_g = num.transpose(B_g, axes).copy()

            b_g = b_cg[0]
            c = 0
            for nx in range(parsize_i[0]):
                for ny in range(parsize_i[1]):
                    for nz in range(parsize_i[2]):
                        b_g[:] = B_g[nx * n0:(nx + 1) * n0,
                                     ny * n1:(ny + 1) * n1,
                                     nz * n2:(nz + 1) * n2]
                        if c != MASTER:
                            self.comm.send(b_g, c)
                        c += 1
            return B_g[:n0, :n1, :n2].copy()

    def collect(self, a_g):
        """Collect distributed array to master-CPU."""
        if self.comm.size == 1:
            return a_g

        # Collect all arrays on the master:
        if self.rank != MASTER:
            self.comm.gather(a_g, MASTER)
            return
        else:
            b_cg = self.new_array(self.comm.size, a_g.typecode())
            self.comm.gather(a_g, MASTER, b_cg)

            # Put the subdomains from the slaves into the big array
            # for the whole domain:
            B_g = num.zeros(a_g.shape[:-3] + tuple(self.N_i), a_g.typecode())
            parsize_i = self.domain.parsize_i
            n0, n1, n2 = self.myN_i
            c = 0
            for nx in range(parsize_i[0]):
                for ny in range(parsize_i[1]):
                    for nz in range(parsize_i[2]):
                        B_g[...,
                            nx * n0:(nx + 1) * n0,
                            ny * n1:(ny + 1) * n1,
                            nz * n2:(nz + 1) * n2] = b_cg[c]
                        c += 1
            return B_g
        
    def calculate_dipole_moment(self, rho_xyz):
        """Calculate dipole moment of density."""
        rho_xy = num.sum(rho_xyz, 2)
        rho_xz = num.sum(rho_xyz, 1)
        rho_ig = [num.sum(rho_xy, 1), num.sum(rho_xy, 0), num.sum(rho_xz, 0)]
        d_i = num.zeros(3, num.Float)
        for i in range(3):
            r_g = (num.arange(self.myN_i[i], typecode=num.Float) +
                   self.beg0_i[i]) * self.h_i[i]
            d_i[i] = -num.dot(r_g, rho_ig[i]) * self.dv
        self.comm.sum(d_i)
        return d_i

    def wannier_matrix(self, psit_nG, i):
        """Wannier localization integrals."""
        nbands = len(psit_nG)
        Z_n1n2 = num.zeros((nbands, nbands), num.Complex)
        shape = (nbands, -1)
        for g in range(self.myN_i[i]):
            e = exp(2j * pi / self.N_i[i] * (g + self.beg0_i[i]))
            if i == 0:
                A_nG = psit_nG[:, g].copy()
            elif i == 1:
                A_nG = psit_nG[:, :, g].copy()
            else:
                A_nG = psit_nG[:, :, :, g].copy()
            A_nG.shape = shape
            Z_n1n2 += e * num.dot(cc(A_nG), num.transpose(A_nG))
        self.comm.sum(Z_n1n2, MASTER)
        #                __        __      __
        #        ~      \         2||  a  \     a  a    a  *
        # Z    = Z    +  )  exp[i --- R ]  )   P  O   (P  )
        #  nmx    nmx   /__        L   x  /__   ni ii'  mi'
        #
        #                a                 ii'
        return Z_n1n2 * self.dv
    

class RadialGridDescriptor:
    """Descriptor-class for radial grid."""
    def __init__(self, r_g, dr_g):
        """Construct `RadialGridDescriptor`.

        The one-dimensional array ``r_g`` gives the radii of the grid
        points according to some possibly non-linear function:
        ``r_g[g]`` = *f(g)*.  The array ``dr_g[g]`` = *f'(g)* is used
        for forming derivatives."""
        
        self.r_g = r_g
        self.dr_g = dr_g
        self.dv_g = 4 * pi * r_g**2 * dr_g

    def derivative(self, n_g, dndr_g):
        """Finite-difference derivative of radial function."""
        dndr_g[0] = n_g[1] - n_g[0]
        dndr_g[1:-1] = 0.5 * (n_g[2:] - n_g[:-2])
        dndr_g[-1] = n_g[-1] - n_g[-2]
        dndr_g /= self.dr_g

    def derivative2(self, a_g, b_g):
        """Finite-difference derivative of radial function.

        For an infinitely dense grid, this method would be identical
        to the `derivative` method."""
        
        c_g = a_g / self.dr_g
        b_g[0] = 0.5 * c_g[1] + c_g[0]
        b_g[1] = 0.5 * c_g[2] - c_g[0]
        b_g[1:-1] = 0.5 * (c_g[2:] - c_g[:-2])
        b_g[-2] = c_g[-1] - 0.5 * c_g[-3]
        b_g[-1] = -c_g[-1] - 0.5 * c_g[-2]
