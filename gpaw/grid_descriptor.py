# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Grid-descriptors

This module contains classes defining two kinds of grids:

* Uniform 3D grids.
* Radial grids.
"""

from math import pi, cos, sin
from cmath import exp

import Numeric as num

from gpaw.utilities.complex import cc

# Be careful!  Python integers and arrays of integers behave differently:
assert (-1) % 3 == 2
assert (num.array([-1]) % 3)[0] == -1 # Grrrr...!!!!


MASTER = 0
NONBLOCKING = False

class GridDescriptor:
    """Descriptor-class for uniform 3D grid

    A ``GridDescriptor`` object holds information on how functions, such
    as wave functions and electron densities, are discreticed in a
    certain domain in space.  The main information here is how many
    grid points are used in each direction of the unit cell.

    There are methods for tasks such as allocating arrays, performing
    rotation- and mirror-symmetry operations and integrating functions
    over space.  All methods work correctly also when the domain is
    parallelized via domain decomposition.

    This is how a 2x2x2 3D array is layed out in memory::

        3-----7
        |\    |\
        | \   | \
        |  1-----5      z
        2--|--6  |   y  |
         \ |   \ |    \ |
          \|    \|     \|
           0-----4      +-----x

    Example:

     >>> a = num.zeros((2, 2, 2))
     >>> a.flat[:] = range(8)
     >>> a
     array([[[0, 1],
             [2, 3]],
            [[4, 5],
             [6, 7]]])
     """
    
    def __init__(self, domain, N_c):
        """Construct `GridDescriptor`

        A uniform 3D grid is defined by a ``Domain`` object and the
        number of grid points ``N_c`` in *x*, *y*, and *z*-directions
        (three integers).

        Attributes:
         ========== ========================================================
         ``domain`` Domain object.
         ``dv``     Volume per grid point.
         ``h_c``    Array of the grid spacing along the three axes.
         ``N_c``    Array of the number of grid points along the three axes.
         ``n_c``    Number of grid points on this CPU.
         ``beg_c``  Beginning of grid-point indices (inclusive).
         ``end_c``  End of grid-point indices (exclusive).
         ``comm``   MPI-communicator for domain decomosition.
         ========== ========================================================
        """
        
        self.domain = domain
        self.comm = domain.comm
        self.rank = self.comm.rank

        self.N_c = num.array(N_c, num.Int)

        #if num.sometrue(self.N_c % domain.parsize_c):
        #    raise ValueError('Bad number of CPUs!')

        parsize_c = domain.parsize_c
        n_c, remainder_c = divmod(N_c, parsize_c)

        self.beg_c = num.empty(3, num.Int)
        self.end_c = num.empty(3, num.Int)

        self.n_cp = []
        for c in range(3):
            n_p = num.arange(parsize_c[c] + 1) * float(N_c[c]) / parsize_c[c]
            n_p = num.around(n_p + 0.4999).astype(num.Int)
            
            if not domain.periodic_c[c]:
                n_p[0] = 1

            if not num.alltrue(n_p[1:] - n_p[:-1]):
                raise ValueError('Grid too small!')
                    
            self.beg_c[c] = n_p[domain.parpos_c[c]]
            self.end_c[c] = n_p[domain.parpos_c[c] + 1]
            self.n_cp.append(n_p)
            
        self.n_c = self.end_c - self.beg_c

        self.h_c = domain.cell_c / N_c
        self.dv = self.h_c[0] * self.h_c[1] * self.h_c[2]

        # Sanity check for grid spacings:
        if max(self.h_c) / min(self.h_c) > 1.3:
            raise ValueError('Very anisotropic grid spacings: %s' % self.h_c)

    def get_size_of_global_array(self):
        return self.N_c - 1 + self.domain.periodic_c

    def zeros(self, n=(), typecode=num.Float, global_array=False):
        return self.new_array(n, typecode, True, global_array)
    
    def empty(self, n=(), typecode=num.Float, global_array=False):
        return self.new_array(n, typecode, False, global_array)
        
    def new_array(self, n=(), typecode=num.Float, zero=True,
                  global_array=False):
        """Return new 3D array for this domain.

        The array will be zeroed unless ``zero=False`` is used.  The
        type can be set with the ``typecode`` keyword (default:
        ``float``).  Extra dimensions can be added with ``n=dim``.
        A global array spanning all domains can be allocated with
        ``global_array=True``."""

        if global_array:
            shape = self.get_size_of_global_array()
        else:
            shape = self.n_c
            
        if isinstance(n, int):
            n = (n,)

        shape = n + tuple(shape)

        if zero:
            return num.zeros(shape, typecode)
        else:
            return num.empty(shape, typecode)
        
    def integrate(self, a_g):
        """Integrate function in array over domain."""
        return self.comm.sum(num.sum(a_g.flat)) * self.dv
    
    def coarsen(self):
        """Return coarsened `GridDescriptor` object.

        Reurned descriptor has 2x2x2 fewer grid points."""
        
        if num.sometrue(self.N_c % 2):
            raise ValueError('Grid %s not divisable by 2!' % self.N_c)

        return GridDescriptor(self.domain, self.N_c // 2)

    def refine(self):
        """Return refined `GridDescriptor` object.

        Reurned descriptor has 2x2x2 more grid points."""
        return GridDescriptor(self.domain, self.N_c * 2)

    def get_boxes(self, spos_c, rcut, cut=True):
        """Find boxes enclosing sphere."""
        N_c = self.N_c
        ncut = rcut / self.h_c
        npos_c = spos_c * N_c
        beg_c = num.ceil(npos_c - ncut).astype(num.Int)
        end_c   = num.ceil(npos_c + ncut).astype(num.Int)

        if cut:
            for c in range(3):
                if not self.domain.periodic_c[c]:
                    if beg_c[c] < 0:
                        beg_c[c] = 0
                    if end_c[c] > N_c[c]:
                        end_c[c] = N_c[c]
        else:
            for c in range(3):
                if (not self.domain.periodic_c[c] and
                    (beg_c[c] < 0 or end_c[c] > N_c[c])):
                    raise RuntimeError(('Atom at %.3f %.3f %.3f ' +
                                        'too close to boundary ' +
                                        '(beg. of box %s, end of box %s)') %
                                       (tuple(spos_c) + (beg_c, end_c)))

        range_c = ([], [], [])
        
        for c in range(3):
            b = beg_c[c]
            e = b
            
            while e < end_c[c]:
                b0 = b % N_c[c]
               
                e = min(end_c[c], b + N_c[c] - b0)

                if b0 < self.beg_c[c]:
                    b1 = b + self.beg_c[c] - b0
                else:
                    b1 = b
                    
                e0 = b0 - b + e
                              
                if e0 > self.end_c[c]:
                    e1 = e - (e0 - self.end_c[c])
                else:
                    e1 = e
                if e1 > b1:
                    range_c[c].append((b1, e1))
                b = e
        
        boxes = []

        for b0, e0 in range_c[0]:
            for b1, e1 in range_c[1]:
                for b2, e2 in range_c[2]:
                    b = num.array((b0, b1, b2))
                    e = num.array((e0, e1, e2))
                    beg_c = num.array((b0 % N_c[0], b1 % N_c[1], b2 % N_c[2]))
                    end_c = beg_c + e - b
                    disp = (b - beg_c) / N_c
                    beg_c = num.maximum(beg_c, self.beg_c)
                    end_c = num.minimum(end_c, self.end_c)
                    if (beg_c[0] < end_c[0] and
                        beg_c[1] < end_c[1] and
                        beg_c[2] < end_c[2]):
                        boxes.append((beg_c, end_c, disp))

        return boxes

    def mirror(self, a_g, c):
        """Apply mirror symmetry to array.

        The mirror plane goes through origo and is perpendicular to
        the ``c``'th axis: 0, 1, 2 -> *x*, *y*, *z*."""
        
        N = self.domain.parsize_c[c]
        if c == 0:
            b_g = a_g.copy()
        else:
            axes = [0, 1, 2]
            axes[c] = 0
            axes[0] = c
            b_g = num.transpose(a_g, axes).copy()
        n = self.domain.parpos_c[c]
        m = (-n) % N
        if n != m:
            rank = self.rank + (m - n) * self.domain.stride_c[c]
            request = self.comm.receive(b_g[0], rank, 117, NONBLOCKING)
            self.comm.send(b_g[0].copy(), rank, 117)
            self.comm.wait(request)
        c_g = b_g[-1:0:-1].copy()
        m = N - n - 1
        if n != m:
            rank = self.rank + (m - n) * self.domain.stride_c[c]
            request = self.comm.receive(b_g[1:], rank, 118, NONBLOCKING)
            self.comm.send(c_g, rank, 118)
            self.comm.wait(request)
        else:
            b_g[1:] = c_g
        if c == 0:
            return b_g
        else:
            return num.transpose(b_g, axes).copy()
                
    def swap_axes(self, a_g, axes):
        """Swap axes of array.

        The ``axes`` argument gives the new ordering of the axes.
        Example: With ``axes=(0, 2, 1)`` the *y*, *z* axes will be
        swapped."""
        
        assert num.alltrue(self.N_c == num.take(self.N_c, axes)), \
               'Can only swap axes with same length!'

        if self.comm.size == 1:
            return num.transpose(a_g, axes).copy()

        # Collect all arrays on the master, do the swapping, and
        # redistribute the result:
        A_g = self.collect(a_g)

        if self.rank == MASTER:
            A_g = num.transpose(A_g, axes).copy()

        b_g = self.new_array()
        self.distribute(A_g, b_g)
        return b_g

    def collect(self, a_xg):
        """Collect distributed array to master-CPU."""
        if self.comm.size == 1:
            return a_xg

        # Collect all arrays on the master:
        if self.rank != MASTER:
            self.comm.send(a_xg, MASTER, 301)
            return

        # Put the subdomains from the slaves into the big array
        # for the whole domain:
        xshape = a_xg.shape[:-3]
        A_xg = self.new_array(xshape, a_xg.typecode(), global_array=True)
        parsize_c = self.domain.parsize_c
        r = 0
        for n0 in range(parsize_c[0]):
            b0, e0 = self.n_cp[0][n0:n0 + 2] - self.beg_c[0]
            for n1 in range(parsize_c[1]):
                b1, e1 = self.n_cp[1][n1:n1 + 2] - self.beg_c[1]
                for n2 in range(parsize_c[2]):
                    b2, e2 = self.n_cp[2][n2:n2 + 2] - self.beg_c[2]
                    if r != MASTER:
                        a_xg = num.empty(xshape + 
                                         ((e0 - b0), (e1 - b1), (e2 - b2)),
                                         a_xg.typecode())
                        self.comm.receive(a_xg, r, 301)
                    A_xg[..., b0:e0, b1:e1, b2:e2] = a_xg
                    r += 1
        return A_xg

    def distribute(self, B_xg, b_xg):
        """ distribute full array B_xg to subdomains, result in
        b_xg. b_xg must be allocated."""

        if self.comm.size == 1:
            b_xg[:] = B_xg
            return
        
        if self.rank != MASTER:
            self.comm.receive(b_xg, MASTER, 42)
            return
        else:
            parsize_c = self.domain.parsize_c
            requests = []
            r = 0
            for n0 in range(parsize_c[0]):
                b0, e0 = self.n_cp[0][n0:n0 + 2] - self.beg_c[0]
                for n1 in range(parsize_c[1]):
                    b1, e1 = self.n_cp[1][n1:n1 + 2] - self.beg_c[1]
                    for n2 in range(parsize_c[2]):
                        b2, e2 = self.n_cp[2][n2:n2 + 2] - self.beg_c[2]
                        if r != MASTER:
                            a_xg = B_xg[..., b0:e0, b1:e1, b2:e2].copy()
                            request = self.comm.send(a_xg, r, 42, NONBLOCKING)
                            # Remember to store a reference to the
                            # send buffer (a_xg) so that is isn't
                            # deallocated:
                            requests.append((request, a_xg))
                        else:
                            b_xg[:] = B_xg[..., b0:e0, b1:e1, b2:e2]
                        r += 1
                        
            for request, a_xg in requests:
                self.comm.wait(request)
        
    def calculate_dipole_moment(self, rho_xyz):
        """Calculate dipole moment of density."""
        rho_xy = num.sum(rho_xyz, 2)
        rho_xz = num.sum(rho_xyz, 1)
        rho_cg = [num.sum(rho_xy, 1), num.sum(rho_xy, 0), num.sum(rho_xz, 0)]
        d_c = num.zeros(3, num.Float)
        for c in range(3):
            r_g = (num.arange(self.n_c[c], typecode=num.Float) +
                   self.beg_c[c]) * self.h_c[c]
            d_c[c] = -num.dot(r_g, rho_cg[c]) * self.dv
        self.comm.sum(d_c)
        return d_c

    def wannier_matrix(self, psit_nG, psit_nG1, c, k, k1, G):
        """Wannier localization integrals

        For a given **k**,**k'** and **Ga** the soft part of Z is
        given by (Eq. 28 ref1)::

            ~                                 *
            Z = Int exp[i (k'-k-Ga) r] u_nk(r) u_mk'(r) dr 

        A gamma-point calculation correspond to the case (k=k').
        If k<>k1 then k'-k-Ga=0. 

        **Ga** is given by::
        
                    __
                   2||
            G_a =  ---
                    La
                    
        ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005) 

        """

        nbands = len(psit_nG)
        Z_nn = num.zeros((nbands, nbands), num.Complex)
        shape = (nbands, -1)

        psit_nG = psit_nG[:]
        psit_nG1 = psit_nG1[:]
        for g in range(self.n_c[c]):

            if c == 0:
                A_nG = psit_nG[:, g].copy()
            elif c == 1:
                A_nG = psit_nG[:, :, g].copy()
            else:
                A_nG = psit_nG[:, :, :, g].copy()
                
            if k != k1:
                if c == 0:
                    B_nG = psit_nG1[:, g].copy()
                elif c == 1:
                    B_nG = psit_nG1[:, :, g].copy()
                else:
                    B_nG = psit_nG1[:, :, :, g].copy()


            if k == k1: 
                e = exp(2j * pi / self.N_c[c] * (g + self.beg_c[c]))
                B_nG = A_nG
            else:
                e = 1.0

            A_nG.shape = shape
            B_nG.shape = shape
            Z_nn += e * num.dot(cc(A_nG), num.transpose(B_nG))
            
        self.comm.sum(Z_nn, MASTER)
        #                __        __      __
        #        ~      \         2||  a  \     a  a    a  *
        # Z    = Z    +  )  exp[i --- R ]  )   P  O   (P  )
        #  nmx    nmx   /__        L   x  /__   ni ii'  mi'
        #
        #                a                 ii'
        
        return Z_nn * self.dv


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
