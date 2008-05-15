# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Grid-descriptors

This module contains classes defining two kinds of grids:

* Uniform 3D grids.
* Radial grids.
"""

from math import pi, cos, sin
from cmath import exp

import numpy as npy

from gpaw.utilities.complex import cc

# Remove this:  XXX
assert (-1) % 3 == 2
assert (npy.array([-1]) % 3)[0] == 2


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

     >>> a = npy.zeros((2, 2, 2))
     >>> a.ravel()[:] = range(8)
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

        self.N_c = npy.array(N_c, int)

        #if npy.sometrue(self.N_c % domain.parsize_c):
        #    raise ValueError('Bad number of CPUs!')

        parsize_c = domain.parsize_c
        n_c, remainder_c = divmod(N_c, parsize_c)

        self.beg_c = npy.empty(3, int)
        self.end_c = npy.empty(3, int)

        self.n_cp = []
        for c in range(3):
            n_p = npy.arange(parsize_c[c] + 1) * float(N_c[c]) / parsize_c[c]
            n_p = npy.around(n_p + 0.4999).astype(int)
            
            if not domain.pbc_c[c]:
                n_p[0] = 1

            if not npy.alltrue(n_p[1:] - n_p[:-1]):
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
        return self.N_c - 1 + self.domain.pbc_c

    def get_slice(self):
        return [slice(b - 1 + p, e - 1 + p) for b, e, p in
                zip(self.beg_c, self.end_c, self.domain.pbc_c)]

    def zeros(self, n=(), dtype=float, global_array=False):
        """Return new zeroed 3D array for this domain.

        The type can be set with the ``dtype`` keyword (default:
        ``float``).  Extra dimensions can be added with ``n=dim``.  A
        global array spanning all domains can be allocated with
        ``global_array=True``."""

        return self._new_array(n, dtype, True, global_array)
    
    def empty(self, n=(), dtype=float, global_array=False):
        """Return new uninitialized 3D array for this domain.

        The type can be set with the ``dtype`` keyword (default:
        ``float``).  Extra dimensions can be added with ``n=dim``.  A
        global array spanning all domains can be allocated with
        ``global_array=True``."""

        return self._new_array(n, dtype, False, global_array)
        
    def _new_array(self, n=(), dtype=float, zero=True,
                  global_array=False):
        if global_array:
            shape = self.get_size_of_global_array()
        else:
            shape = self.n_c
            
        if isinstance(n, int):
            n = (n,)

        shape = n + tuple(shape)

        if zero:
            return npy.zeros(shape, dtype)
        else:
            return npy.empty(shape, dtype)
        
    def integrate(self, a_xg):
        """Integrate function in array over domain."""
        shape = a_xg.shape
        if len(shape) == 3:
            return self.comm.sum(a_xg.sum()) * self.dv
        A_x = npy.sum(npy.reshape(a_xg, shape[:-3] + (-1,)), axis=-1)
        self.comm.sum(A_x)
        return A_x * self.dv
    
    def coarsen(self):
        """Return coarsened `GridDescriptor` object.

        Reurned descriptor has 2x2x2 fewer grid points."""
        
        if npy.sometrue(self.N_c % 2):
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
        beg_c = npy.ceil(npos_c - ncut).astype(int)
        end_c   = npy.ceil(npos_c + ncut).astype(int)

        if cut:
            for c in range(3):
                if not self.domain.pbc_c[c]:
                    if beg_c[c] < 0:
                        beg_c[c] = 0
                    if end_c[c] > N_c[c]:
                        end_c[c] = N_c[c]
        else:
            for c in range(3):
                if (not self.domain.pbc_c[c] and
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
                    b = npy.array((b0, b1, b2))
                    e = npy.array((e0, e1, e2))
                    beg_c = npy.array((b0 % N_c[0], b1 % N_c[1], b2 % N_c[2]))
                    end_c = beg_c + e - b
                    disp = (b - beg_c) / N_c
                    beg_c = npy.maximum(beg_c, self.beg_c)
                    end_c = npy.minimum(end_c, self.end_c)
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
            b_g = npy.transpose(a_g, axes).copy()
        n = self.domain.parpos_c[c]
        m = (-n) % N
        if n != m:
            rank = self.rank + (m - n) * self.domain.stride_c[c]
            b_yz = b_g[0].copy()
            request = self.comm.receive(b_g[0], rank, 117, NONBLOCKING)
            self.comm.send(b_yz, rank, 117)
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
            return npy.transpose(b_g, axes).copy()
                
    def swap_axes(self, a_g, axes):
        """Swap axes of array.

        The ``axes`` argument gives the new ordering of the axes.
        Example: With ``axes=(0, 2, 1)`` the *y*, *z* axes will be
        swapped."""
        
        assert npy.alltrue(self.N_c == npy.take(self.N_c, axes)), \
               'Can only swap axes with same length!'

        if self.comm.size == 1:
            return npy.transpose(a_g, axes).copy()

        # Collect all arrays on the master, do the swapping, and
        # redistribute the result:
        A_g = self.collect(a_g)

        if self.rank == MASTER:
            A_g = npy.transpose(A_g, axes).copy()

        b_g = self.empty()
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
        A_xg = self.empty(xshape, a_xg.dtype.char, global_array=True)
        parsize_c = self.domain.parsize_c
        r = 0
        for n0 in range(parsize_c[0]):
            b0, e0 = self.n_cp[0][n0:n0 + 2] - self.beg_c[0]
            for n1 in range(parsize_c[1]):
                b1, e1 = self.n_cp[1][n1:n1 + 2] - self.beg_c[1]
                for n2 in range(parsize_c[2]):
                    b2, e2 = self.n_cp[2][n2:n2 + 2] - self.beg_c[2]
                    if r != MASTER:
                        a_xg = npy.empty(xshape + 
                                         ((e0 - b0), (e1 - b1), (e2 - b2)),
                                         a_xg.dtype.char)
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
        
    def zero_pad(self, a_xg):
        """Pad array with zeros as first element along non-periodic directions.

        XXX Does not work for parallel domain-distributed arrays.
        """
        pbc_c = self.domain.pbc_c

        if pbc_c.all():
            return a_xg

        npbx, npby, npbz = 1 - pbc_c
        shape = npy.array(a_xg.shape)
        shape[-3:] += [npbx, npby, npbz]
        b_xg = npy.zeros(shape, dtype=a_xg.dtype)
        b_xg[..., npbx:, npby:, npbz:] = a_xg
        return b_xg

    def calculate_dipole_moment(self, rho_xyz):
        """Calculate dipole moment of density."""
        rho_xy = npy.sum(rho_xyz, axis=2)
        rho_xz = npy.sum(rho_xyz, axis=1)
        rho_cg = [npy.sum(rho_xy, axis=1),
                  npy.sum(rho_xy, axis=0),
                  npy.sum(rho_xz, axis=0)]
        d_c = npy.zeros(3)
        for c in range(3):
            r_g = (npy.arange(self.n_c[c], dtype=float) +
                   self.beg_c[c]) * self.h_c[c]
            d_c[c] = -npy.dot(r_g, rho_cg[c]) * self.dv
        self.comm.sum(d_c)
        return d_c

    def wannier_matrix(self, psit_nG, psit_nG1, c, G):
        """Wannier localization integrals

        The soft part of Z is given by (Eq. 27 ref1)::

            ~       ~     -i G.r   ~
            Z   = <psi | e      |psi >
             nm       n             m
                    
        G is 1/N_c, where N_c is the number of k-points along axis c, psit_nG
        and psit_nG1 are the set of wave functions for the two different
        spin/kpoints in question.

        ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005) 
        """
        same_wave = False
        if psit_nG is psit_nG1:
            same_wave = True

        nbands = len(psit_nG)
        Z_nn = npy.zeros((nbands, nbands), complex)
        psit_nG = psit_nG[:]
        if not same_wave:
            psit_nG1 = psit_nG1[:]
            
        def get_slice(c, g, psit_nG):
            if c == 0:
                slice_nG = psit_nG[:, g].copy()
            elif c == 1:
                slice_nG = psit_nG[:, :, g].copy()
            else:
                slice_nG = psit_nG[:, :, :, g].copy()
            slice_nG.shape = (nbands, -1)
            return slice_nG

        for g in range(self.n_c[c]):
            A_nG = get_slice(c, g, psit_nG)
                
            if same_wave:
                B_nG = A_nG
            else:
                B_nG = get_slice(c, g, psit_nG1)
                
            e = exp(-2.j * pi * G * (g + self.beg_c[c]) / self.N_c[c])
            Z_nn += e * npy.dot(cc(A_nG), npy.transpose(B_nG)) * self.dv
            
        return Z_nn

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

    def integrate(self, f_g):
        """Integrate over a radial grid."""
        
        return npy.sum(self.dv_g * f_g)
