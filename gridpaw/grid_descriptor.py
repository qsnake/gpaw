# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

__docformat__ = "restructuredtext"

import Numeric as num

from math import pi, cos, sin
from cmath import exp

from gridpaw.utilities.complex import cc

assert (-1) % 3 == 2
assert (num.array((-1,)) % 3)[0] != 2 # Grrrr...!!!!


MASTER = 0


class GridDescriptor:
    def __init__(self, domain, ng):
        self.domain = domain
        self.comm = domain.comm
        self.rank = self.comm.rank
        ng = num.array(ng, num.Int)
        self.ng = ng
        if num.sometrue(ng % domain.parsize):
            raise ValueError('Bad number of cells!')
        self.myng = ng / domain.parsize
        self.begin0 = domain.parpos * self.myng #XXX use zero-function instead?
        self.begin = self.begin0.copy()
        for i in range(3):
            if not self.domain.periodic[i] and self.begin[i] == 0:
                self.begin[i] = 1
        self.end = self.begin0 + self.myng
        self.h = domain.cell_i / ng
        self.dv = self.h[0] * self.h[1] * self.h[2]
        self.arraysize = num.product(self.myng)

    def array(self, typecode=num.Float, shape=()):
        return num.zeros(shape + tuple(self.myng), typecode)

    def new_array(self, n=None, typecode=num.Float, zero=True):
        """Return new 3D array for this domain.

        The array will be zeroed unless `zero=True` is used.
        The type can be set with the `typecode` keyword (default:
        `Float`).  An extra dimension can be added with `n=dim`."""

        shape = self.myng
        if n is not None:
            shape = (n,) + tuple(shape)
            
        if zero:
            return num.zeros(shape, typecode)
        else:
            return num.empty(shape, typecode)

    def is_healthy(self):
        return max(self.h) / min(self.h) < 1.3

    def integrate(self, a_g):
        return self.comm.sum(num.sum(a_g.flat)) * self.dv
    
    def coarsen(self):
        if num.sometrue(self.myng % 2):
            raise ValueError('Grid %s not divisable by 2!' % self.myng)
        return GridDescriptor(self.domain, self.ng / 2)

    def get_boxes(self, spos, rcut, cut):
        ng = self.ng
        ncut = rcut / self.h
        npos = spos * ng
        begin = num.ceil(npos - ncut).astype(num.Int)
        end   = num.ceil(npos + ncut).astype(num.Int)

        if cut:
            for i in range(3):
                if not self.domain.periodic[i]:
                    if begin[i] < 0:
                        begin[i] = 0
                    if end[i] > ng[i]:
                        end[i] = ng[i]
        else:
            for i in range(3):
                if not self.domain.periodic[i] and \
                       (begin[i] < 0 or end[i] > ng[i]):
                    raise RuntimeError('Atom too close to boundary!')

        ranges = ([], [], [])
        
        for i in range(3):
            b = begin[i]
            e = b
            
            while e < end[i]:
                b0 = b % ng[i]
               
                e = min(end[i], b + ng[i] - b0)

                if b0 < self.begin[i]:
                    b1 = b + self.begin[i] - b0
                else:
                    b1 = b
                    
                e0 = b0 - b + e
                              
                if e0 > self.end[i]:
                    e1 = e - (e0 - self.end[i])
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
                        begin = num.array((b0 % ng[0], b1 % ng[1], b2 % ng[2]))
                        end = begin + e - b
                        disp = (b - begin) / ng
                        begin = num.maximum(begin, self.begin)
                        end = num.minimum(end, self.end)
                        if (begin[0] < end[0] and
                            begin[1] < end[1] and
                            begin[2] < end[2]):
                            boxes.append((begin, end, disp))

            return boxes
        else:
            #angle er on, derfor kun periodicitet i 1.akse
            #roter centrum.
         
            for b0, e0 in ranges[0]:
                b1, e1 = ranges[1][0]
                [(b2, e2)] = ranges[2]
                
                b = num.array((b0, b1, b2))
                e = num.array((e0, e1, e2))
                  
                begin = num.array((b0 % ng[0], b1 % ng[1], b2 % ng[2]))
                end = begin + e - b
                
                disp = (b - begin) / ng
                da = self.domain.angle*disp[0]
                
                begin = num.maximum(begin, self.begin)
                end = num.minimum(end, self.end)
                
                ###Noget her, foskydning?!?                  
                l = 0.5*(end - begin)
                c = 0.5*(end + begin) - 0.5 * ng
                
                newc = num.array([c[0],c[1]*cos(da)-c[2]*sin(da),
                                 c[1]*sin(da) + c[2]*cos(da)])+0.5*ng
                
                begin = num.floor(newc - l).astype(num.Int) - 1
                end = num.ceil(newc + l).astype(num.Int) + 1
                begin = num.maximum(begin, self.begin)
                end = num.minimum(end, self.end)                 
                                  
                if (begin[0] < end[0] and
                    begin[1] < end[1] and
                    begin[2] < end[2]):
                    boxes.append((begin, end, disp))
            return boxes


    def mirror(self, a, axis):
        N = self.domain.parsize[axis]
        if axis == 0:
            b = a.copy()
        else:
            axes = [0, 1, 2]
            axes[axis] = 0
            axes[0] = axis
            b = num.transpose(a, axes).copy()
        n = self.domain.parpos[axis]
        m = (-n) % N
        if n != m:
            rank = self.rank + (m - n) * self.domain.strides[axis]
            request = self.comm.receive(b[0], rank, False)
            self.comm.send(b[0].copy(), rank)
            self.comm.wait(request)
        c = b[-1:0:-1].copy()
        m = N - n - 1
        if n != m:
            rank = self.rank + (m - n) * self.domain.strides[axis]
            request = self.comm.receive(b[1:], rank, False)
            self.comm.send(c, rank)
            self.comm.wait(request)
        else:
            b[1:] = c
        if axis == 0:
            return b
        else:
            return num.transpose(b, axes).copy()
                
    def swap_axes(self, a_g, axes):
        assert num.alltrue(self.ng == num.take(self.ng, axes)), \
               'Can only swap axes with same length!'

        if not (self.comm.size>1):
            return num.transpose(a_g, axes).copy()

        # Collect all arrays on the master, do the swapping, and
        # redistribute the result:
        if self.rank != MASTER:
            self.comm.gather(a_g, MASTER)
            b_g = self.array()
            self.comm.receive(b_g, MASTER)
            return b_g
        else:
            b_cg = num.zeros((self.comm.size,) + a_g.shape, num.Float)
            self.comm.gather(a_g, MASTER, b_cg)

            # Put the subdomains from the slaves into the big array
            # for the whole domain:
            big_g = num.zeros(self.ng, num.Float)
            parsize = self.domain.parsize
            n0, n1, n2 = self.myng
            c = 0
            for nx in range(parsize[0]):
                for ny in range(parsize[1]):
                    for nz in range(parsize[2]):
                        big_g[nx * n0:(nx + 1) * n0,
                              ny * n1:(ny + 1) * n1,
                              nz * n2:(nz + 1) * n2] = b_cg[c]
                        c += 1
                        
            big_g = num.transpose(big_g, axes).copy()

            b_g = b_cg[0]
            c = 0
            for nx in range(parsize[0]):
                for ny in range(parsize[1]):
                    for nz in range(parsize[2]):
                        b_g[:] = big_g[nx * n0:(nx + 1) * n0,
                                       ny * n1:(ny + 1) * n1,
                                       nz * n2:(nz + 1) * n2]
                        if c != MASTER:
                            self.comm.send(b_g, c)
                        c += 1
            return big_g[:n0, :n1, :n2].copy()

    def collect(self, a_g):
        if self.comm.size == 1:
            return a_g

        # Collect all arrays on the master:
        if self.rank != MASTER:
            self.comm.gather(a_g, MASTER)
            return
        else:
            b_cg = num.zeros((self.comm.size,) + a_g.shape, a_g.typecode())
            self.comm.gather(a_g, MASTER, b_cg)

            # Put the subdomains from the slaves into the big array
            # for the whole domain:
            big_g = num.zeros(a_g.shape[:-3] + tuple(self.ng), a_g.typecode())
            parsize = self.domain.parsize
            n0, n1, n2 = self.myng
            c = 0
            for nx in range(parsize[0]):
                for ny in range(parsize[1]):
                    for nz in range(parsize[2]):
                        big_g[...,
                              nx * n0:(nx + 1) * n0,
                              ny * n1:(ny + 1) * n1,
                              nz * n2:(nz + 1) * n2] = b_cg[c]
                        c += 1
            return big_g
        
    def calculate_dipole_moment(self, rho_xyz):
        rho_xy = num.sum(rho_xyz, 2)
        rho_xz = num.sum(rho_xyz, 1)
        rho_ig = [num.sum(rho_xy, 1), num.sum(rho_xy, 0), num.sum(rho_xz, 0)]
        d_i = num.zeros(3, num.Float)
        for i in range(3):
            ri = (num.arange(self.myng[i], typecode=num.Float) +
                  self.begin0[i]) * self.h[i]
            d_i[i] = -num.dot(ri, rho_ig[i]) * self.dv
        self.comm.sum(d_i)
        return d_i

    def wannier_matrix(self, psit_nG, i):
        nbands = len(psit_nG)
        Z_n1n2 = num.zeros((nbands, nbands), num.Complex)
        shape = (nbands, self.arraysize / self.myng[i])
        for g in range(self.myng[i]):
            e = exp(2j * pi / self.ng[i] * (g + self.begin0[i]))
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
    def __init__(self, r_g, dr_g):
        self.r_g = r_g
        self.dr_g = dr_g
        self.dv_g = 4 * pi * r_g**2 * dr_g

    def derivative(self, n_g, dndr_g):
        dndr_g[0] = n_g[1] - n_g[0]
        dndr_g[1:-1] = 0.5 * (n_g[2:] - n_g[:-2])
        dndr_g[-1] = n_g[-1] - n_g[-2]
        dndr_g /= self.dr_g

    def derivative2(self, a_g, b_g):
        c_g = a_g / self.dr_g
        b_g[0] = 0.5 * c_g[1] + c_g[0]
        b_g[1] = 0.5 * c_g[2] - c_g[0]
        b_g[1:-1] = 0.5 * (c_g[2:] - c_g[:-2])
        b_g[-2] = c_g[-1] - 0.5 * c_g[-3]
        b_g[-1] = -c_g[-1] - 0.5 * c_g[-2]
