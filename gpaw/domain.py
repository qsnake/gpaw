# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Domain object.

This module contains the definition of the ``Domain`` class and some
helper functins for parallel domain decomposition.  """

import numpy as npy

from gpaw.mpi import serial_comm


class Domain:
    """Domain class.

    A ``Domain`` object (in `domain.py`) holds informaion on the unit
    cell and the boundary conditions"""
    
    def __init__(self, cell, pbc=(True, True, True)):
        """Create Domain object from a unit cell and boundary conditions.

        The arguments are the lengths of the three axes, followed by a
        tuple of three periodic-boundary condition flags (``bool``'s).

        Parallel stuff:
         =============== ==================================================
         ``sdisp_cd``    Scaled displacement in direction ``d`` along axis
                         ``c``.
         ``comm``        MPI-communicator for this domain.
         ``neighbor_cd`` Rank of neighbor CPU in direction ``d`` along axis
                         ``c``.
         ``parpos_c``    Position of this CPU in the 3D grid of all CPUs.
         ``parsize_c``   Domain decomposition.
         ``stride_c``    Strides.
         =============== ==================================================
        """
        
        self.cell_c = npy.array(cell, float)
        if self.cell_c.ndim == 1:
            self.cell_cv = npy.diag(self.cell_c)
        else:
            self.cell_cv = self.cell_c
            self.cell_c = self.cell_cv.diagonal()

        self.icell_cv = npy.linalg.inv(self.cell_cv).T
            
        self.pbc_c = npy.asarray(pbc, bool)
        
        self.set_decomposition(serial_comm, (1, 1, 1))

        self.comms = {}
        
    def set_decomposition(self, comm, parsize_c=None, N_c=None):
        """Set MPI-communicator and do domain decomposition.

        With ``parsize_c=(a, b, c)``, the domin will be divided in
        ``a*b*c`` sub-domains - one for each CPU in ``comm``.  If
        ``parsize_c`` is not given, the number of grid points will be
        used to suggest a good domain decomposition."""
        
        self.comm = comm

        if parsize_c is None:
            parsize_c = decompose_domain(N_c, comm.size)
        self.parsize_c = npy.array(parsize_c)

        self.stride_c = npy.array([parsize_c[1] * parsize_c[2],
                                   parsize_c[2],
                                   1])
        
        if npy.product(self.parsize_c) != self.comm.size:
            raise RuntimeError('Bad domain decomposition!')

        rnk = self.comm.rank
        self.parpos_c = npy.array(
            [rnk // self.stride_c[0],
             (rnk % self.stride_c[0]) // self.stride_c[1],
             rnk % self.stride_c[1]])
        assert npy.dot(self.parpos_c, self.stride_c) == rnk

        self.find_neighbor_processors()

    def scale_position(self, pos_v):
        """Return scaled position.

        Return array with the coordinates scaled to the interval [0,
        1)."""
        
        spos_c = npy.linalg.solve(self.cell_cv.T, pos_v)

        for c in range(3):
            if self.pbc_c[c]:
                spos_c[c] %= 1.0
        return spos_c

    def get_rank_for_position(self, spos_c):
        """Calculate rank of domain containing scaled position."""
        rnk_c = npy.clip(npy.floor(spos_c * self.parsize_c).astype(int),
                         0, npy.array(self.parsize_c) - 1)
        for c in range(3):
            assert 0 <= rnk_c[c] < self.parsize_c[c], 'Bad bad!'
        return npy.dot(rnk_c, self.stride_c)

    def find_neighbor_processors(self):
        """Find neighbor processors - surprise!

        With ``i`` and ``d`` being the indices for the axis (x, y or
        z) and direction + or - (0 or 1), two attributes are
        calculated:

        * ``neighbor_cd``:  Rank of neighbor.
        * ``disp_cd``:  Displacement for neighbor.
        """
        
        self.neighbor_cd = npy.zeros((3, 2), int)
        self.sdisp_cd = npy.zeros((3, 2), int)
        for c in range(3):
            p = self.parpos_c[c]
            for d in range(2):
                sdisp = 2 * d - 1
                pd = p + sdisp
                pd0 = pd % self.parsize_c[c]
                self.neighbor_cd[c, d] = (self.comm.rank +
                                          (pd0 - p) * self.stride_c[c])
                if pd0 != pd:
                    # Wrap around the box?
                    if self.pbc_c[c]:
                        # Yes:
                        self.sdisp_cd[c, d] = -sdisp
                    else:
                        # No:
                        self.neighbor_cd[c, d] = -1

    def get_communicator(self, group):
        t = tuple(group)
        if t in self.comms:
            return self.comms[t]
        comm = self.comm.new_communicator(npy.array(group))
        self.comms[t] = comm
        return comm


def decompose_domain(ng, p):
    if p == 1:
        return (1, 1, 1)
    n1, n2, n3 = ng
    plist = prims(p)
    pdict = {}
    for n in plist:
        pdict[n] = 0
    for n in plist:
        pdict[n] += 1
    candidates = factorizations(pdict.items())
    mincost = 10000000000.0  
    best = None
    for p1, p2, p3 in candidates:
        if n1 % p1 != 0 or n2 % p2 != 0 or n3 % p3 != 0:
            continue
        m1 = n1 / p1
        m2 = n2 / p2
        m3 = n3 / p3
        cost = abs(m1 - m2) + abs(m2 - m3) + abs(m3 - m1)
        # A long z-axis (unit stride) is best:
        if m1 <= m2 <= m3:
            cost -= 0.1
        if cost < mincost:
            mincost = cost
            best = (p1, p2, p3)
    if best is None:
        raise RuntimeError("Can't decompose a %dx%dx%d grid on %d cpu's!" %
                           (n1, n2, n3, p))
    return best


def factorizations(f):
    p, n = f[0]
    facs0 = []
    for n1 in range(n + 1):
        for n2 in range(n - n1 + 1):
            n3 = n - n1 - n2
            facs0.append((p**n1, p**n2, p**n3))
    if len(f) == 1:
        return facs0
    else:
        facs = factorizations(f[1:])
        facs1 = []
        for p1, p2, p3 in facs0:
            facs1 += [(p1 * q1, p2 * q2, p3 * q3) for q1, q2, q3 in facs]
        return facs1

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
          59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
          127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
          191, 193, 197, 199]

def prims(p):
    if p == 1:
        return []
    for n in primes:
        if p % n == 0:
            return prims(p / n) + [n]
    raise RuntimeError
