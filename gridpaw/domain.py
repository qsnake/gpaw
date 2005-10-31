# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Domain object.

This module contains the definition of the ``Domain`` class and some helper functins for parallel domain decomposition.
"""

import Numeric as num

from gridpaw.utilities.mpi import serial_comm


class Domain:
    """Domain class."""
    def __init__(self, cell_i, periodic_i=(True, True, True), angle=None):
        """Create Domain object from a unit cell and boundary conditions.

        The arguments are the lengths of the three axes, followed by a
        tuple of three periodic-boundary flags (``bool``'s) and
        finally a rotation angle applied to the unit cell after
        translation of one lattice vector in the *x*-direction
        (experimental feature)."""
        
        self.cell_i = num.array(cell_i, num.Float)
        self.periodic_i = periodic_i
	self.angle = angle
        
        self.set_decomposition(serial_comm, (1, 1, 1))
        
    def set_decomposition(self, comm, parsize_i=None, N_i=None):
        """Set MPI-communicator and do domain decomposition.

        With ``parsize_i=(a, b, c)``, the domin will be divided in
        ``a*b*c`` sub-domains - one for each CPU in ``comm``.  If
        ``parsize_i`` is not given, the number of grid points will be
        used to suggest a good domain decomposition."""
        
        self.comm = comm

        if parsize_i is None:
            parsize_i = decompose_domain(N_i, comm.size)
        self.parsize_i = num.array(parsize_i)

        self.stride_i = num.array([parsize_i[1] * parsize_i[2],
                                   parsize_i[2],
                                   1])
        
        if num.product(self.parsize_i) != self.comm.size:
            raise RuntimeError('Bad domain decomposition!')

        rnk = self.comm.rank
        self.parpos_i = num.array(
            [rnk // self.stride_i[0],
             (rnk % self.stride_i[0]) // self.stride_i[1],
             rnk % self.stride_i[1]])
        assert num.dot(self.parpos_i, self.stride_i) == rnk

        self.find_neighbor_processors()

    def scale_position(self, pos_i):
        """Return scaled position.

        Return array with the coordinates scaled to the interval [0,
        1)."""
        
        spos_i = pos_i / self.cell_i
        for i in range(3):
            if self.periodic_i[i]:
                spos_i[i] %= 1.0
        return spos_i

    def rank(self, spos_i):
        """Calculate rank of domain containing scaled position."""
        rnk_i = num.clip(num.floor(spos_i * self.parsize_i).astype(num.Int),
                         0, num.array(self.parsize_i) - 1)
        for i in range(3):
            assert 0 <= rnk_i[i] < self.parsize_i[i], "Bad bad!"
        return num.dot(rnk_i, self.stride_i)

    def find_neighbor_processors(self):
        """Find neighbor processors - surprise!

        With ``i`` and ``d`` being the indices for the axis (x, y or
        z) and direction + or - (0 or 1), two attributes are
        calculated:

        * ``neighbor_id``:  Rank of neighbor.
        * ``disp_id``:  Displacement for neighbor.
        """
        
        self.neighbor_id = num.zeros((3, 2), num.Int)
        self.disp_id = num.zeros((3, 2), num.Int)
        for i in range(3):
            p = self.parpos_i[i]
            for d in range(2):
                disp = 2 * d - 1
                pd = p + disp
                pd0 = pd % self.parsize_i[i]
                self.neighbor_id[i, d] = (self.comm.rank +
                                          (pd0 - p) * self.stride_i[i])
                if pd0 != pd:
                    # Wrap around the box?
                    if self.periodic_i[i]:
                        # Yes:
                        self.disp_id[i, d] = -disp
                    else:
                        # No:
                        self.neighbor_id[i, d] = -1


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
