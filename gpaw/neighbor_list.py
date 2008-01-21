# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a neighbor-list class."""

import numpy as npy


class NeighborList:
    """Neighbor-list class."""
    def __init__(self, symbol_a, pos_ac, domain, cutoffs, drift=0.3):
        """Construct a ``NeighborList`` object.

        Construct a neighbor list object from a list of atomic numbers
        and a list of atomic positions.  The `cutoffs` argument is a
        list of (symbol, cutoff) tuples::

            [('H', 3.0), ('Au', 4.9)].

        
        """

        self.drift = drift
        self.cell_c = domain.cell_c
        self.symbol_a = symbol_a

        self.stuff = {}
        n = 0
        for symbol1, rcut1 in cutoffs:
            for symbol2, rcut2 in cutoffs[n:]:
                rcut = rcut1 + rcut2 + 2 * drift
                ncells = (rcut / self.cell_c + 0.5).astype(int)
                for i in (0, 1, 2):
                    if not domain.pbc_c[i]:
                        ncells[i] = 0
                self.stuff[(symbol1, symbol2)] = (rcut, ncells)
                if symbol1 != symbol2:
                    self.stuff[(symbol2, symbol1)] = (rcut, ncells)
            n += 1

        self.make_list(pos_ac)

    def neighbors(self, n):
        """Return a list of neighbors of atom number `n`.

        The minimum image convention is **not** used.  Therefore, an
        atom can be a neighbor several times - images in different unit
        cells!  A list of tuples is returned:

            [(m, offsets), ...]

        where `m` is the neighbor atom index and `offsets` is a list
        of unit cell offset vectors (one offset for each neighbor
        image).  **Notice that only neighbors with atom index `m <= n`
        are returned and an atom is always a neighbor to
        itself!!**."""
        
        return self.list[n]

    def update_list(self, pos_ac):
        """Make sure that the list is up to date.

        `UpdateList` must be called every time the positions change.
        If an atom has moved more than `drift`, MakeList() will be
        called, and a new list is generated.  The method returns `True`
        if a new list was build, otherwise `False` is returned."""
        
        # Check if any atom has moved more than drift:
        drift2 = self.drift**2
        for a, pos_c in enumerate(pos_ac):
            diff_c = pos_c - self.oldpos_ac[a]
            if npy.dot(diff_c, diff_c) > drift2:
                # Update list:
                self.make_list(pos_ac)
                return True
        # No update requred:
        return False

    def make_list(self, pos_ac):
        """Build the list."""
        self.list = []
        # Build the list:
        cell_c = self.cell_c
        for a1, pos1_c in enumerate(pos_ac):
            symbol1 = self.symbol_a[a1]
            neighbors1 = []
            for a2, pos2_c in enumerate(pos_ac[:a1 + 1]):
                symbol2 = self.symbol_a[a2]
                diff_c = pos2_c - pos1_c
                offset0 = npy.floor(diff_c / cell_c + 0.5) * cell_c
                diff_c -= offset0  # XXX only for periodic BC's!!!!
                offsets = []
                rcut, ncells = self.stuff[(symbol1, symbol2)]
                for n0 in range(-ncells[0], ncells[0] + 1):
                    for n1 in range(-ncells[1], ncells[1] + 1):
                        for n2 in range(-ncells[2], ncells[2] + 1):
                            offset = cell_c * (n0, n1, n2)
                            d_c = diff_c + offset
                            if npy.dot(d_c, d_c) < rcut**2:
                                offsets.append(offset)
                if offsets:
                    neighbors1.append((a2, npy.array(offsets) - offset0))
            self.list.append(neighbors1)
        self.oldpos_ac = pos_ac.copy()
