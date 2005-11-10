# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

import Numeric as num

from gridpaw.transrotation import rotate


class NeighborList:
    """Skin stuff ...


    The following code will print the distance vectors from atom
    number 27 to its neighbors::
    
        >>> atoms = ...
        >>> nblist = NeighborList(atoms, cutoffs=...)
        >>> nbs = nblist.Neighbors(27)
        >>> for n, offsets in nbs:
        ...     for offset in offsets:
        ...         print atoms[n].position + offset - atoms[27].position
        ...         

    
    """
    
    def __init__(self, numbers, positions, cell, bc, angle, cutoffs, drift):
        """NeighborList(atoms, cutoffs) -> neighbor list.

        Construct a neighbor list object from a list of atoms and some
        cutoffs. The `cutoffs` argument is a list of (symbol, cutoff)
        tuples::

            [('H', 3.0), ('Au', 4.9)].
        """

        self.drift = drift

        self.stuff = {}
        n = 0
        for Z1, rcut1 in cutoffs:
            for Z2, rcut2 in cutoffs[n:]:
                rcut = rcut1 + rcut2 + 2 * drift
                ncells = (rcut / cell + 0.5).astype(num.Int)
                for i in (0, 1, 2):
                    if not bc[i]:
                        ncells[i] = 0
                self.stuff[(Z1, Z2)] = (rcut, ncells)
                if Z1 != Z2:
                    self.stuff[(Z2, Z1)] = (rcut, ncells)
            n += 1

        self.cell = cell
        self.angle = angle
        self.numbers = numbers
        self.make_list(positions)

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

    def number_of_pairs(self):
        """Return the number of pairs.

        Note that all atoms are pairs with them selves!"""
        npairs = 0
        for neighbors in self.list:
            for n, offsets in neighbors:
                npairs += len(offsets)
        return npairs
    
    def update_list(self, positions):
        """Make sure that the list is up to date.

        `UpdateList` must be called every time the positions change.
        If an atom has moved more than `drift`, MakeList() will be
        called, and a new list is generated.  The method returns `True`
        if a new list was build, otherwise `False` is returned."""
        
        # Check if any atom has moved more than drift:
        drift2 = self.drift**2
        for a, position in enumerate(positions):
            diff = position - self.oldpositions[a]
            if num.dot(diff, diff) > drift2:
                # Update list:
                self.make_list(positions)
                return True
        # No update requred:
        return False

    def make_list(self, positions):
        """Build the list."""
        self.list = []
        # Using an O(N^2) method!!!!!!
        # Build the list:
        size = self.cell
        for a1, pos1 in enumerate(positions):
            Z1 = self.numbers[a1]
            neighbors1 = []
            for a2, pos2 in enumerate(positions[:a1 + 1]):
                Z2 = self.numbers[a2]
                diff = pos2 - pos1
                offset0 = num.floor(diff / size + 0.5) * size
                diff -= offset0
                if self.angle is not None:
                    r_c = pos2 - size / 2
                    rotate(diff, r_c, self.angle * offset0[0] / size[0])
                offsets = []
                rcut, ncells = self.stuff[(Z1, Z2)]
                for n0 in range(-ncells[0], ncells[0] + 1):
                    for n1 in range(-ncells[1], ncells[1] + 1):
                        for n2 in range(-ncells[2], ncells[2] + 1):
                            offset = size * (n0, n1, n2)
                            difference = diff + offset
                            if self.angle is not None:
                                rotate(difference, r_c, self.angle * n0)
                            if num.dot(difference, difference) < rcut**2:
                                offsets.append(offset)
                if offsets:
                    neighbors1.append((a2, num.array(offsets) - offset0))
            self.list.append(neighbors1)
        self.oldpositions = positions.copy()
